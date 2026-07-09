// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pod discovery for standalone (raw inference-engine) mode, driven by the `InferencePool`.
//!
//! The pod label selector and HTTP target port come from the GAIE
//! [`InferencePool`](crate::inference_pool) this EPP backs — the same object the
//! gateway routes to — so EPP and gateway can never disagree about pool
//! membership. To pick up live selector/target-port edits without restarting any
//! watch, the reflector watches **all** pods in the namespace and filters them
//! in memory against the current [`PoolState`]; the pool watch just swaps the
//! filter.
//!
//! Pods are `Ready`-filtered (and excluded once terminating), so in-flight
//! rollouts and crash-looping pods receive no traffic.
//!
//! `worker_id = hash_pod_name(pod_name)`, so the IDs produced here line up with
//! whatever consumes them (the topology adapter and selector catalog).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use dynamo_runtime::discovery::hash_pod_name;
use k8s_openapi::api::core::v1::Pod;
use tokio::sync::watch;

use crate::epp_standalone_config::EppStandaloneConfig;
use crate::inference_pool::{PoolState, spawn_pool_watch};

/// A discovered, `Ready` raw inference engine worker normalized for selector registration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawWorker {
    /// Stable hash of the pod name; the selector catalog key.
    pub worker_id: u64,
    /// Kubernetes pod name.
    pub pod_name: String,
    /// Pod IP.
    pub pod_ip: String,
    /// OpenAI HTTP inference endpoint, `http://<ip>:<target_port>`.
    pub http_endpoint: String,
    /// Inference engine KV-event ZMQ PUB endpoint, `tcp://<ip>:<kv_event_port>`.
    pub kv_events_endpoint: String,
    /// Optional ZMQ REQ endpoint for live-stream gap replay.
    pub replay_endpoint: Option<String>,
    /// Routing identity fed to the selector's `stable_routing_id`. Currently the
    /// pod name, so it is NOT yet restart-stable: a Deployment pod restart yields
    /// a new name and a new identity. A truly stable source (StatefulSet ordinal
    /// or a pod label) is a follow-up.
    pub stable_routing_id: String,
}

/// Immutable snapshot of the `Ready`, pool-selected workers, rebuilt whenever the
/// pod set or the pool changes. Holds the materialized worker list plus a
/// `worker_id -> "ip:port"` endpoint index so request-path reads (notably
/// [`PodDiscovery::resolve_endpoint`]) are O(1) lookups that never construct a
/// [`RawWorker`] or clone the [`PoolState`].
#[derive(Debug, Default)]
struct Snapshot {
    workers: Vec<RawWorker>,
    endpoints: HashMap<u64, String>,
}

/// Lock-free view over the `Ready` raw inference engine pods selected by the EPP's
/// `InferencePool`. Reads never touch the Kubernetes API; they read a cached
/// [`Snapshot`] that a background task rebuilds on pod/pool changes.
#[derive(Clone)]
pub struct PodDiscovery {
    snapshot: watch::Receiver<Arc<Snapshot>>,
    changes: watch::Receiver<u64>,
}

impl PodDiscovery {
    /// Start the InferencePool watch and a namespace-wide pod reflector. Returns
    /// a *live* readiness flag that is `true` only while the pod cache has synced
    /// (initial LIST done) **and** the `InferencePool` is resolved. It clears back
    /// to `false` if the pool is later deleted or edited into an unsupported spec
    /// (so nothing is routable), and recovers when both are healthy again — this
    /// is the gRPC health SERVING signal, so it must not latch true.
    pub async fn spawn(cfg: &EppStandaloneConfig) -> Result<(Self, Arc<AtomicBool>)> {
        use futures::StreamExt;
        use kube::{Api, Client, runtime::WatchStreamExt, runtime::reflector, runtime::watcher};

        let client = Client::try_default().await?;
        let namespace = cfg.namespace.clone();

        let (pool_rx, _pool_task) = spawn_pool_watch(
            client.clone(),
            namespace.clone(),
            cfg.inference_pool_name.clone(),
        )
        .await?;

        // Namespace-wide pod watch; membership is decided in memory by the pool
        // selector so selector edits never require re-spawning this watch.
        let pods: Api<Pod> = Api::namespaced(client, &namespace);
        let writer = reflector::store::Writer::default();
        let store = writer.as_reader();
        let ready = Arc::new(AtomicBool::new(false));
        // `default_backoff()` lets the watcher own retry/backoff (exponential +
        // jitter, capped) on watch errors; the error arm below just logs, so
        // persistent failures (e.g. RBAC, API-server hiccup) don't hot-loop.
        let reflect = reflector::reflector(
            writer,
            watcher(pods, watcher::Config::default()).default_backoff(),
        );

        let (changes_tx, changes_rx) = watch::channel(0u64);

        let kv_event_port = cfg.kv_event_port;
        let replay_port = cfg.replay_port;

        // Cached snapshot of the ready, pool-selected workers. Rebuilt by the task
        // below (before it bumps the change generation), so any consumer that wakes
        // on a generation bump observes a snapshot already consistent with the
        // store/pool that triggered it.
        let (snapshot_tx, snapshot_rx) = watch::channel(Arc::new(build_snapshot(
            &store,
            pool_rx.borrow().as_ref(),
            kv_event_port,
            replay_port,
        )));

        tracing::info!(
            namespace = %namespace,
            pool = %cfg.inference_pool_name,
            kv_event_port = cfg.kv_event_port,
            "Starting namespace pod reflector for standalone mode"
        );

        // A single task owns both wake sources so snapshot builds are serialized:
        // on either a pod event or a pool change it rebuilds once from the latest
        // store + latest pool and publishes in order. Two independent producers
        // could each read state, build, and push to the watch channel in arbitrary
        // order, letting a stale build overwrite a fresher one (notably during a
        // pool relist).
        let store_task = store.clone();
        let ready_task = ready.clone();
        tokio::spawn(async move {
            let mut pool_rx = pool_rx;
            tokio::pin!(reflect);
            let mut generation = 0u64;
            // The pod cache is "synced" once the reflector's initial LIST lands
            // (InitDone); readiness stays gated on this AND pool presence below.
            let mut pod_synced = false;
            loop {
                tokio::select! {
                    ev = reflect.next() => match ev {
                        None => {
                            tracing::warn!("Inference engine pod reflector stream ended unexpectedly");
                            break;
                        }
                        // During a relist the reflector emits Init + one InitApply
                        // per pod + InitDone (n+2 events). Rebuilding on each is
                        // quadratic, so skip the per-object relist events: the store
                        // is already consistent at InitDone, and Apply/Delete are
                        // single-object deltas. (Errors don't change the store.)
                        Some(Ok(watcher::Event::Init | watcher::Event::InitApply(_))) => continue,
                        Some(Ok(watcher::Event::InitDone)) => pod_synced = true,
                        Some(Ok(_)) => {}
                        Some(Err(e)) => {
                            tracing::warn!(error = %e, "Pod reflector watch error; retrying");
                            continue;
                        }
                    },
                    changed = pool_rx.changed() => {
                        if changed.is_err() {
                            tracing::warn!("InferencePool watch ended");
                            break;
                        }
                    }
                }
                // Rebuild the snapshot and recompute readiness under a single pool
                // borrow. Readiness is live: it drops to false whenever the pool is
                // absent/invalid (empty snapshot) and recovers once both are healthy.
                let (snap, is_ready) = {
                    let pool = pool_rx.borrow();
                    let snap =
                        build_snapshot(&store_task, pool.as_ref(), kv_event_port, replay_port);
                    (snap, pod_synced && pool.is_some())
                };
                ready_task.store(is_ready, Ordering::Release);
                let _ = snapshot_tx.send(Arc::new(snap));
                generation = generation.wrapping_add(1);
                let _ = changes_tx.send(generation);
            }
            // Either watch stream ended: the producer is gone and can no longer
            // refresh discovery, so stop advertising readiness.
            ready_task.store(false, Ordering::Release);
        });

        Ok((
            Self {
                snapshot: snapshot_rx,
                changes: changes_rx,
            },
            ready,
        ))
    }

    /// All currently `Ready` workers selected by the pool, normalized for
    /// selector registration. Empty until the `InferencePool` has resolved.
    /// Reads the cached snapshot; no per-call filtering or API access.
    pub fn ready_workers(&self) -> Vec<RawWorker> {
        self.snapshot.borrow().workers.clone()
    }

    /// Worker IDs of all currently `Ready`, pool-selected workers. Reads the
    /// cached snapshot's endpoint index, so it stays consistent with
    /// [`Self::resolve_endpoint`].
    pub fn ready_worker_ids(&self) -> HashSet<u64> {
        self.snapshot.borrow().endpoints.keys().copied().collect()
    }

    /// Resolve a `worker_id` to its current `ip:port` HTTP endpoint, if the pod
    /// is still `Ready` and pool-selected. On the request hot path: an O(1)
    /// lookup into the cached endpoint index — no `RawWorker` materialization,
    /// no `PoolState` clone, no pod scan.
    pub fn resolve_endpoint(&self, worker_id: u64) -> Option<String> {
        self.snapshot.borrow().endpoints.get(&worker_id).cloned()
    }

    /// Retain the `worker_ids` whose current `ip:port` endpoint satisfies `pred`,
    /// under a **single** snapshot borrow and **without cloning** any endpoint
    /// (`pred` borrows it). Used on the subset-routing path so membership testing
    /// doesn't allocate a throwaway `String` per candidate. Unknown workers are
    /// dropped.
    pub fn filter_workers_by_endpoint(
        &self,
        worker_ids: &HashSet<u64>,
        pred: impl Fn(&str) -> bool,
    ) -> HashSet<u64> {
        let snapshot = self.snapshot.borrow();
        worker_ids
            .iter()
            .copied()
            .filter(|worker_id| {
                snapshot
                    .endpoints
                    .get(worker_id)
                    .is_some_and(|endpoint| pred(endpoint.as_str()))
            })
            .collect()
    }

    /// Subscribe to change notifications (a generation counter) bumped on pod or
    /// pool changes, so a reconciler can re-sync.
    pub fn subscribe_changes(&self) -> watch::Receiver<u64> {
        self.changes.clone()
    }
}

/// Return `true` iff the pod is `Ready` and not terminating. Mirrors llm-d's
/// `IsPodReady`: a pod with a deletion timestamp is excluded even if it still
/// reports `Ready=True`, so draining pods stop receiving traffic promptly.
fn pod_is_ready(pod: &Pod) -> bool {
    if pod.metadata.deletion_timestamp.is_some() {
        return false;
    }
    pod.status
        .as_ref()
        .and_then(|s| s.conditions.as_ref())
        .map(|conds| {
            conds
                .iter()
                .any(|c| c.type_ == "Ready" && c.status == "True")
        })
        .unwrap_or(false)
}

/// Return `true` iff the pod carries every `match_labels` key with the equal
/// value (equality-based selector, matching `InferencePool.spec.selector`).
fn pod_matches(pod: &Pod, match_labels: &BTreeMap<String, String>) -> bool {
    let Some(labels) = pod.metadata.labels.as_ref() else {
        return match_labels.is_empty();
    };
    match_labels
        .iter()
        .all(|(k, v)| labels.get(k).map(|pv| pv == v).unwrap_or(false))
}

fn strip_scheme(endpoint: &str) -> &str {
    endpoint
        .strip_prefix("http://")
        .or_else(|| endpoint.strip_prefix("https://"))
        .unwrap_or(endpoint)
}

/// Build the cached [`Snapshot`] from the current pod store and pool selector.
/// Empty until the `InferencePool` has resolved. Pure function — unit-testable.
fn build_snapshot(
    store: &kube::runtime::reflector::Store<Pod>,
    pool: Option<&PoolState>,
    kv_event_port: u16,
    replay_port: Option<u16>,
) -> Snapshot {
    let Some(pool) = pool else {
        return Snapshot::default();
    };
    let mut workers = Vec::new();
    let mut endpoints = HashMap::new();
    for pod in store.state().iter() {
        if let Some(worker) = raw_worker_from_pod(pod, pool, kv_event_port, replay_port) {
            endpoints.insert(
                worker.worker_id,
                strip_scheme(&worker.http_endpoint).to_string(),
            );
            workers.push(worker);
        }
    }
    Snapshot { workers, endpoints }
}

/// Build a [`RawWorker`] from a pod, or `None` if it is not `Ready`, not
/// pool-selected, or lacks an IP/name. Pure function — unit-testable.
fn raw_worker_from_pod(
    pod: &Pod,
    pool: &PoolState,
    kv_event_port: u16,
    replay_port: Option<u16>,
) -> Option<RawWorker> {
    if !pod_is_ready(pod) || !pod_matches(pod, &pool.match_labels) {
        return None;
    }
    let pod_name = pod.metadata.name.as_deref()?;
    let pod_ip = pod.status.as_ref()?.pod_ip.as_deref()?;
    // A Pod IP is always an IP literal (never a hostname). Parse it so each
    // host/port pair is rendered via `SocketAddr`, which brackets IPv6 as
    // `[fd00::10]:8000`; a bare IPv6 literal (`fd00::10:8000`) is ambiguous, as
    // the trailing group can't be told apart from the port. IPv4 is unchanged.
    // An empty or malformed IP fails to parse and skips the pod.
    let ip: IpAddr = pod_ip.parse().ok()?;

    Some(RawWorker {
        worker_id: hash_pod_name(pod_name),
        pod_name: pod_name.to_string(),
        pod_ip: pod_ip.to_string(),
        http_endpoint: format!("http://{}", SocketAddr::new(ip, pool.target_port)),
        kv_events_endpoint: format!("tcp://{}", SocketAddr::new(ip, kv_event_port)),
        replay_endpoint: replay_port.map(|p| format!("tcp://{}", SocketAddr::new(ip, p))),
        stable_routing_id: pod_name.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use k8s_openapi::api::core::v1::{PodCondition, PodStatus};
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::Time;
    use kube::api::ObjectMeta;

    fn pool() -> PoolState {
        PoolState {
            match_labels: BTreeMap::from([("app".to_string(), "vllm-qwen".to_string())]),
            target_port: 8000,
        }
    }

    fn pod(name: &str, ip: Option<&str>, ready: Option<bool>, labels: &[(&str, &str)]) -> Pod {
        let conditions = ready.map(|r| {
            vec![PodCondition {
                type_: "Ready".to_string(),
                status: if r { "True" } else { "False" }.to_string(),
                ..Default::default()
            }]
        });
        let label_map = labels
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Pod {
            metadata: ObjectMeta {
                name: Some(name.to_string()),
                labels: Some(label_map),
                ..Default::default()
            },
            status: Some(PodStatus {
                pod_ip: ip.map(|s| s.to_string()),
                conditions,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn ready_selected_pod_maps_to_worker() {
        let w = raw_worker_from_pod(
            &pod(
                "vllm-0",
                Some("10.0.0.1"),
                Some(true),
                &[("app", "vllm-qwen")],
            ),
            &pool(),
            5557,
            Some(5560),
        )
        .expect("ready, selected pod should map");
        assert_eq!(w.worker_id, hash_pod_name("vllm-0"));
        assert_eq!(w.http_endpoint, "http://10.0.0.1:8000");
        assert_eq!(w.kv_events_endpoint, "tcp://10.0.0.1:5557");
        assert_eq!(w.replay_endpoint.as_deref(), Some("tcp://10.0.0.1:5560"));
    }

    #[test]
    fn ipv6_pod_ip_is_bracketed_in_all_endpoints() {
        let w = raw_worker_from_pod(
            &pod(
                "vllm-0",
                Some("fd00::10"),
                Some(true),
                &[("app", "vllm-qwen")],
            ),
            &pool(),
            5557,
            Some(5560),
        )
        .expect("ready, selected IPv6 pod should map");
        // SocketAddr brackets the IPv6 host so host and port are unambiguous.
        assert_eq!(w.http_endpoint, "http://[fd00::10]:8000");
        assert_eq!(w.kv_events_endpoint, "tcp://[fd00::10]:5557");
        assert_eq!(w.replay_endpoint.as_deref(), Some("tcp://[fd00::10]:5560"));
    }

    #[test]
    fn malformed_pod_ip_is_skipped() {
        assert!(
            raw_worker_from_pod(
                &pod(
                    "vllm-0",
                    Some("not-an-ip"),
                    Some(true),
                    &[("app", "vllm-qwen")]
                ),
                &pool(),
                5557,
                None,
            )
            .is_none()
        );
    }

    #[test]
    fn pod_not_matching_selector_is_skipped() {
        assert!(
            raw_worker_from_pod(
                &pod(
                    "other-0",
                    Some("10.0.0.1"),
                    Some(true),
                    &[("app", "something-else")]
                ),
                &pool(),
                5557,
                None,
            )
            .is_none()
        );
    }

    #[test]
    fn not_ready_pod_is_skipped() {
        assert!(
            raw_worker_from_pod(
                &pod(
                    "vllm-0",
                    Some("10.0.0.1"),
                    Some(false),
                    &[("app", "vllm-qwen")]
                ),
                &pool(),
                5557,
                None,
            )
            .is_none()
        );
    }

    #[test]
    fn terminating_pod_is_skipped() {
        let mut p = pod(
            "vllm-0",
            Some("10.0.0.1"),
            Some(true),
            &[("app", "vllm-qwen")],
        );
        p.metadata.deletion_timestamp = Some(Time(k8s_openapi::chrono::Utc::now()));
        assert!(raw_worker_from_pod(&p, &pool(), 5557, None).is_none());
    }

    #[test]
    fn pod_without_ip_is_skipped() {
        assert!(
            raw_worker_from_pod(
                &pod("vllm-0", None, Some(true), &[("app", "vllm-qwen")]),
                &pool(),
                5557,
                None,
            )
            .is_none()
        );
    }

    fn store_from_pods(pods: Vec<Pod>) -> kube::runtime::reflector::Store<Pod> {
        use kube::runtime::watcher;
        let mut writer = kube::runtime::reflector::store::Writer::<Pod>::default();
        let store = writer.as_reader();
        writer.apply_watcher_event(&watcher::Event::Init);
        for p in pods {
            writer.apply_watcher_event(&watcher::Event::InitApply(p));
        }
        writer.apply_watcher_event(&watcher::Event::InitDone);
        store
    }

    #[test]
    fn build_snapshot_indexes_only_ready_selected_pods() {
        let store = store_from_pods(vec![
            pod(
                "vllm-0",
                Some("10.0.0.1"),
                Some(true),
                &[("app", "vllm-qwen")],
            ),
            pod(
                "vllm-1",
                Some("10.0.0.2"),
                Some(false),
                &[("app", "vllm-qwen")],
            ),
            pod("other-0", Some("10.0.0.3"), Some(true), &[("app", "nope")]),
        ]);

        let snap = build_snapshot(&store, Some(&pool()), 5557, Some(5560));

        // Only the ready, correctly-labeled pod is materialized.
        assert_eq!(snap.workers.len(), 1);
        let id = hash_pod_name("vllm-0");
        assert_eq!(snap.workers[0].worker_id, id);
        // Endpoint index is keyed by worker_id and carries a scheme-less ip:port.
        assert_eq!(snap.endpoints.len(), 1);
        assert_eq!(
            snap.endpoints.get(&id).map(String::as_str),
            Some("10.0.0.1:8000")
        );
    }

    #[test]
    fn build_snapshot_is_empty_without_pool() {
        let store = store_from_pods(vec![pod(
            "vllm-0",
            Some("10.0.0.1"),
            Some(true),
            &[("app", "vllm-qwen")],
        )]);
        let snap = build_snapshot(&store, None, 5557, None);
        assert!(snap.workers.is_empty());
        assert!(snap.endpoints.is_empty());
    }

    /// Build a `PodDiscovery` over a fixed endpoint index (no cluster) so we can
    /// unit-test the borrow-based subset filter. Dropping the senders is fine —
    /// `watch::Receiver::borrow()` still returns the last value.
    fn discovery_with_endpoints(endpoints: HashMap<u64, String>) -> PodDiscovery {
        let (_, snapshot) = watch::channel(Arc::new(Snapshot {
            endpoints,
            ..Default::default()
        }));
        let (_, changes) = watch::channel(0u64);
        PodDiscovery { snapshot, changes }
    }

    #[test]
    fn filter_workers_by_endpoint_matches_without_cloning() {
        let discovery = discovery_with_endpoints(HashMap::from([
            (1u64, "10.0.0.1:8000".to_string()),
            (2u64, "10.0.0.2:8000".to_string()),
            (3u64, "10.0.0.3:8000".to_string()),
        ]));
        let allowed: HashSet<u64> = [1, 2, 3].into_iter().collect();

        // Predicate borrows the endpoint; only worker 2 matches.
        let filtered =
            discovery.filter_workers_by_endpoint(&allowed, |endpoint| endpoint == "10.0.0.2:8000");
        assert_eq!(filtered, HashSet::from([2]));

        // A worker id with no endpoint in the snapshot is dropped.
        let allowed_with_unknown: HashSet<u64> = [1, 99].into_iter().collect();
        let filtered = discovery.filter_workers_by_endpoint(&allowed_with_unknown, |_| true);
        assert_eq!(filtered, HashSet::from([1]));
    }
}
