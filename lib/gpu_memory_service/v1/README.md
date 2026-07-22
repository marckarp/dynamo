# Snapshot-only GPU Memory Service V1

V1 has one fail-stop contract: Dynamo Snapshot/CRIU restores the engine
process and all Python, `TensorImpl`, `StorageImpl`, alias, and module topology;
an unchanged sidecar preserves exact parameter allocation IDs and physical
backing on the same GPU.

Torch 2.11 uses two dedicated CUDA `MemPool`s. Parameter allocations are
writable while loading, become read-only as one complete set before the
artifact is published, and preserve their backing across capture. Private
allocations are discarded during sleep and receive fresh read-write backing at
the CRIU-preserved virtual addresses after restore.

Capture destroys the dedicated pools so Torch evicts inactive segments.
Registered CUDA `Parameter` storages are deduplicated by `UntypedStorage`
identity, resolved by complete range containment to one parameter mapping, and
then deduplicated by exact containing allocation. Sleep/wake process each
canonical allocation mapping once; V1 never serializes or reconstructs tensor
topology.

## Commands

Start the non-checkpointed sidecar:

```bash
gms-v1-server --device 0 --socket-path /gms/gms-v1.sock
```

Run the engine inside a Dynamo Snapshot target container:

```bash
gms-v1-e2e \
  --device 0 \
  --socket-path /gms/gms-v1.sock \
  --artifact-id "${CHECKPOINT_ID}" \
  --standby-marker /state/captured
```

## DRA + Snapshot deployment test

The collected test creates one DRA `ResourceClaimTemplate`, one two-container
Pod, and one `PodSnapshot`. It snapshots only `engine`, restores that container
in place, proves the `gms-server` container never restarted, and machine-checks
inference, object/storage/data-pointer identity, stable parameter allocation
IDs, and fresh private backing.

It requires explicit cluster placement and never evicts workloads:

```bash
KUBE_CONTEXT=<context> \
NAMESPACE=<namespace> \
NODE=<gpu-node> \
IMAGE=<torch-2.11-dynamo-image> \
CHECKPOINT_PVC=<snapshot-pvc> \
lib/gpu_memory_service/v1/deploy/run.sh
```

Optional variables are `CHECKPOINT_PATH` (default `/checkpoints`),
`DEVICE_CLASS` (default `gpu.nvidia.com`), and `RUNTIME_CLASS` (default
`nvidia`). The test deletes only its uniquely named resources.
