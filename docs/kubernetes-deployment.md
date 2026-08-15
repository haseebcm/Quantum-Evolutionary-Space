# Kubernetes deployment

This document shows a reference Kubernetes deployment for the QES runtime
worker pool (the same `qes.worker` service described in
[`deployment-patterns.md`](deployment-patterns.md#5-containerized-deployment-pattern)
and packaged by the repository's `Dockerfile`).

## Reference manifests

Save the following as `k8s/qes-worker.yaml` (not committed to this repo, since
the right namespace/image registry/secrets vary per cluster) and adapt the
placeholders (`<registry>`, secrets, resource limits) to your environment.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qes-worker
  labels:
    app: qes-worker
spec:
  replicas: 3
  selector:
    matchLabels:
      app: qes-worker
  template:
    metadata:
      labels:
        app: qes-worker
    spec:
      containers:
        - name: qes-worker
          image: <registry>/qes-runtime:0.4.0
          args: ["--worker-count", "4"]
          env:
            - name: QES_LOG_LEVEL
              value: "INFO"
            - name: QES_RUNTIME_STATE_DIR
              value: "/app/state"
            - name: QES_WORKER_CYCLES
              value: "0"
            - name: QES_WORKER_INTERVAL_SECONDS
              value: "10"
            # For a shared, multi-replica durable store, point the worker at
            # Redis/Postgres instead of the default SQLite file store (see
            # RedisRuntimeStore / PostgresRuntimeStore) via your own env
            # variables and a small wrapper around src/qes/worker.py.
            - name: QES_REDIS_URL
              valueFrom:
                secretKeyRef:
                  name: qes-runtime-secrets
                  key: redis-url
          resources:
            requests:
              cpu: "250m"
              memory: "256Mi"
            limits:
              cpu: "1"
              memory: "512Mi"
          volumeMounts:
            - name: qes-state
              mountPath: /app/state
          readinessProbe:
            exec:
              command: ["python", "-c", "import qes; print('ok')"]
            initialDelaySeconds: 5
            periodSeconds: 30
          livenessProbe:
            exec:
              command: ["python", "-c", "import qes; print('ok')"]
            initialDelaySeconds: 10
            periodSeconds: 60
      volumes:
        - name: qes-state
          # A single PVC across replicas only makes sense with SQLite if
          # replicas=1; for replicas > 1, use RedisRuntimeStore/
          # PostgresRuntimeStore and drop this volume (or scope it per-pod
          # with an emptyDir if only transient local state is needed).
          persistentVolumeClaim:
            claimName: qes-runtime-state
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: qes-runtime-state
spec:
  accessModes: ["ReadWriteOnce"]
  resources:
    requests:
      storage: 1Gi
```

## Scaling guidance

- **Single-node durability (`SQLiteRuntimeStore`, the default)**: keep
  `replicas: 1`. SQLite is not safe for concurrent multi-pod writers even
  with a shared volume.
- **Horizontal scaling (`replicas > 1`)**: switch the store to
  `RedisRuntimeStore` or `PostgresRuntimeStore` (see
  [`deployment-patterns.md`](deployment-patterns.md#6-durable-multi-node-storage-backends))
  so every replica observes the same job state. Point workers at a managed
  Redis/Postgres instance (or an in-cluster `Service`) via environment
  variables/secrets rather than a shared local volume.
- **Access control**: if the runtime is reachable by more than the worker
  pods themselves (e.g. an API layer submits jobs), configure an
  `ApiKeyGuard` and inject the key(s) via a Kubernetes `Secret`, not directly
  in the manifest.
- **Autoscaling**: a `HorizontalPodAutoscaler` targeting CPU or a custom
  metric (e.g. queue depth exposed via the runtime's `telemetry()`) can scale
  `qes-worker` replicas up and down; this requires a shared durable store as
  above.

## Local validation without a cluster

Before deploying, validate the container image and worker entrypoint
locally with `docker compose` (see the repository's `docker-compose.yml`):

```bash
docker compose up --build
```

This exercises the same `Dockerfile`/`ENTRYPOINT` that the Kubernetes
manifest above references, so a working `docker compose up` is a good
pre-flight check before rolling the image out to a cluster.
