from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass

from .config import ApiServerConfig
from .model_runner import OcrPrediction, build_predictor


class QueueFullError(RuntimeError):
    pass


@dataclass(slots=True)
class PendingJob:
    image_bytes: bytes
    future: Future[OcrPrediction]


class OcrService:
    def __init__(self, config: ApiServerConfig) -> None:
        self.config = config
        self.start_time = time.monotonic()
        self.total_requests = 0
        self.success_count = 0
        self.failure_count = 0
        self.active_workers = 0
        self.server_name = config.server_name
        self.worker_count = config.resolved_worker_count()
        self.queue_capacity = config.resolved_queue_capacity()
        self._queue: queue.Queue[PendingJob | None] = queue.Queue(maxsize=self.queue_capacity)
        self._workers: list[threading.Thread] = []
        self._ready_events: list[threading.Event] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._init_error: str | None = None
        self._bootstrap_workers()

    def _bootstrap_workers(self) -> None:
        for worker_index in range(self.worker_count):
            ready = threading.Event()
            thread = threading.Thread(
                target=self._worker_main,
                args=(worker_index, ready),
                name=f"cas-ocr-api-worker-{worker_index}",
                daemon=True,
            )
            self._workers.append(thread)
            self._ready_events.append(ready)
            thread.start()

        for ready in self._ready_events:
            ready.wait()

    def _worker_main(self, worker_index: int, ready: threading.Event) -> None:
        try:
            predictor = build_predictor(self.config)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                if self._init_error is None:
                    self._init_error = f"worker {worker_index} failed to load models: {exc}"
            ready.set()
            return

        ready.set()
        while not self._stop_event.is_set():
            job = self._queue.get()
            if job is None:
                self._queue.task_done()
                break

            with self._lock:
                self.active_workers += 1
            try:
                result = predictor.predict(job.image_bytes)
                with self._lock:
                    self.success_count += 1
                job.future.set_result(result)
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self.failure_count += 1
                job.future.set_exception(exc)
            finally:
                with self._lock:
                    self.active_workers -= 1
                self._queue.task_done()

    @property
    def models_loaded(self) -> bool:
        return self._init_error is None

    @property
    def init_error(self) -> str | None:
        return self._init_error

    def submit(self, image_bytes: bytes) -> OcrPrediction:
        with self._lock:
            self.total_requests += 1

        if not self.models_loaded:
            with self._lock:
                self.failure_count += 1
            raise RuntimeError(self._init_error or "Models not loaded")

        future: Future[OcrPrediction] = Future()
        try:
            self._queue.put_nowait(PendingJob(image_bytes=image_bytes, future=future))
        except queue.Full as exc:
            with self._lock:
                self.failure_count += 1
            raise QueueFullError("Server overloaded") from exc
        return future.result()

    def health_payload(self) -> dict[str, object]:
        return {
            "status": "healthy" if self.models_loaded else "unhealthy",
            "modelsLoaded": self.models_loaded,
            "poolSize": self.worker_count,
            "serverName": self.server_name or None,
        }

    def status_payload(self) -> dict[str, object]:
        with self._lock:
            active_workers = self.active_workers
            total_requests = self.total_requests
            success_count = self.success_count
            failure_count = self.failure_count

        pending_requests = self._queue.qsize()
        if not self.models_loaded:
            status = "unavailable"
            availability_level = "unavailable"
            reason = self._init_error or "Models not loaded"
        elif pending_requests > self.queue_capacity // 2:
            status = "healthy"
            availability_level = "busy"
            reason = ""
        else:
            status = "healthy"
            availability_level = "available"
            reason = ""

        return {
            "status": status,
            "availabilityLevel": availability_level,
            "reason": reason,
            "modelsLoaded": self.models_loaded,
            "poolSize": self.worker_count,
            "queueCapacity": self.queue_capacity,
            "pendingRequests": pending_requests,
            "activeWorkers": active_workers,
            "totalRequests": total_requests,
            "successCount": success_count,
            "failureCount": failure_count,
            "uptimeSeconds": int(time.monotonic() - self.start_time),
            "serverName": self.server_name or None,
            "modelKind": self.config.model_kind,
        }

    def shutdown(self) -> None:
        self._stop_event.set()
        for _ in self._workers:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                self._queue.put(None)
        for worker in self._workers:
            worker.join(timeout=2.0)
