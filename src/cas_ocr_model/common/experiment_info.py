"""实验元信息收集：超参、时间戳、tracker 信息等。"""
from __future__ import annotations

from typing import Any


def collect_experiment_metadata(cfg: Any, accelerator: Any, start_utc: int, end_utc: int, best_epoch: int | None) -> dict[str, Any]:
    duration = max(0, int(end_utc - start_utc))
    meta: dict[str, Any] = {
        "training_start_utc": int(start_utc),
        "training_end_utc": int(end_utc),
        "training_duration_s": int(duration),
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
    }

    # hyperparameters 摘要
    try:
        hp = {
            "learning_rate": float(getattr(cfg.train, "learning_rate", None)),
            "per_device_batch_size": int(getattr(cfg.train, "per_device_batch_size", 0)),
            "epochs": int(getattr(cfg.train, "epochs", 0)),
            "weight_decay": float(getattr(cfg.train, "weight_decay", 0.0)),
            "grad_clip": float(getattr(cfg.train, "grad_clip", 0.0)),
            "seed": int(getattr(cfg.train, "seed", 0)),
        }
        meta["hyperparameters"] = hp
    except Exception:
        meta.setdefault("hyperparameters", {})

    # ddp / device 信息
    try:
        meta["ddp_world_size"] = int(getattr(accelerator, "num_processes", 1))
    except Exception:
        meta["ddp_world_size"] = None

    # tracker 信息 (如 wandb)
    try:
        trackers = getattr(accelerator, "trackers", None)
        if trackers:
            tracker_names = {t.name for t in trackers}
            if "wandb" in tracker_names:
                run = accelerator.get_tracker("wandb", unwrap=True)
                try:
                    meta.setdefault("tracker", {})
                    meta["tracker"]["wandb_run_id"] = getattr(run, "id", None)
                    # run has method or attribute to get url
                    meta["tracker"]["wandb_url"] = getattr(run, "get_url", lambda: getattr(run, "url", None))()
                except Exception:
                    pass
    except Exception:
        pass

    return meta
