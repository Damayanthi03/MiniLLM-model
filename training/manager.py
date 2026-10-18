"""Runs training in a background thread so the web page stays responsive.

One shared manager exists per server process, so only one training job can run
at a time - a second "Start" click is refused instead of starting a duplicate.
"""
from __future__ import annotations

import copy
import threading
import time
import traceback

import config
from training.train import train


class TrainingManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._state = self._fresh_state()

    @staticmethod
    def _fresh_state() -> dict:
        return {
            "status": "idle",            # idle | running | finished | stopped | error
            "step": 0,
            "max_steps": 0,
            "batch_loss": None,
            "train_loss": None,
            "val_loss": None,
            "lr": None,
            "elapsed": 0.0,
            "started_at": None,
            "history": {"step": [], "train_loss": [], "val_loss": []},
            "result": None,
            "error": None,
            "traceback": None,
        }

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, model_cfg, train_cfg, device=None, checkpoint_path=None, text=None, log_path=config.TRAIN_LOG_PATH) -> bool:
        """Start a job.  Returns False (and does nothing) if one is already running."""
        with self._lock:
            if self.is_running():
                return False
            self._stop.clear()
            self._state = self._fresh_state()
            self._state.update(status="running", max_steps=train_cfg.max_steps, started_at=time.time())
            self._thread = threading.Thread(
                target=self._run,
                args=(model_cfg, train_cfg, device, checkpoint_path, text, log_path),
                name="minilm-training",
                daemon=True,
            )
            self._thread.start()
            return True

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def snapshot(self) -> dict:
        with self._lock:
            snap = copy.deepcopy(self._state)
        if snap["status"] == "running" and snap["started_at"]:
            snap["elapsed"] = time.time() - snap["started_at"]
        return snap

    # -- internals -----------------------------------------------------------
    def _on_progress(self, ev: dict) -> None:
        with self._lock:
            s = self._state
            s["step"] = ev["step"]
            s["max_steps"] = ev.get("max_steps", s["max_steps"])
            s["elapsed"] = ev.get("elapsed", s["elapsed"])
            s["lr"] = ev.get("lr", s["lr"])
            if ev.get("batch_loss") is not None:
                s["batch_loss"] = ev["batch_loss"]
            if "val_loss" in ev:
                s["train_loss"] = ev["train_loss"]
                s["val_loss"] = ev["val_loss"]
                s["history"]["step"].append(ev["step"])
                s["history"]["train_loss"].append(ev["train_loss"])
                s["history"]["val_loss"].append(ev["val_loss"])

    def _run(self, model_cfg, train_cfg, device, checkpoint_path, text, log_path) -> None:
        try:
            result = train(
                model_cfg, train_cfg,
                checkpoint_path=checkpoint_path,
                progress_callback=self._on_progress,
                stop_event=self._stop,
                device=device,
                text=text,
                log_path=log_path,
            )
            with self._lock:
                self._state["result"] = result
                self._state["status"] = "finished" if result["completed"] else "stopped"
        except Exception as exc:  # show the real error in the UI instead of crashing
            with self._lock:
                self._state["status"] = "error"
                self._state["error"] = f"{type(exc).__name__}: {exc}"
                self._state["traceback"] = traceback.format_exc()
