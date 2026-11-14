"""Stronger admission after confirmed intermittent foreign GPU contexts.

Keep the same shared research lock and strict contamination monitoring; require
120s continuously empty compute-app observations before starting a fresh run.
No external process is stopped or changed.
"""
from contextlib import contextmanager
import time
from .gpu_lease import lease_when_idle,apps


@contextmanager
def stable_lease(devices):
    while True:
        with lease_when_idle(devices) as selected:
            ready=True
            for _ in range(120):
                if any(gpu in selected for gpu,pid in apps()):
                    ready=False;break
                time.sleep(1)
            if not ready:continue
            yield selected
            return
