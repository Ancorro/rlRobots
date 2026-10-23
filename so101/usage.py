"""Background sampler of CPU, RAM and GPU memory use while a run is going (every 0.5 s).

CPU and RAM are this process only. GPU numbers come from nvidia-smi and cover the whole card
(viewer, desktop and other apps included), since Windows does not report per-process GPU memory.
If PyTorch is loaded (Laya, SmolVLA), its own peak GPU memory is reported separately.
"""
import subprocess
import sys
import threading

import psutil


def _gpu():
    """(memory used MB, utilisation %) for the whole card, or None without an NVIDIA GPU."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        mem, util = out.splitlines()[0].split(",")
        return float(mem), float(util)
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


class Usage:
    def __init__(self, period=0.5):
        self.proc, self.period = psutil.Process(), period
        self.cpu, self.ram, self.gpu_mem, self.gpu_util = [], [], [], []
        self.cores = psutil.cpu_count()
        self._stop = threading.Event()
        self.proc.cpu_percent()  # first call only starts the measurement
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self._stop.wait(self.period):
            self.cpu.append(self.proc.cpu_percent() / self.cores)  # % of the whole CPU
            self.ram.append(self.proc.memory_info().rss / 1e9)
            g = _gpu()
            if g:
                self.gpu_mem.append(g[0] / 1e3)
                self.gpu_util.append(g[1])

    @staticmethod
    def low_ram(min_free_gb=2.0):
        """True when the whole PC is short of RAM: callers save and stop instead of freezing it."""
        return psutil.virtual_memory().available / 1e9 < min_free_gb

    def now(self):
        """One line for the viewer overlay."""
        if not self.cpu:
            return "measuring..."
        gpu = f"   GPU {self.gpu_mem[-1]:.1f} GB, {self.gpu_util[-1]:.0f}%" if self.gpu_mem else ""
        return f"CPU {self.cpu[-1]:.0f}%  RAM {self.ram[-1]:.1f} GB{gpu}"

    def summary(self):
        self._stop.set()
        s = {"cpu_percent_mean": _mean(self.cpu), "cpu_percent_peak": max(self.cpu, default=None),
             "ram_gb_peak": max(self.ram, default=None),
             "gpu_card_mem_gb_peak": max(self.gpu_mem, default=None), "gpu_util_percent_mean": _mean(self.gpu_util)}
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            s["torch_gpu_mem_gb_peak"] = torch.cuda.max_memory_reserved() / 1e9
        return {k: round(v, 2) if isinstance(v, float) else v for k, v in s.items()}


def _mean(xs):
    return sum(xs) / len(xs) if xs else None
