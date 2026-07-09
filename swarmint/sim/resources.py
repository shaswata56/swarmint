"""Resource governor for the multiprocess harness.

Caps how much of the machine the swarm may consume — default 60% of CPU cores
and available RAM (user directive), fully configurable via constructor args or
environment variables. The node count auto-scales to fit within this budget.

Config precedence (highest first):
  1. explicit constructor args
  2. environment: SWARMINT_CPU_FRACTION, SWARMINT_MEM_FRACTION,
     SWARMINT_MEM_PER_NODE_MB, SWARMINT_MAX_NODES
  3. defaults below

Memory source precedence: psutil (cross-platform) -> Windows ctypes
GlobalMemoryStatusEx -> a conservative assumed default. The harness never
*enforces* memory via the OS; it BUDGETS: it estimates per-process footprint
and refuses to launch more processes than the budget allows, logging what it
capped (no silent truncation — plan risk note).
"""

import ctypes
import os
from dataclasses import dataclass

DEFAULT_CPU_FRACTION = 0.60
DEFAULT_MEM_FRACTION = 0.60
DEFAULT_MEM_PER_NODE_MB = 120.0   # measured ballpark: numpy + asyncio + 2 UDP sockets/node
ASSUMED_TOTAL_MEM_GB = 8.0        # last-resort fallback if no memory source is available


def _env_float(name):
    v = os.environ.get(name)
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _env_int(name):
    v = os.environ.get(name)
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def available_memory_bytes():
    """(available, total) bytes. Available > total*fraction is the real cap."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.available, vm.total
    except Exception:
        pass
    if os.name == "nt":
        try:
            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullAvailPhys, stat.ullTotalPhys
        except Exception:
            pass
    total = int(ASSUMED_TOTAL_MEM_GB * 1e9)
    return total // 2, total  # assume ~half available


@dataclass
class ResourceBudget:
    cpu_fraction: float = None
    mem_fraction: float = None
    mem_per_node_mb: float = None
    max_nodes: int = None          # hard override; skips resource math entirely if set
    requested_nodes: int = None    # what the caller *wants*; may be capped down

    def __post_init__(self):
        self.cpu_fraction = (self.cpu_fraction if self.cpu_fraction is not None
                             else _env_float("SWARMINT_CPU_FRACTION") or DEFAULT_CPU_FRACTION)
        self.mem_fraction = (self.mem_fraction if self.mem_fraction is not None
                             else _env_float("SWARMINT_MEM_FRACTION") or DEFAULT_MEM_FRACTION)
        self.mem_per_node_mb = (self.mem_per_node_mb if self.mem_per_node_mb is not None
                                else _env_float("SWARMINT_MEM_PER_NODE_MB") or DEFAULT_MEM_PER_NODE_MB)
        if self.max_nodes is None:
            self.max_nodes = _env_int("SWARMINT_MAX_NODES")

    def cpu_cap(self) -> int:
        return max(1, int(os.cpu_count() * self.cpu_fraction))

    def mem_cap_nodes(self) -> int:
        avail, total = available_memory_bytes()
        budget_bytes = min(avail, total * self.mem_fraction)
        per_node = self.mem_per_node_mb * 1024 * 1024
        return max(1, int(budget_bytes / per_node))

    def resolve(self, requested_nodes: int) -> dict:
        """Return {"nodes": N, "reason": ...} — N capped to fit the budget."""
        if self.max_nodes is not None:
            n = min(requested_nodes, self.max_nodes)
            return {"nodes": n, "cpu_cap": self.max_nodes, "mem_cap": self.max_nodes,
                    "capped_by": "max_nodes-override" if n < requested_nodes else "none",
                    "requested": requested_nodes}
        cpu_cap = self.cpu_cap()
        mem_cap = self.mem_cap_nodes()
        hard_cap = min(cpu_cap, mem_cap)
        n = min(requested_nodes, hard_cap)
        capped_by = "none"
        if n < requested_nodes:
            capped_by = "cpu" if cpu_cap <= mem_cap else "memory"
        return {"nodes": n, "cpu_cap": cpu_cap, "mem_cap": mem_cap,
                "capped_by": capped_by, "requested": requested_nodes}

    def describe(self) -> str:
        avail, total = available_memory_bytes()
        return (f"budget: cpu={self.cpu_fraction:.0%} of {os.cpu_count()} cores -> {self.cpu_cap()} procs; "
                f"mem={self.mem_fraction:.0%} of {total/1e9:.1f}GB (avail {avail/1e9:.1f}GB) "
                f"@ {self.mem_per_node_mb:.0f}MB/node -> {self.mem_cap_nodes()} procs"
                + (f"; max_nodes override={self.max_nodes}" if self.max_nodes else ""))
