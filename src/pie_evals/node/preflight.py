"""Machine hygiene around a measurement. Everything is best-effort and
string-valued: the goal is to *record* the state and to refuse to read a
number taken on a machine that was not in the expected state.

Apple Silicon: thermal pressure, battery vs AC, low power mode and a second
display all change the GPU clock, and another inference process (the resident
Ollama app with a model loaded, another runner's pie job) shares the GPU. All
of it is captured before and after each model.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.request
from typing import Any

ENGINE_PROCESSES = ("pie", "ollama", "llama-server", "mlx_lm.server")


def _run(argv: list[str], timeout: float = 20.0) -> str | None:
    if shutil.which(argv[0]) is None:
        return None
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if p.returncode != 0:
        return None
    return p.stdout


def engine_processes(exclude: set[int] | None = None) -> list[str]:
    out = _run(["ps", "-axo", "pid=,comm="]) or ""
    found = []
    skip = exclude or set()
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit() or int(parts[0]) in skip:
            continue
        if os.path.basename(parts[1]) in ENGINE_PROCESSES:
            found.append(f"{os.path.basename(parts[1])}:{parts[0]}")
    return sorted(found)


def ollama_loaded(host: str | None = None) -> list[str]:
    url = (host or os.environ.get("OLLAMA_HOST") or "127.0.0.1:11434").rstrip("/")
    url = url if url.startswith("http") else f"http://{url}"
    try:
        with urllib.request.urlopen(f"{url}/api/ps", timeout=3) as r:
            return sorted(m.get("name", "") for m in json.loads(r.read()).get("models", []))
    except (OSError, ValueError):
        return []


# ---- macOS ----------------------------------------------------------------------


def _parse_pmset_therm(text: str | None) -> dict[str, str]:
    """``pmset -g therm``: 'CPU_Scheduler_Limit = 100', 'CPU_Available_CPUs = 10',
    'CPU_Speed_Limit = 100', and on notebooks a line about thermal warning level."""
    d: dict[str, str] = {}
    if not text:
        return d
    for line in text.splitlines():
        m = re.match(r"\s*(CPU_\w+)\s*=\s*(\d+)", line)
        if m:
            d[m.group(1).lower()] = m.group(2)
    lower = text.lower()
    if "thermal warning level" in lower:
        m = re.search(r"thermal warning level\s*[:=]?\s*(\d+)", lower)
        d["thermal_warning_level"] = m.group(1) if m else "unknown"
    limit = d.get("cpu_speed_limit")
    sched = d.get("cpu_scheduler_limit")
    warned = (
        (limit is not None and limit != "100")
        or (sched is not None and sched != "100")
        or (d.get("thermal_warning_level") not in (None, "0", "unknown"))
    )
    d["thermal_warning"] = "yes" if warned else "no"
    return d


def _parse_pmset_batt(text: str | None) -> dict[str, str]:
    d: dict[str, str] = {}
    if not text:
        return d
    lower = text.lower()
    if "ac power" in lower:
        d["power_source"] = "ac"
    elif "battery power" in lower:
        d["power_source"] = "battery"
    m = re.search(r"(\d+)%", text)
    if m:
        d["battery_pct"] = m.group(1)
    return d


def _parse_lowpowermode(text: str | None) -> str | None:
    if not text:
        return None
    m = re.search(r"lowpowermode\s+(\d)", text)
    return {"0": "off", "1": "on"}.get(m.group(1), m.group(1)) if m else None


def _parse_display_count(text: str | None) -> str | None:
    if not text:
        return None
    # each display is a block with a 'Resolution:' line under 'Displays:'
    n = len(re.findall(r"^\s*Resolution:", text, flags=re.MULTILINE))
    return str(n) if n else None


def mac_machine_state() -> dict[str, Any]:
    state: dict[str, Any] = {"os": "macos"}
    state.update(_parse_pmset_therm(_run(["pmset", "-g", "therm"])))
    state.update(_parse_pmset_batt(_run(["pmset", "-g", "batt"])))
    lpm = _parse_lowpowermode(_run(["pmset", "-g"]))
    if lpm is not None:
        state["low_power_mode"] = lpm
    disp = _parse_display_count(_run(["system_profiler", "SPDisplaysDataType"], timeout=60))
    if disp is not None:
        state["displays"] = disp
    try:
        la = os.getloadavg()
        state["loadavg"] = ",".join(f"{x:.2f}" for x in la)
    except (OSError, AttributeError):
        pass
    state["engine_processes"] = engine_processes()
    state["ollama_loaded"] = ollama_loaded()
    return state


def mac_thermal_changed(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """A thermal warning appeared, or the CPU speed/scheduler limit moved."""
    if before.get("thermal_warning") != after.get("thermal_warning"):
        return True
    for k in ("cpu_speed_limit", "cpu_scheduler_limit", "thermal_warning_level"):
        if before.get(k) != after.get(k):
            return True
    return False


# ---- orchestration ----------------------------------------------------------------


def machine_state(platform_os: str | None = None) -> dict[str, Any]:
    return mac_machine_state()


class Preflight:
    """Checks around a job / engine switch / model, returning the recorded
    state (goes into ``Provenance.machine_state``) and invalidation reasons."""

    def before_job(self, platform_os: str | None = None) -> dict[str, Any]:
        return machine_state(platform_os)

    def between_engines(self, platform_os: str | None, leftover_names: list[str]) -> dict[str, Any]:
        return machine_state(platform_os)

    def after_model(self, platform_os: str | None, before_state: dict[str, Any]) -> list[str]:
        """Reasons the numbers taken since ``before_state`` cannot be read."""
        reasons: list[str] = []
        after = machine_state(platform_os)
        if mac_thermal_changed(before_state, after):
            reasons.append(
                f"thermal state changed during model: {before_state.get('thermal_warning')}"
                f"->{after.get('thermal_warning')} "
                f"(speed limit {before_state.get('cpu_speed_limit')}->{after.get('cpu_speed_limit')})"
            )
        if before_state.get("power_source") != after.get("power_source"):
            reasons.append(f"power source changed: {before_state.get('power_source')}->{after.get('power_source')}")
        if after.get("power_source") == "battery":
            reasons.append("running on battery")
        if after.get("low_power_mode") == "on":
            reasons.append("low power mode on")
        if before_state.get("displays") != after.get("displays"):
            reasons.append(f"display count changed: {before_state.get('displays')}->{after.get('displays')}")
        new = sorted(set(after.get("engine_processes") or []) - set(before_state.get("engine_processes") or []))
        if new:
            reasons.append(f"another inference engine started during the model: {', '.join(new)}")
        loaded = sorted(set(after.get("ollama_loaded") or []) - set(before_state.get("ollama_loaded") or []))
        if loaded:
            reasons.append(f"the resident Ollama loaded a model during the run: {', '.join(loaded)}")
        return reasons
