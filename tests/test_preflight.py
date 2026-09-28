
from pie_evals.node import preflight as pf

THERM_OK = """Note: No thermal warning level has been recorded
Note: No performance warning level has been recorded
2026-09-28 10:00:00 +0000 CPU Power notify
\tCPU_Scheduler_Limit \t= 100
\tCPU_Available_CPUs \t= 12
\tCPU_Speed_Limit \t= 100
"""

THERM_HOT = """2026-09-28 10:00:00 +0000 Thermal Warning Level: 2
\tCPU_Scheduler_Limit \t= 100
\tCPU_Available_CPUs \t= 12
\tCPU_Speed_Limit \t= 70
"""


def test_pmset_therm():
    ok = pf._parse_pmset_therm(THERM_OK)
    assert ok["cpu_speed_limit"] == "100" and ok["cpu_available_cpus"] == "12"
    assert ok["thermal_warning"] == "no"
    hot = pf._parse_pmset_therm(THERM_HOT)
    assert hot["thermal_warning_level"] == "2" and hot["thermal_warning"] == "yes"
    assert pf._parse_pmset_therm(None) == {}


def test_pmset_batt_lowpower_and_displays():
    assert pf._parse_pmset_batt("Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t100%; charged;") == {"power_source": "ac", "battery_pct": "100"}
    assert pf._parse_pmset_batt("Now drawing from 'Battery Power'\n -InternalBattery-0\t81%; discharging;")["power_source"] == "battery"
    assert pf._parse_lowpowermode("System-wide power settings:\n lowpowermode         1\n") == "on"
    assert pf._parse_lowpowermode(" lowpowermode         0") == "off"
    assert pf._parse_lowpowermode("") is None
    two = "Displays:\n  Color LCD:\n    Resolution: 3456 x 2234\n  DELL:\n    Resolution: 2560 x 1440\n"
    assert pf._parse_display_count(two) == "2"
    assert pf._parse_display_count("Displays:\n") is None


def test_engine_processes_skips_our_own_children(monkeypatch):
    ps = "  101 /opt/homebrew/bin/ollama\n  102 /Users/x/pie/target/release/pie\n  103 /usr/bin/python3\n  104 ollama\n"
    monkeypatch.setattr(pf, "_run", lambda argv, timeout=20.0: ps)
    assert pf.engine_processes() == ["ollama:101", "ollama:104", "pie:102"]
    assert pf.engine_processes(exclude={104}) == ["ollama:101", "pie:102"]


def _state(**over):
    base = {"os": "macos", "thermal_warning": "no", "cpu_speed_limit": "100", "power_source": "ac",
            "displays": "1", "engine_processes": ["ollama:101"], "ollama_loaded": []}
    return {**base, **over}


def test_after_model_reads_a_steady_machine_as_clean(monkeypatch):
    monkeypatch.setattr(pf, "machine_state", lambda os_=None: _state())
    assert pf.Preflight().after_model("macos", _state()) == []


def test_after_model_names_every_change(monkeypatch):
    after = _state(thermal_warning="yes", cpu_speed_limit="70", power_source="battery", low_power_mode="on",
                   displays="2", engine_processes=["ollama:101", "pie:555"], ollama_loaded=["gemma4:26b"])
    monkeypatch.setattr(pf, "machine_state", lambda os_=None: after)
    reasons = " | ".join(pf.Preflight().after_model("macos", _state()))
    for want in ("thermal state changed", "power source changed", "running on battery", "low power mode on",
                 "display count changed", "pie:555", "gemma4:26b"):
        assert want in reasons, want
    assert "ollama:101" not in reasons


def test_no_pkill_invocation():
    with open(pf.__file__) as f:
        for line in f:
            code = line.split("#", 1)[0]
            assert '"pkill"' not in code and "'pkill'" not in code, f"pkill invoked: {line!r}"
