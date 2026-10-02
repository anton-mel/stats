import threading
import time

import pytest

from pie_evals.node.machine import MachineBusy, machine_lock


@pytest.fixture(autouse=True)
def lock_file(tmp_path, monkeypatch):
    monkeypatch.setenv("PIE_MACHINE_LOCK", str(tmp_path / "machine.lock"))
    return tmp_path / "machine.lock"


def test_a_second_benchmark_waits_for_the_first(lock_file):
    order = []

    def first():
        with machine_lock("first", log=lambda m: None):
            order.append("first in")
            time.sleep(0.4)
            order.append("first out")

    t = threading.Thread(target=first)
    t.start()
    time.sleep(0.1)
    said = []
    with machine_lock("second", poll_s=0.05, log=said.append) as waited:
        order.append("second in")
    t.join()
    assert order == ["first in", "first out", "second in"] and waited > 0.2
    assert "pid" in said[0] and "first" in said[0]
    assert "second" in lock_file.read_text()


def test_a_benchmark_gives_up_after_its_wait(lock_file):
    with machine_lock("holder", log=lambda m: None):
        with pytest.raises(MachineBusy) as e:
            with machine_lock("late", max_wait_s=0.2, poll_s=0.05, log=lambda m: None):
                pass
    assert "holder" in str(e.value)
    with machine_lock("after", max_wait_s=5, log=lambda m: None) as waited:
        assert waited < 5
