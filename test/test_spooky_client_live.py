"""Integration test: hits a real Spooky server. Opt-in, skipped by default.

Run with a Spooky server already running (see README.md) and either:
    SPOOKY_LIVE_TESTS=1 pytest test/test_spooky_client_live.py
or just: pytest --run-live test/test_spooky_client_live.py
(the --run-live flag is registered in conftest.py)

Override the target server with SPOOKY_BASE_URL (default http://localhost:8000).
Skips automatically if the server isn't reachable, so it's safe to leave in
the normal test run for anyone who happens to have Spooky up locally --
it just won't run for anyone who doesn't opt in either way.
"""

from __future__ import annotations

import os

import pytest
import requests

from fleet_coordinator.robot import Fleet
from fleet_coordinator.spooky_client import SpookySettings, plan_fleet

BASE_URL = os.environ.get("SPOOKY_BASE_URL", "http://localhost:8000")


def _live_tests_requested(request) -> bool:
    return bool(os.environ.get("SPOOKY_LIVE_TESTS")) or request.config.getoption(
        "--run-live", default=False
    )


def _server_reachable() -> bool:
    try:
        requests.get(f"{BASE_URL}/v1/maps", timeout=2)
        return True
    except requests.RequestException:
        return False


@pytest.fixture(autouse=True)
def _skip_unless_opted_in_and_reachable(request):
    if not _live_tests_requested(request):
        pytest.skip("live Spooky test skipped -- pass --run-live or set SPOOKY_LIVE_TESTS=1")
    if not _server_reachable():
        pytest.skip(f"no Spooky server reachable at {BASE_URL}")


def test_plan_fleet_against_live_server():
    fleet = Fleet.from_specs(
        [
            {"id": "ranger_1", "start": {"x": 0.0, "y": 0.0}, "goal": {"x": 2.0, "y": 2.0}},
            {"id": "ranger_2", "start": {"x": 2.0, "y": 0.0}, "goal": {"x": 0.0, "y": 2.0}},
        ]
    )
    plan = plan_fleet(fleet, SpookySettings(base_url=BASE_URL))

    assert set(plan.robot_plans) == {"ranger_1", "ranger_2"}
    for robot_plan in plan.robot_plans.values():
        assert len(robot_plan.path) >= 2
        assert robot_plan.coordinate_format == "world"
