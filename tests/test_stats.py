from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from worldline_social.providers import BalanceInfo, DeepSeekProvider
from worldline_social.stats import (
    SQLiteUsageStore,
    UsageRecord,
    aggregate_usage,
)

BALANCE_JSON = json.dumps(
    {
        "is_available": True,
        "balance_infos": [
            {
                "currency": "CNY",
                "total_balance": "110.00",
                "granted_balance": "10.00",
                "topped_up_balance": "100.00",
            }
        ],
    }
)


class FakeUrlOpener:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload.encode("utf-8")


class FakeResponse:
    def __init__(self, payload: str) -> None:
        self._inner = FakeUrlOpener(payload)

    def __enter__(self) -> FakeUrlOpener:
        return self._inner

    def __exit__(self, *args: object) -> None:
        return None


class BalanceTests(unittest.TestCase):
    def test_get_balance_parses_deepseek_envelope(self) -> None:
        provider = DeepSeekProvider(api_key="test-key")
        with mock.patch(
            "worldline_social.providers.deepseek.urlopen",
            return_value=FakeResponse(BALANCE_JSON),
        ):
            balance = asyncio.run(provider.get_balance())
        self.assertTrue(balance.is_available)
        self.assertEqual("CNY", balance.currency)
        self.assertEqual("110.00", balance.total_balance)
        self.assertEqual("10.00", balance.granted_balance)
        self.assertEqual("100.00", balance.topped_up_balance)

    def test_get_balance_handles_missing_infos(self) -> None:
        provider = DeepSeekProvider(api_key="test-key")
        with mock.patch(
            "worldline_social.providers.deepseek.urlopen",
            return_value=FakeResponse(json.dumps({"is_available": False})),
        ):
            balance = asyncio.run(provider.get_balance())
        self.assertFalse(balance.is_available)
        self.assertEqual("0.00", balance.total_balance)


class UsageStoreTests(unittest.TestCase):
    def test_record_and_totals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteUsageStore(Path(tmp) / "run.sqlite")
            store.record(
                UsageRecord(
                    simulation_id="sim-1",
                    tick_id=3,
                    entity_id="alice",
                    model="deepseek-v4-flash",
                    usage={
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "prompt_cache_hit_tokens": 60,
                        "prompt_cache_miss_tokens": 40,
                    },
                )
            )
            store.record(
                UsageRecord(
                    simulation_id="sim-1",
                    tick_id=4,
                    entity_id="bob",
                    model="deepseek-v4-flash",
                    usage={
                        "prompt_tokens": 200,
                        "completion_tokens": 25,
                        "prompt_cache_hit_tokens": 0,
                        "prompt_cache_miss_tokens": 200,
                    },
                )
            )
            totals = store.totals("sim-1")
            self.assertEqual(300, totals["prompt_tokens"])
            self.assertEqual(75, totals["completion_tokens"])
            self.assertEqual(60, totals["prompt_cache_hit_tokens"])
            self.assertEqual(240, totals["prompt_cache_miss_tokens"])
            # total_tokens falls back to prompt + completion
            self.assertEqual(375, totals["total_tokens"])
            self.assertEqual(2, len(store.usage_rows("sim-1")))
            store.close()

    def test_aggregate_usage(self) -> None:
        records = [
            UsageRecord("s", 0, "a", "m", usage={"prompt_tokens": 10}),
            UsageRecord("s", 0, "b", "m", usage={"completion_tokens": 5}),
        ]
        totals = aggregate_usage(records)
        self.assertEqual(10, totals["prompt_tokens"])
        self.assertEqual(5, totals["completion_tokens"])
        self.assertEqual(0, totals["total_tokens"])

    def test_run_cost_spent_is_before_minus_after(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteUsageStore(Path(tmp) / "run.sqlite")
            store.record_run_cost(
                simulation_id="sim-1",
                balance_before="110.0000",
                balance_after="109.1234",
                started_at=1.0,
                finished_at=2.0,
            )
            cost = store.latest_run_cost("sim-1")
            self.assertIsNotNone(cost)
            assert cost is not None
            self.assertEqual("0.8766", cost["spent"])
            self.assertEqual("110.0000", cost["balance_before"])
            self.assertEqual("109.1234", cost["balance_after"])
            store.close()

    def test_run_cost_without_balances_has_no_spent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteUsageStore(Path(tmp) / "run.sqlite")
            store.record_run_cost(
                simulation_id="sim-1",
                balance_before=None,
                balance_after=None,
                started_at=1.0,
                finished_at=2.0,
            )
            cost = store.latest_run_cost("sim-1")
            self.assertIsNotNone(cost)
            assert cost is not None
            self.assertIsNone(cost["spent"])
            store.close()

    def test_latest_run_cost_returns_most_recent_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteUsageStore(Path(tmp) / "run.sqlite")
            store.record_run_cost(
                simulation_id="sim-1",
                balance_before="10", balance_after="9", started_at=1, finished_at=2,
            )
            store.record_run_cost(
                simulation_id="sim-1",
                balance_before="9", balance_after="7", started_at=3, finished_at=4,
            )
            cost = store.latest_run_cost("sim-1")
            self.assertIsNotNone(cost)
            assert cost is not None
            self.assertEqual("2.0000", cost["spent"])
            store.close()

    def test_records_are_scoped_by_simulation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteUsageStore(Path(tmp) / "run.sqlite")
            store.record(UsageRecord("sim-a", 0, "alice", "m", usage={"prompt_tokens": 7}))
            store.record(UsageRecord("sim-b", 0, "bob", "m", usage={"prompt_tokens": 9}))
            self.assertEqual(1, len(store.usage_rows("sim-a")))
            self.assertEqual(7, store.totals("sim-a")["prompt_tokens"])
            self.assertEqual(9, store.totals("sim-b")["prompt_tokens"])
            store.close()


if __name__ == "__main__":
    unittest.main()
