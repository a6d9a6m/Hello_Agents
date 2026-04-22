import time
import unittest
from datetime import datetime, timedelta

from src.memory.base import MemoryConfig, MemoryItem, MemoryType
from src.memory.types.working import WorkingMemory


class TestWorkingMemory(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = WorkingMemory(
            MemoryConfig(
                working_memory_ttl=10,
                max_working_memory_items=10,
            )
        )

    def test_hybrid_prefers_semantic_and_uses_formula_sorting(self) -> None:
        older_high_importance = MemoryItem(
            id="older-high",
            content="用户需要准备机器学习项目复盘",
            memory_type=MemoryType.WORKING,
            importance=1.0,
            created_at=datetime.now() - timedelta(seconds=3),
        )
        newer_low_importance = MemoryItem(
            id="newer-low",
            content="今天继续做机器学习项目总结",
            memory_type=MemoryType.WORKING,
            importance=0.0,
            created_at=datetime.now(),
        )

        self.memory.store(older_high_importance)
        self.memory.store(newer_low_importance)

        results = self.memory.search_hybrid("机器学习项目", limit=2)

        self.assertEqual([item.id for item in results], ["newer-low", "older-high"])

    def test_hybrid_falls_back_to_keyword_when_semantic_has_no_result(self) -> None:
        item = MemoryItem(
            id="keyword-only",
            content="提醒：ABC-123 需要单独登记",
            memory_type=MemoryType.WORKING,
        )
        self.memory.store(item)

        results = self.memory.search_hybrid("ABC-123", limit=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "keyword-only")

    def test_cleanup_removes_expired_items(self) -> None:
        expired = MemoryItem(
            id="expired",
            content="这条记忆应该过期",
            memory_type=MemoryType.WORKING,
            ttl=1,
            created_at=datetime.now() - timedelta(seconds=2),
        )
        active = MemoryItem(
            id="active",
            content="这条记忆仍然有效",
            memory_type=MemoryType.WORKING,
            ttl=10,
        )

        self.memory.store(expired)
        self.memory.store(active)

        items = self.memory.get_all_items()

        self.assertEqual([item.id for item in items], ["active"])

    def test_get_recent_returns_latest_items_first(self) -> None:
        first = MemoryItem(
            id="first",
            content="第一条",
            memory_type=MemoryType.WORKING,
            created_at=datetime.now() - timedelta(seconds=2),
        )
        second = MemoryItem(
            id="second",
            content="第二条",
            memory_type=MemoryType.WORKING,
            created_at=datetime.now() - timedelta(seconds=1),
        )

        self.memory.store(first)
        time.sleep(0.01)
        self.memory.store(second)

        results = self.memory.get_recent(limit=2)

        self.assertEqual([item.id for item in results], ["second", "first"])


if __name__ == "__main__":
    unittest.main()
