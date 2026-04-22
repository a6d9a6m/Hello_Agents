import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from src.memory.base import MemoryConfig, MemoryItem, MemoryType
from src.memory.types.episodic import EpisodicMemory


class FakeEmbeddingService:
    """测试用嵌入服务，避免外部网络依赖。"""

    def embed(self, text: str):
        base = float(len(text.strip()) or 1)
        return [base, base / 2.0, 1.0]

    def embed_batch(self, texts):
        return [self.embed(text) for text in texts]

    def cosine_similarity(self, vec1, vec2):
        if not vec1 or not vec2:
            return 0.0

        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)


class TestEpisodicMemory(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = MemoryConfig(
            document_store_path=str(Path(self.temp_dir.name) / "episodic.db"),
            vector_store_url="http://127.0.0.1:65530",
            embedding_dimension=3,
        )
        self.embedding_service = FakeEmbeddingService()
        self.memory = EpisodicMemory(self.config, self.embedding_service)

    def tearDown(self):
        if self.memory.document_store:
            self.memory.document_store.close()
        self.temp_dir.cleanup()

    def test_store_and_get_all_items_keep_timeline(self):
        older_time = datetime.now() - timedelta(days=2)
        newer_time = datetime.now() - timedelta(hours=1)

        older = MemoryItem(
            id="older",
            content="项目启动会议",
            memory_type=MemoryType.EPISODIC,
            created_at=older_time,
            updated_at=older_time,
            metadata={"event_time": older_time.isoformat()},
            importance=0.4,
        )
        newer = MemoryItem(
            id="newer",
            content="项目复盘会议",
            memory_type=MemoryType.EPISODIC,
            created_at=newer_time,
            updated_at=newer_time,
            metadata={"event_time": newer_time.isoformat()},
            importance=0.9,
        )

        self.memory.store(newer)
        self.memory.store(older)

        items = self.memory.get_all_items()
        self.assertEqual([item.id for item in items], ["older", "newer"])

    def test_keyword_search_works_when_qdrant_unavailable(self):
        item = MemoryItem(
            id="keyword-only",
            content="用户在周会上强调要先完成 SQLite 集成",
            memory_type=MemoryType.EPISODIC,
            importance=0.8,
        )
        self.memory.store(item)

        results = self.memory.search_by_semantic("SQLite 集成", limit=5)
        self.assertTrue(results)
        self.assertEqual(results[0].id, "keyword-only")

    def test_hybrid_search_prefers_recent_and_important_event(self):
        old_time = datetime.now() - timedelta(days=30)
        recent_time = datetime.now() - timedelta(hours=2)

        old_item = MemoryItem(
            id="old",
            content="发布事故复盘记录",
            memory_type=MemoryType.EPISODIC,
            created_at=old_time,
            updated_at=old_time,
            metadata={"event_time": old_time.isoformat()},
            importance=0.2,
        )
        recent_item = MemoryItem(
            id="recent",
            content="发布事故复盘记录",
            memory_type=MemoryType.EPISODIC,
            created_at=recent_time,
            updated_at=recent_time,
            metadata={"event_time": recent_time.isoformat()},
            importance=1.0,
        )

        self.memory.store(old_item)
        self.memory.store(recent_item)

        results = self.memory.search_hybrid("发布事故复盘记录", limit=5)
        self.assertGreaterEqual(len(results), 2)
        self.assertEqual(results[0].id, "recent")


if __name__ == "__main__":
    unittest.main()
