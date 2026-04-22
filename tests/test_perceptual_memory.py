import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from src.memory.base import MemoryConfig, MemoryItem, MemoryType
from src.memory.manager import MemoryManager
from src.memory.types.perceptual import PerceptualMemory


class FakeEmbeddingService:
    """测试用嵌入服务，避免外部网络依赖。"""

    def embed(self, text: str):
        lowered = (text or "").lower()
        return [
            float(lowered.count("猫") + lowered.count("cat") + lowered.count("图像")),
            float(lowered.count("音乐") + lowered.count("audio") + lowered.count("音频")),
            float(lowered.count("会议") + lowered.count("文本") + lowered.count("记录")),
        ]

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


class TestPerceptualMemory(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = MemoryConfig(
            document_store_path=str(Path(self.temp_dir.name) / "perceptual.db"),
            vector_store_url="http://127.0.0.1:65530",
            embedding_dimension=3,
            vector_collection_name="test_memories",
        )
        self.embedding_service = FakeEmbeddingService()
        self.memory = PerceptualMemory(self.config, self.embedding_service)

    def tearDown(self):
        self.memory.document_store.close()
        self.temp_dir.cleanup()

    def test_store_uses_modality_isolated_vector_collections(self):
        text_item = MemoryItem(
            id="text-1",
            content="会议文本记录",
            memory_type=MemoryType.PERCEPTUAL,
        )
        image_item = MemoryItem(
            id="image-1",
            content="一只猫的图像",
            memory_type=MemoryType.PERCEPTUAL,
            metadata={"description": "白色小猫趴在沙发上"},
        )

        self.memory.store(text_item, media_type="text")
        self.memory.store(image_item, media_data=b"fake-image", media_type="image")

        self.assertIn("text", self.memory._vector_stores)
        self.assertIn("image", self.memory._vector_stores)
        self.assertNotEqual(
            self.memory._vector_stores["text"].collection_name,
            self.memory._vector_stores["image"].collection_name,
        )

    def test_cross_modal_retrieve_works_via_bridge_text(self):
        image_item = MemoryItem(
            id="image-cat",
            content="",
            memory_type=MemoryType.PERCEPTUAL,
            metadata={"description": "白色小猫在沙发上睡觉"},
            importance=0.8,
        )
        text_item = MemoryItem(
            id="text-meeting",
            content="会议记录：讨论预算审批",
            memory_type=MemoryType.PERCEPTUAL,
            importance=0.6,
        )

        self.memory.store(image_item, media_data=b"image-bytes", media_type="image")
        self.memory.store(text_item, media_type="text")

        cross_modal_results = self.memory.search_by_semantic(
            "小猫",
            limit=5,
            query_media_type="text",
            cross_modal=True,
        )
        same_modal_results = self.memory.search_by_semantic(
            "小猫",
            limit=5,
            query_media_type="text",
            cross_modal=False,
        )

        self.assertTrue(cross_modal_results)
        self.assertEqual(cross_modal_results[0].id, "image-cat")
        self.assertFalse(same_modal_results)

    def test_score_formula_combines_similarity_recency_and_importance(self):
        recent = MemoryItem(
            id="recent",
            content="猫 图像 描述",
            memory_type=MemoryType.PERCEPTUAL,
            embedding=self.embedding_service.embed("猫 图像 描述"),
            metadata={"media_type": "image", "bridge_text": "猫 图像 描述"},
            created_at=datetime.now(),
            importance=1.0,
        )
        old = MemoryItem(
            id="old",
            content="猫 图像 描述",
            memory_type=MemoryType.PERCEPTUAL,
            embedding=self.embedding_service.embed("猫 图像 描述"),
            metadata={"media_type": "image", "bridge_text": "猫 图像 描述"},
            created_at=datetime.now() - timedelta(days=14),
            importance=0.0,
        )

        recent_score = self.memory.score_item("猫", recent)
        old_score = self.memory.score_item("猫", old)

        self.assertIsNotNone(recent_score)
        self.assertIsNotNone(old_score)
        self.assertGreater(recent_score, old_score)

    def test_degraded_mode_keeps_keyword_and_cross_modal_description_search(self):
        degraded_memory = PerceptualMemory(self.config, embedding_service=None)
        try:
            image_item = MemoryItem(
                id="image-desc",
                content="",
                memory_type=MemoryType.PERCEPTUAL,
                metadata={"description": "海边日落照片"},
            )
            degraded_memory.store(image_item, media_data=b"image", media_type="image")

            results = degraded_memory.search_hybrid(
                "日落",
                limit=5,
                query_media_type="text",
                cross_modal=True,
            )

            self.assertTrue(results)
            self.assertEqual(results[0].id, "image-desc")
        finally:
            degraded_memory.document_store.close()

    def test_manager_supports_all_keyword_semantic_and_hybrid_perceptual_search(self):
        manager = MemoryManager(self.config)
        try:
            manager.embedding_service = self.embedding_service
            manager.memories[MemoryType.PERCEPTUAL] = PerceptualMemory(self.config, self.embedding_service)

            manager.store_perceptual(
                media_type="text",
                content="会议文本记录",
                importance=0.5,
            )
            manager.store_perceptual(
                media_type="image",
                content="",
                media_data=b"image",
                description="黑猫蹲在窗台上",
                importance=0.9,
            )

            all_items = manager.search_perceptual("", search_mode="all", limit=10)
            keyword_results = manager.search_perceptual("会议", search_mode="keyword", limit=10)
            semantic_results = manager.search_perceptual(
                "黑猫",
                search_mode="semantic",
                query_media_type="text",
                cross_modal=True,
                limit=10,
            )
            hybrid_results = manager.search_perceptual("黑猫", search_mode="hybrid", limit=10)

            self.assertEqual(len(all_items), 2)
            self.assertTrue(keyword_results)
            self.assertTrue(semantic_results)
            self.assertTrue(hybrid_results)
        finally:
            manager.memories[MemoryType.PERCEPTUAL].document_store.close()
            manager.memories[MemoryType.EPISODIC].document_store.close()


if __name__ == "__main__":
    unittest.main()
