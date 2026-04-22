import tempfile
import unittest
from pathlib import Path

from src.memory.base import MemoryConfig, MemoryItem, MemoryType
from src.memory.manager import MemoryManager
from src.memory.types.semantic import SemanticMemory


class FakeEmbeddingService:
    """测试用嵌入服务，避免外部网络依赖。"""

    def embed(self, text: str):
        text = text or ""
        lowered = text.lower()
        return [
            float(lowered.count("python") + lowered.count("语义") + lowered.count("知识")),
            float(lowered.count("neo4j") + lowered.count("图谱") + lowered.count("关系")),
            float(lowered.count("qdrant") + lowered.count("向量") + lowered.count("检索")),
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


class TestSemanticMemory(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = MemoryConfig(
            document_store_path=str(Path(self.temp_dir.name) / "semantic.db"),
            vector_store_url="http://127.0.0.1:65530",
            graph_store_url="bolt://127.0.0.1:65531",
            embedding_dimension=3,
        )
        self.embedding_service = FakeEmbeddingService()
        self.memory = SemanticMemory(self.config, self.embedding_service)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_store_extracts_entities_and_relations(self):
        item = MemoryItem(
            id="semantic-1",
            content="Qdrant使用Neo4j，Neo4j连接知识图谱",
            memory_type=MemoryType.SEMANTIC,
            importance=0.7,
        )

        self.memory.store(item)
        graph_scores = self.memory._graph_scores("Neo4j 图谱", limit=5)

        self.assertIn("semantic-1", self.memory._items)
        self.assertGreater(graph_scores.get("semantic-1", 0.0), 0.0)

    def test_hybrid_search_prefers_vector_and_graph_combined_score(self):
        stronger = MemoryItem(
            id="stronger",
            content="Python语义记忆使用Qdrant向量检索，并用Neo4j构建知识图谱关系",
            memory_type=MemoryType.SEMANTIC,
            importance=1.0,
        )
        weaker = MemoryItem(
            id="weaker",
            content="Python基础语法笔记",
            memory_type=MemoryType.SEMANTIC,
            importance=0.1,
        )

        self.memory.store(weaker)
        self.memory.store(stronger)

        results = self.memory.search_hybrid("Python Neo4j 语义图谱", limit=5)

        self.assertGreaterEqual(len(results), 2)
        self.assertEqual(results[0].id, "stronger")

    def test_keyword_and_all_items_work_when_neo4j_service_unavailable(self):
        item = MemoryItem(
            id="fallback-item",
            content="Qdrant 负责向量检索，语义记忆保留基础能力",
            memory_type=MemoryType.SEMANTIC,
            importance=0.6,
        )
        self.memory.store(item)

        keyword_results = self.memory.search_by_keyword("Qdrant", limit=5)
        all_items = self.memory.get_all_items()

        self.assertTrue(keyword_results)
        self.assertEqual(keyword_results[0].id, "fallback-item")
        self.assertEqual([stored.id for stored in all_items], ["fallback-item"])

    def test_manager_can_use_semantic_search_modes(self):
        manager = MemoryManager(self.config)
        manager.embedding_service = self.embedding_service
        manager.memories[MemoryType.SEMANTIC] = SemanticMemory(self.config, self.embedding_service)

        manager.store(
            "语义记忆中，Qdrant负责向量检索，Neo4j负责知识图谱关系",
            MemoryType.SEMANTIC,
            importance=0.9,
            tags=["memory", "semantic"],
        )

        self.assertTrue(manager.search("Qdrant", [MemoryType.SEMANTIC], search_mode="keyword", limit=5))
        self.assertTrue(manager.search("知识图谱", [MemoryType.SEMANTIC], search_mode="semantic", limit=5))
        self.assertTrue(manager.search("Neo4j 关系", [MemoryType.SEMANTIC], search_mode="hybrid", limit=5))

    def test_manager_store_degrades_when_embedding_service_fails(self):
        class BrokenEmbeddingService:
            """测试用故障嵌入服务，验证 manager 不会把异常向外放大。"""

            def embed(self, text: str):
                raise RuntimeError("embedding unavailable")

            def embed_batch(self, texts):
                raise RuntimeError("embedding unavailable")

            def cosine_similarity(self, vec1, vec2):
                return 0.0

        manager = MemoryManager(self.config)
        manager.embedding_service = BrokenEmbeddingService()
        manager.memories[MemoryType.SEMANTIC] = SemanticMemory(self.config, embedding_service=None)

        memory_id = manager.store(
            "Qdrant 负责向量检索，Neo4j 负责关系补充",
            MemoryType.SEMANTIC,
            importance=0.8,
        )

        self.assertTrue(memory_id)
        self.assertEqual(len(manager.memories[MemoryType.SEMANTIC].get_all_items()), 1)


if __name__ == "__main__":
    unittest.main()
