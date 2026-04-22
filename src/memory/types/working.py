"""工作记忆实现。"""

from __future__ import annotations

import math
import re
import time
from collections import Counter, OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from src.memory.base import BaseMemory, MemoryConfig, MemoryItem


class WorkingMemory(BaseMemory):
    """工作记忆：由 manager 管理生命周期，但数据只保存在进程内存中。

    设计目标：
        1. 仅做纯内存存储，进程重启后允许直接丢失。
        2. 每次读写前都执行 TTL 清理，保证过期项不会继续参与检索。
        3. 提供关键词、语义、混合检索以及原始访问接口，便于 manager 调用。
    """

    # 使用一个较平滑的衰减窗口，避免工作记忆在 TTL 中后段突然失效。
    # 指数衰减的 half-life 取 TTL 的一半，意味着记忆存活到 TTL/2 时，时间权重约为 0.5。
    # 这样既能让新记忆更容易排前，也不会让稍早但重要的内容瞬间失去价值。
    _HALF_LIFE_RATIO = 0.5

    def __init__(self, config: Optional[MemoryConfig] = None):
        super().__init__(config)
        self.memories: OrderedDict[str, MemoryItem] = OrderedDict()
        self.ttl = self.config.working_memory_ttl
        self.max_items = self.config.max_working_memory_items

    def store(self, item: MemoryItem) -> str:
        """存储工作记忆项。

        说明:
            - 若调用方未显式指定 item.ttl，则自动继承工作记忆默认 TTL。
            - 容量达到上限时，移除当前最旧的项目，保持常数级追加性能。
        """
        self._cleanup()

        if item.ttl is None:
            item.ttl = self.ttl

        if item.id in self.memories:
            del self.memories[item.id]

        while len(self.memories) >= self.max_items:
            self.memories.popitem(last=False)

        self.memories[item.id] = item
        return item.id

    def retrieve(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """默认检索入口，直接走混合检索。"""
        return self.search_hybrid(query, limit=limit)

    def delete(self, memory_id: str) -> bool:
        """删除工作记忆项。"""
        self._cleanup()
        if memory_id in self.memories:
            del self.memories[memory_id]
            return True
        return False

    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """更新工作记忆项。"""
        self._cleanup()
        item = self.memories.get(memory_id)
        if item is None:
            return False

        for key, value in updates.items():
            if hasattr(item, key):
                setattr(item, key, value)

        from datetime import datetime

        item.updated_at = datetime.now()
        self.memories.move_to_end(memory_id)
        return True

    def clear(self) -> bool:
        """清空工作记忆。"""
        self.memories.clear()
        return True

    def get_recent(self, limit: int = 10) -> List[MemoryItem]:
        """获取最近的工作记忆项。"""
        self._cleanup()
        return list(reversed(self.get_all_items()))[:limit]

    def get_all_items(self) -> List[MemoryItem]:
        """获取当前仍有效的全部工作记忆项。"""
        self._cleanup()
        return list(self.memories.values())

    def search_by_keyword(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """关键词搜索。

        说明:
            - 这里不是简单的字符串包含判断，而是按词项重叠比例做轻量匹配。
            - 返回结果已经按工作记忆统一评分公式排序，便于 manager 直接使用。
        """
        return self._search(query=query, limit=limit, mode="keyword")

    def search_by_semantic(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """语义搜索。

        实现说明:
            - 优先使用轻量 TF-IDF 余弦相似度，避免依赖外部向量服务。
            - 当查询或候选文本无法构成有效 TF-IDF 向量时，返回空结果，
              由混合检索逻辑统一回退到关键词匹配。
        """
        return self._search(query=query, limit=limit, mode="semantic")

    def search_hybrid(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """混合检索。

        规则:
            - 优先尝试 TF-IDF 语义检索。
            - 若语义检索没有有效结果，则回退到关键词检索。
            - 最终严格按 `(相似度 × 时间衰减) × (0.8 + 重要性 × 0.4)` 排序。
        """
        semantic_results = self._search(query=query, limit=limit, mode="semantic")
        if semantic_results:
            return semantic_results
        return self._search(query=query, limit=limit, mode="keyword")

    def score_item(self, query: str, item: MemoryItem) -> Optional[float]:
        """为 manager 提供工作记忆的内部评分。"""
        self._cleanup()
        if item.id not in self.memories:
            return None

        semantic_similarity = self._semantic_similarity(query, item)
        if semantic_similarity > 0:
            return self._final_score(item, semantic_similarity)

        keyword_similarity = self._keyword_similarity(query, item)
        if keyword_similarity > 0:
            return self._final_score(item, keyword_similarity)

        return 0.0

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息。"""
        self._cleanup()
        return {
            "total_items": len(self.memories),
            "max_capacity": self.max_items,
            "ttl_seconds": self.ttl,
        }

    def _cleanup(self) -> None:
        """按项目 TTL 清理过期的工作记忆。

        说明:
            - 工作记忆不依赖后台线程；采用“访问时清理”策略，简单且足够稳定。
            - item.ttl 优先，其次回退到工作记忆默认 TTL。
        """
        current_time = time.time()
        expired_keys: List[str] = []

        for key, item in self.memories.items():
            ttl_seconds = item.ttl if item.ttl is not None else self.ttl
            if ttl_seconds is None:
                continue

            item_age = current_time - item.created_at.timestamp()
            if item_age >= ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            del self.memories[key]

    def _search(self, query: str, limit: int, mode: str) -> List[MemoryItem]:
        """统一搜索入口，负责清理、评分、排序和截断。"""
        self._cleanup()
        if limit <= 0:
            return []

        scored_items: List[Tuple[float, float, MemoryItem]] = []
        for item in self.memories.values():
            if mode == "semantic":
                similarity = self._semantic_similarity(query, item)
            elif mode == "keyword":
                similarity = self._keyword_similarity(query, item)
            else:
                similarity = 0.0

            if similarity <= 0:
                continue

            score = self._final_score(item, similarity)
            scored_items.append((score, item.created_at.timestamp(), item))

        scored_items.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        return [item for _, _, item in scored_items[:limit]]

    def _final_score(self, item: MemoryItem, similarity: float) -> float:
        """按要求实现统一评分公式。

        公式严格为:
            `(相似度 × 时间衰减) × (0.8 + 重要性 × 0.4)`

        其中:
            - 相似度来自 TF-IDF 或关键词匹配。
            - 时间衰减仅基于 created_at，而不是 updated_at，避免修改操作“刷新年龄”。
            - 重要性按 0.0 到 1.0 夹紧，防止异常值破坏排序。
        """
        importance = min(max(item.importance, 0.0), 1.0)
        time_decay = self._time_decay(item)
        return (similarity * time_decay) * (0.8 + importance * 0.4)

    def _time_decay(self, item: MemoryItem) -> float:
        """基于创建时间计算指数衰减。

        说明:
            - 记忆越旧，衰减越明显，但不会线性突然降到 0。
            - 使用指数衰减 `0.5 ** (age / half_life)`，half_life 取 TTL 的一半。
            - 当 age = half_life 时，时间权重为 0.5；当 age 接近 TTL 时，时间权重会平滑下降。
            - 若 TTL 不可用，则退化到一个稳定的默认窗口，避免出现除零或无衰减行为。
        """
        ttl_seconds = item.ttl if item.ttl is not None else self.ttl
        effective_ttl = max(float(ttl_seconds or self.ttl or 300), 1.0)
        half_life = max(effective_ttl * self._HALF_LIFE_RATIO, 1.0)
        age_seconds = max(time.time() - item.created_at.timestamp(), 0.0)
        return 0.5 ** (age_seconds / half_life)

    def _semantic_similarity(self, query: str, item: MemoryItem) -> float:
        """计算轻量 TF-IDF 语义相似度。"""
        query_tokens = self._tokenize(query)
        item_tokens = self._tokenize(self._item_search_text(item))
        if not query_tokens or not item_tokens:
            return 0.0

        tfidf_query, tfidf_item = self._build_tfidf_vectors(query_tokens, item_tokens)
        if not tfidf_query or not tfidf_item:
            return 0.0

        return self._cosine_similarity(tfidf_query, tfidf_item)

    def _keyword_similarity(self, query: str, item: MemoryItem) -> float:
        """计算关键词匹配相似度。

        说明:
            - 优先使用词项交集比例。
            - 若分词后没有有效词项，则退回到原始子串包含判断，保证极短查询也能命中。
        """
        query_tokens = self._tokenize(query)
        item_tokens = self._tokenize(self._item_search_text(item))

        if query_tokens and item_tokens:
            overlap = len(set(query_tokens).intersection(item_tokens))
            return overlap / len(set(query_tokens))

        query_text = query.strip().lower()
        if query_text and query_text in self._item_search_text(item).lower():
            return 1.0
        return 0.0

    def _item_search_text(self, item: MemoryItem) -> str:
        """把内容、标签和可检索元数据拼成统一检索文本。"""
        metadata_values = [str(value) for value in item.metadata.values() if value is not None]
        parts = [item.content, *item.tags, *metadata_values]
        return " ".join(part for part in parts if part)

    def _tokenize(self, text: str) -> List[str]:
        """将文本切分为轻量词项。

        说明:
            - 英文和数字按连续字符分词。
            - 中文在没有空格的情况下，额外保留双字切片，以提升短语匹配能力。
            - 这是一个尽量轻量的本地实现，不引入额外中文分词依赖。
        """
        lowered = text.lower()
        latin_tokens = re.findall(r"[\w\u4e00-\u9fff]+", lowered)
        if not latin_tokens:
            return []

        tokens: List[str] = []
        for token in latin_tokens:
            tokens.append(token)
            if self._contains_cjk(token) and len(token) > 1:
                tokens.extend(token[index : index + 2] for index in range(len(token) - 1))
        return tokens

    def _build_tfidf_vectors(
        self,
        query_tokens: List[str],
        item_tokens: List[str],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """为查询与单条记忆构造最小 TF-IDF 向量。"""
        query_counts = Counter(query_tokens)
        item_counts = Counter(item_tokens)
        vocabulary = set(query_counts) | set(item_counts)
        if not vocabulary:
            return {}, {}

        total_docs = 2
        query_vector: Dict[str, float] = {}
        item_vector: Dict[str, float] = {}

        for token in vocabulary:
            doc_freq = int(token in query_counts) + int(token in item_counts)
            idf = math.log((1 + total_docs) / (1 + doc_freq)) + 1.0
            query_vector[token] = (query_counts[token] / len(query_tokens)) * idf
            item_vector[token] = (item_counts[token] / len(item_tokens)) * idf

        return query_vector, item_vector

    def _cosine_similarity(
        self,
        left: Dict[str, float],
        right: Dict[str, float],
    ) -> float:
        """计算稀疏字典向量的余弦相似度。"""
        if not left or not right:
            return 0.0

        common_tokens = set(left).intersection(right)
        numerator = sum(left[token] * right[token] for token in common_tokens)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))

        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0

        return numerator / (left_norm * right_norm)

    def _contains_cjk(self, text: str) -> bool:
        """判断文本是否包含中日韩统一表意文字。"""
        return any("\u4e00" <= char <= "\u9fff" for char in text)
