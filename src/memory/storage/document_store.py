"""SQLite文档存储实现"""

from __future__ import annotations

import sqlite3
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from src.memory.base import MemoryConfig


class DocumentStore:
    """SQLite文档存储"""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.db_path = Path(self.config.document_store_path)
        self.connection = None
        self._initialized = False
        
        # 确保目录存在
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _ensure_initialized(self):
        """确保数据库已初始化"""
        if not self._initialized:
            try:
                self.connection = sqlite3.connect(self.db_path)
                self.connection.row_factory = sqlite3.Row
                self._create_tables()
                self._initialized = True
            except Exception as e:
                print(f"SQLite连接失败：{e}")
                self.connection = None
                self._initialized = True
    
    def _create_tables(self):
        """创建数据库表"""
        if self.connection is None:
            return
        
        cursor = self.connection.cursor()
        
        # 文档表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            doc_type TEXT NOT NULL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # 关系表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_id TEXT NOT NULL,
            to_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            properties TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (from_id) REFERENCES documents(id),
            FOREIGN KEY (to_id) REFERENCES documents(id)
        )
        """)
        
        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(doc_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_id)")
        
        self.connection.commit()
    
    def store_document(
        self,
        doc_id: str,
        content: str,
        doc_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ) -> bool:
        """存储文档"""
        self._ensure_initialized()
        
        if self.connection is None:
            return True
        
        try:
            cursor = self.connection.cursor()
            
            # 检查文档是否已存在
            cursor.execute("SELECT id FROM documents WHERE id = ?", (doc_id,))
            existing = cursor.fetchone()
            
            metadata_json = json.dumps(metadata or {})
            
            created_at_text = created_at.isoformat() if created_at else None
            updated_at_text = updated_at.isoformat() if updated_at else None

            if existing:
                # 更新现有文档
                if updated_at_text:
                    cursor.execute(
                        """
                        UPDATE documents
                        SET content = ?, doc_type = ?, metadata = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (content, doc_type, metadata_json, updated_at_text, doc_id),
                    )
                else:
                    cursor.execute(
                        """
                        UPDATE documents
                        SET content = ?, doc_type = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """,
                        (content, doc_type, metadata_json, doc_id),
                    )
            else:
                # 插入新文档
                if created_at_text or updated_at_text:
                    cursor.execute(
                        """
                        INSERT INTO documents (id, content, doc_type, metadata, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            doc_id,
                            content,
                            doc_type,
                            metadata_json,
                            created_at_text or datetime.now().isoformat(),
                            updated_at_text or created_at_text or datetime.now().isoformat(),
                        ),
                    )
                else:
                    cursor.execute(
                        """
                        INSERT INTO documents (id, content, doc_type, metadata)
                        VALUES (?, ?, ?, ?)
                        """,
                        (doc_id, content, doc_type, metadata_json),
                    )
            
            self.connection.commit()
            return True
        except Exception as e:
            print(f"存储文档失败：{e}")
            return False

    def update_document(
        self,
        doc_id: str,
        *,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """更新文档内容或元数据。"""
        self._ensure_initialized()

        if self.connection is None:
            return True

        try:
            current = self.get_document(doc_id)
            if current is None:
                return False

            next_content = current["content"] if content is None else content
            next_metadata = current["metadata"] if metadata is None else metadata

            cursor = self.connection.cursor()
            cursor.execute(
                """
                UPDATE documents
                SET content = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (next_content, json.dumps(next_metadata or {}), doc_id),
            )
            self.connection.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"更新文档失败：{e}")
            return False
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """获取文档"""
        self._ensure_initialized()
        
        if self.connection is None:
            return None
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            
            if row:
                return {
                    "id": row["id"],
                    "content": row["content"],
                    "doc_type": row["doc_type"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                }
            return None
        except Exception as e:
            print(f"获取文档失败：{e}")
            return None
    
    def search_documents(self, query: str, doc_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索文档（简单全文搜索）"""
        self._ensure_initialized()
        
        if self.connection is None:
            return []
        
        try:
            cursor = self.connection.cursor()
            
            if doc_type:
                cursor.execute("""
                SELECT * FROM documents 
                WHERE content LIKE ? AND doc_type = ?
                ORDER BY created_at DESC
                LIMIT ?
                """, (f"%{query}%", doc_type, limit))
            else:
                cursor.execute("""
                SELECT * FROM documents 
                WHERE content LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """, (f"%{query}%", limit))
            
            rows = cursor.fetchall()
            results = []
            
            for row in rows:
                results.append({
                    "id": row["id"],
                    "content": row["content"],
                    "doc_type": row["doc_type"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                })
            
            return results
        except Exception as e:
            print(f"搜索文档失败：{e}")
            return []

    def list_documents(self, doc_type: Optional[str] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """按时间顺序列出文档。"""
        self._ensure_initialized()

        if self.connection is None:
            return []

        try:
            cursor = self.connection.cursor()
            if doc_type:
                cursor.execute(
                    """
                    SELECT * FROM documents
                    WHERE doc_type = ?
                    ORDER BY created_at ASC, id ASC
                    LIMIT ?
                    """,
                    (doc_type, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM documents
                    ORDER BY created_at ASC, id ASC
                    LIMIT ?
                    """,
                    (limit,),
                )

            return [self._row_to_document(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"列出文档失败：{e}")
            return []

    def search_documents_by_metadata_keyword(
        self,
        query: str,
        doc_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """在内容和元数据 JSON 中做轻量关键词检索。"""
        self._ensure_initialized()

        if self.connection is None:
            return []

        try:
            cursor = self.connection.cursor()
            like_query = f"%{query}%"
            if doc_type:
                cursor.execute(
                    """
                    SELECT * FROM documents
                    WHERE doc_type = ? AND (content LIKE ? OR metadata LIKE ?)
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                    """,
                    (doc_type, like_query, like_query, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM documents
                    WHERE content LIKE ? OR metadata LIKE ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                    """,
                    (like_query, like_query, limit),
                )

            return [self._row_to_document(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"按元数据搜索文档失败：{e}")
            return []

    def list_documents_by_time_range(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        doc_type: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """按创建时间范围查询文档，保留时间线顺序。"""
        self._ensure_initialized()

        if self.connection is None:
            return []

        try:
            conditions: List[str] = []
            parameters: List[Any] = []

            if doc_type:
                conditions.append("doc_type = ?")
                parameters.append(doc_type)

            if start_time:
                conditions.append("created_at >= ?")
                parameters.append(start_time.isoformat())

            if end_time:
                conditions.append("created_at <= ?")
                parameters.append(end_time.isoformat())

            where_clause = ""
            if conditions:
                where_clause = " WHERE " + " AND ".join(conditions)

            cursor = self.connection.cursor()
            cursor.execute(
                f"""
                SELECT * FROM documents
                {where_clause}
                ORDER BY created_at ASC, id ASC
                LIMIT ?
                """,
                (*parameters, limit),
            )
            return [self._row_to_document(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"按时间范围查询文档失败：{e}")
            return []
    
    def delete(self, doc_id: str) -> bool:
        """删除文档"""
        self._ensure_initialized()
        
        if self.connection is None:
            return True
        
        try:
            cursor = self.connection.cursor()
            
            # 删除文档
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            
            # 删除相关关系
            cursor.execute("DELETE FROM relations WHERE from_id = ? OR to_id = ?", (doc_id, doc_id))
            
            self.connection.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"删除文档失败：{e}")
            return False
    
    def create_relation(self, from_id: str, to_id: str, relation_type: str, properties: Optional[Dict[str, Any]] = None) -> bool:
        """创建文档关系"""
        self._ensure_initialized()
        
        if self.connection is None:
            return True
        
        try:
            cursor = self.connection.cursor()
            
            properties_json = json.dumps(properties or {})
            
            cursor.execute("""
            INSERT INTO relations (from_id, to_id, relation_type, properties)
            VALUES (?, ?, ?, ?)
            """, (from_id, to_id, relation_type, properties_json))
            
            self.connection.commit()
            return True
        except Exception as e:
            print(f"创建关系失败：{e}")
            return False
    
    def get_relations(self, doc_id: str, relation_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取文档关系"""
        self._ensure_initialized()
        
        if self.connection is None:
            return []
        
        try:
            cursor = self.connection.cursor()
            
            if relation_type:
                cursor.execute("""
                SELECT * FROM relations 
                WHERE (from_id = ? OR to_id = ?) AND relation_type = ?
                ORDER BY created_at DESC
                """, (doc_id, doc_id, relation_type))
            else:
                cursor.execute("""
                SELECT * FROM relations 
                WHERE from_id = ? OR to_id = ?
                ORDER BY created_at DESC
                """, (doc_id, doc_id))
            
            rows = cursor.fetchall()
            results = []
            
            for row in rows:
                results.append({
                    "id": row["id"],
                    "from_id": row["from_id"],
                    "to_id": row["to_id"],
                    "relation_type": row["relation_type"],
                    "properties": json.loads(row["properties"]) if row["properties"] else {},
                    "created_at": row["created_at"]
                })
            
            return results
        except Exception as e:
            print(f"获取关系失败：{e}")
            return []
    
    def clear(self) -> bool:
        """清空数据库"""
        self._ensure_initialized()
        
        if self.connection is None:
            return True
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("DELETE FROM documents")
            cursor.execute("DELETE FROM relations")
            self.connection.commit()
            return True
        except Exception as e:
            print(f"清空数据库失败：{e}")
            return False

    def close(self) -> None:
        """关闭 SQLite 连接，便于测试和进程退出时释放文件句柄。"""
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        self._initialized = False

    def _row_to_document(self, row: sqlite3.Row) -> Dict[str, Any]:
        """统一将 SQLite 行对象转换为字典。"""
        return {
            "id": row["id"],
            "content": row["content"],
            "doc_type": row["doc_type"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        self._ensure_initialized()
        
        if self.connection is None:
            return {"status": "simulated", "document_count": 0, "relation_count": 0}
        
        try:
            cursor = self.connection.cursor()
            
            # 文档数量
            cursor.execute("SELECT COUNT(*) as count FROM documents")
            doc_count = cursor.fetchone()["count"]
            
            # 关系数量
            cursor.execute("SELECT COUNT(*) as count FROM relations")
            rel_count = cursor.fetchone()["count"]
            
            # 文档类型分布
            cursor.execute("SELECT doc_type, COUNT(*) as count FROM documents GROUP BY doc_type")
            type_distribution = {row["doc_type"]: row["count"] for row in cursor.fetchall()}
            
            return {
                "status": "ok",
                "document_count": doc_count,
                "relation_count": rel_count,
                "type_distribution": type_distribution
            }
        except Exception as e:
            print(f"获取统计信息失败：{e}")
            return {"status": "error", "error": str(e)}
