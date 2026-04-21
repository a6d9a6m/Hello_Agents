"""Neo4j图存储实现"""

from __future__ import annotations

from typing import List, Dict, Any, Optional

from src.memory.base import MemoryConfig


class Neo4jStore:
    """Neo4j图数据库客户端"""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.driver = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """确保客户端已初始化"""
        if not self._initialized:
            try:
                # 尝试导入neo4j
                from neo4j import GraphDatabase
                
                self.driver = GraphDatabase.driver(
                    self.config.graph_store_url,
                    auth=(self.config.graph_username, self.config.graph_password)
                )
                
                # 测试连接
                with self.driver.session() as session:
                    session.run("RETURN 1")
                
                self._initialized = True
                
            except ImportError:
                print("警告：未安装neo4j，使用模拟模式")
                self.driver = None
                self._initialized = True
            except Exception as e:
                print(f"Neo4j连接失败：{e}")
                self.driver = None
                self._initialized = True
    
    def create_node(self, node_id: str, node_type: str, properties: Dict[str, Any]) -> bool:
        """创建节点"""
        self._ensure_initialized()
        
        if self.driver is None:
            # 模拟模式
            return True
        
        try:
            with self.driver.session() as session:
                query = """
                MERGE (n:Node {id: $node_id})
                SET n.type = $node_type,
                    n += $properties,
                    n.created_at = timestamp()
                RETURN n
                """
                
                session.run(
                    query,
                    node_id=node_id,
                    node_type=node_type,
                    properties=properties
                )
            return True
        except Exception as e:
            print(f"创建节点失败：{e}")
            return False
    
    def create_relation(
        self, 
        from_id: str, 
        to_id: str, 
        relation_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """创建关系"""
        self._ensure_initialized()
        
        if self.driver is None:
            return True
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (a:Node {id: $from_id})
                MATCH (b:Node {id: $to_id})
                MERGE (a)-[r:RELATION {type: $relation_type}]->(b)
                SET r += $properties,
                    r.created_at = timestamp()
                RETURN r
                """
                
                session.run(
                    query,
                    from_id=from_id,
                    to_id=to_id,
                    relation_type=relation_type,
                    properties=properties or {}
                )
            return True
        except Exception as e:
            print(f"创建关系失败：{e}")
            return False
    
    def query(self, cypher_query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """执行Cypher查询"""
        self._ensure_initialized()
        
        if self.driver is None:
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            print(f"执行查询失败：{e}")
            return []
    
    def get_related_nodes(self, node_id: str, depth: int = 2) -> List[Dict[str, Any]]:
        """获取相关节点"""
        self._ensure_initialized()
        
        if self.driver is None:
            return []
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH path = (start:Node {id: $node_id})-[*1..$depth]-(related:Node)
                RETURN related, length(path) as distance
                ORDER BY distance
                LIMIT 50
                """
                
                result = session.run(query, node_id=node_id, depth=depth)
                return [record.data() for record in result]
        except Exception as e:
            print(f"获取相关节点失败：{e}")
            return []
    
    def delete_nodes_by_source(self, source_memory: str) -> bool:
        """删除指定来源的节点"""
        self._ensure_initialized()
        
        if self.driver is None:
            return True
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (n:Node {source_memory: $source_memory})
                DETACH DELETE n
                RETURN count(n) as deleted_count
                """
                
                result = session.run(query, source_memory=source_memory)
                return True
        except Exception as e:
            print(f"删除节点失败：{e}")
            return False
    
    def clear(self) -> bool:
        """清空图数据库"""
        self._ensure_initialized()
        
        if self.driver is None:
            return True
        
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            return True
        except Exception as e:
            print(f"清空图数据库失败：{e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        self._ensure_initialized()
        
        if self.driver is None:
            return {"status": "simulated", "node_count": 0, "relation_count": 0}
        
        try:
            with self.driver.session() as session:
                # 获取节点数量
                node_result = session.run("MATCH (n) RETURN count(n) as node_count")
                node_count = node_result.single()["node_count"]
                
                # 获取关系数量
                rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as relation_count")
                relation_count = rel_result.single()["relation_count"]
                
                return {
                    "status": "ok",
                    "node_count": node_count,
                    "relation_count": relation_count
                }
        except Exception as e:
            print(f"获取统计信息失败：{e}")
            return {"status": "error", "error": str(e)}