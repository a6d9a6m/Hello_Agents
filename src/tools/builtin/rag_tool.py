"""RAG工具（智能问答能力）"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from pathlib import Path

from src.tools.base import Tool, ToolParameter, ToolResult
from src.memory.rag.pipeline import SimpleRAGPipeline, HybridRAGPipeline
from src.memory.rag.document import DocumentProcessor


class RAGTool(Tool):
    """RAG工具：为Agent提供智能问答能力"""
    
    name = "rag"
    description = "基于检索增强生成的智能问答工具，支持文档上传和知识查询"
    
    parameters = [
        ToolParameter(
            name="operation",
            param_type=str,
            description="操作类型：ingest(摄取文档)、query(查询知识)、clear(清空知识库)",
            required=True,
            enum=["ingest", "query", "clear"]
        ),
        ToolParameter(
            name="content",
            param_type=str,
            description="查询内容或文档内容",
            required=False
        ),
        ToolParameter(
            name="file_path",
            param_type=str,
            description="文档文件路径",
            required=False
        ),
        ToolParameter(
            name="top_k",
            param_type=int,
            description="检索结果数量",
            required=False,
            default=3
        ),
        ToolParameter(
            name="pipeline_type",
            param_type=str,
            description="RAG管道类型：simple(简单)、hybrid(混合)",
            required=False,
            default="simple",
            enum=["simple", "hybrid"]
        )
    ]
    
    def __init__(self):
        super().__init__()
        self.pipelines: Dict[str, Any] = {}
        self.document_processor = DocumentProcessor()
        self.current_pipeline = None
    
    def execute(self, **validated) -> ToolResult:
        """执行RAG操作"""
        operation = validated["operation"]
        
        try:
            if operation == "ingest":
                result = self._ingest_documents(validated)
            elif operation == "query":
                result = self._query_knowledge(validated)
            elif operation == "clear":
                result = self._clear_knowledge(validated)
            else:
                result = ToolResult(
                    tool_name=self.name,
                    output=f"不支持的操作：{operation}",
                    success=False
                )
            
            return result
            
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                output=f"RAG操作失败：{str(e)}",
                success=False
            )
    
    def _ingest_documents(self, params: Dict[str, Any]) -> ToolResult:
        """摄取文档"""
        content = params.get("content", "")
        file_path = params.get("file_path", "")
        pipeline_type = params.get("pipeline_type", "simple")
        
        documents = []
        
        # 处理文件
        if file_path:
            path = Path(file_path)
            if not path.exists():
                return ToolResult(
                    tool_name=self.name,
                    output=f"文件不存在：{file_path}",
                    success=False
                )
            
            try:
                processed = self.document_processor.process_file(file_path)
                documents.append({
                    "content": processed["content"],
                    "file_path": file_path,
                    "metadata": processed["metadata"]
                })
            except Exception as e:
                return ToolResult(
                    tool_name=self.name,
                    output=f"处理文件失败：{str(e)}",
                    success=False
                )
        
        # 处理直接内容
        if content and not file_path:
            documents.append({
                "content": content,
                "metadata": {"source": "direct_input"}
            })
        
        if not documents:
            return ToolResult(
                tool_name=self.name,
                output="没有提供文档内容或文件路径",
                success=False
            )
        
        # 获取或创建管道
        pipeline = self._get_pipeline(pipeline_type)
        
        # 摄取文档
        doc_ids = pipeline.ingest(documents)
        
        # 分块统计
        total_chunks = 0
        for doc in documents:
            chunks = self.document_processor.chunk_document(doc["content"])
            total_chunks += len(chunks)
        
        return ToolResult(
            tool_name=self.name,
            output=f"文档摄取成功，处理了 {len(documents)} 个文档，{total_chunks} 个文本块",
            success=True,
            metadata={
                "document_count": len(documents),
                "chunk_count": total_chunks,
                "pipeline_type": pipeline_type,
                "document_ids": doc_ids
            }
        )
    
    def _query_knowledge(self, params: Dict[str, Any]) -> ToolResult:
        """查询知识"""
        query = params.get("content", "")
        pipeline_type = params.get("pipeline_type", "simple")
        top_k = params.get("top_k", 3)
        
        if not query:
            return ToolResult(
                tool_name=self.name,
                output="查询需要提供content参数",
                success=False
            )
        
        # 获取管道
        pipeline = self._get_pipeline(pipeline_type)
        
        # 执行查询
        answer = pipeline.query(query)
        
        # 获取检索上下文（用于调试）
        context = pipeline.retrieve(query, top_k=top_k)
        
        # 构建详细输出
        output_parts = [f"问题：{query}", f"\n答案：{answer}"]
        
        if context:
            output_parts.append(f"\n检索到的相关信息（{len(context)} 条）：")
            for i, doc in enumerate(context, 1):
                similarity = doc.get("similarity", 0)
                method = doc.get("retrieval_method", "unknown")
                output_parts.append(f"\n{i}. [相似度：{similarity:.3f}, 方法：{method}]")
                output_parts.append(f"   {doc['content'][:150]}...")
        
        return ToolResult(
            tool_name=self.name,
            output="\n".join(output_parts),
            success=True,
            metadata={
                "query": query,
                "context_count": len(context),
                "pipeline_type": pipeline_type
            }
        )
    
    def _clear_knowledge(self, params: Dict[str, Any]) -> ToolResult:
        """清空知识库"""
        pipeline_type = params.get("pipeline_type", "simple")
        
        if pipeline_type in self.pipelines:
            pipeline = self.pipelines[pipeline_type]
            if hasattr(pipeline, 'clear'):
                pipeline.clear()
            
            # 从字典中移除
            del self.pipelines[pipeline_type]
            
            return ToolResult(
                tool_name=self.name,
                output=f"{pipeline_type}管道知识库已清空",
                success=True
            )
        else:
            return ToolResult(
                tool_name=self.name,
                output=f"未找到{pipeline_type}管道",
                success=False
            )
    
    def _get_pipeline(self, pipeline_type: str):
        """获取或创建RAG管道"""
        if pipeline_type not in self.pipelines:
            if pipeline_type == "simple":
                self.pipelines[pipeline_type] = SimpleRAGPipeline()
            elif pipeline_type == "hybrid":
                self.pipelines[pipeline_type] = HybridRAGPipeline()
            else:
                raise ValueError(f"不支持的管道类型：{pipeline_type}")
        
        self.current_pipeline = self.pipelines[pipeline_type]
        return self.current_pipeline
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的文档格式"""
        return list(self.document_processor.supported_formats.keys())
    
    def batch_ingest(self, file_paths: List[str], pipeline_type: str = "simple") -> ToolResult:
        """批量摄取文档"""
        success_count = 0
        fail_count = 0
        errors = []
        
        for file_path in file_paths:
            try:
                result = self._ingest_documents({
                    "operation": "ingest",
                    "file_path": file_path,
                    "pipeline_type": pipeline_type
                })
                
                if result.success:
                    success_count += 1
                else:
                    fail_count += 1
                    errors.append(f"{file_path}: {result.output}")
            except Exception as e:
                fail_count += 1
                errors.append(f"{file_path}: {str(e)}")
        
        output = f"批量处理完成：成功 {success_count} 个，失败 {fail_count} 个"
        if errors:
            output += f"\n失败文件：\n" + "\n".join(errors[:5])  # 只显示前5个错误
        
        return ToolResult(
            tool_name=self.name,
            output=output,
            success=fail_count == 0,
            metadata={
                "success_count": success_count,
                "fail_count": fail_count,
                "total_count": len(file_paths)
            }
        )