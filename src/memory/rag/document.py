"""文档处理器"""

from __future__ import annotations

import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import mimetypes


class DocumentProcessor:
    """多格式文档处理器"""
    
    def __init__(self):
        self.supported_formats = {
            'text/plain': self._process_text,
            'text/markdown': self._process_markdown,
            'application/pdf': self._process_pdf,
            'application/msword': self._process_doc,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self._process_docx,
            'text/html': self._process_html,
            'application/json': self._process_json,
        }
    
    def process(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """处理文档"""
        content = document.get("content", "")
        file_path = document.get("file_path", "")
        content_type = document.get("content_type", "")
        
        # 确定内容类型
        if not content_type and file_path:
            content_type, _ = mimetypes.guess_type(file_path)
        
        if not content_type:
            content_type = 'text/plain'
        
        # 获取处理器
        processor = self.supported_formats.get(content_type, self._process_text)
        
        # 处理内容
        processed_content = processor(content, file_path)
        
        # 提取元数据
        metadata = self._extract_metadata(processed_content, document.get("metadata", {}))
        
        return {
            "content": processed_content,
            "metadata": metadata,
            "content_type": content_type,
            "file_path": file_path
        }
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """处理文件"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"文件不存在：{file_path}")
        
        # 读取文件内容
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # 尝试解码为文本
            try:
                content_str = content.decode('utf-8')
            except UnicodeDecodeError:
                # 如果是二进制文件（如PDF），保持为bytes
                content_str = ""
                content_bytes = content
            else:
                content_bytes = None
            
            return self.process({
                "content": content_str,
                "file_path": file_path,
                "content_bytes": content_bytes
            })
        except Exception as e:
            raise Exception(f"处理文件失败：{file_path} - {e}")
    
    def chunk_document(self, content: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """分块文档"""
        if not content:
            return []
        
        chunks = []
        start = 0
        content_length = len(content)
        
        while start < content_length:
            end = min(start + chunk_size, content_length)
            
            # 尝试在句子边界处截断
            if end < content_length:
                # 查找最近的句子结束符
                sentence_breaks = ['.', '!', '?', '\n\n', '\r\n\r\n']
                for break_char in sentence_breaks:
                    break_pos = content.rfind(break_char, start, end)
                    if break_pos != -1 and break_pos > start + chunk_size // 2:
                        end = break_pos + len(break_char)
                        break
            
            chunk = content[start:end].strip()
            if chunk:
                chunks.append({
                    "content": chunk,
                    "start_pos": start,
                    "end_pos": end,
                    "chunk_index": len(chunks)
                })
            
            start = end - overlap
        
        return chunks
    
    def _process_text(self, content: str, file_path: str = "") -> str:
        """处理纯文本"""
        # 清理空白字符
        content = re.sub(r'\s+', ' ', content)
        return content.strip()
    
    def _process_markdown(self, content: str, file_path: str = "") -> str:
        """处理Markdown"""
        # 移除Markdown标记
        content = re.sub(r'#+\s*', '', content)  # 标题
        content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # 粗体
        content = re.sub(r'\*(.*?)\*', r'\1', content)  # 斜体
        content = re.sub(r'`(.*?)`', r'\1', content)  # 代码
        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)  # 图片
        content = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', content)  # 链接
        
        return self._process_text(content)
    
    def _process_html(self, content: str, file_path: str = "") -> str:
        """处理HTML"""
        # 简单移除HTML标签
        content = re.sub(r'<[^>]+>', ' ', content)
        content = re.sub(r'&[a-z]+;', ' ', content)
        
        return self._process_text(content)
    
    def _process_json(self, content: str, file_path: str = "") -> str:
        """处理JSON"""
        import json
        try:
            data = json.loads(content)
            # 将JSON转换为可读文本
            return self._json_to_text(data)
        except:
            return self._process_text(content)
    
    def _process_pdf(self, content: str, file_path: str = "") -> str:
        """处理PDF"""
        # 需要PyPDF2或pdfplumber
        # 简化实现：返回占位符
        return f"[PDF文件：{file_path or '未命名'}]"
    
    def _process_doc(self, content: str, file_path: str = "") -> str:
        """处理DOC"""
        # 需要python-docx
        return f"[DOC文件：{file_path or '未命名'}]"
    
    def _process_docx(self, content: str, file_path: str = "") -> str:
        """处理DOCX"""
        # 需要python-docx
        return f"[DOCX文件：{file_path or '未命名'}]"
    
    def _extract_metadata(self, content: str, existing_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """提取元数据"""
        metadata = existing_metadata.copy()
        
        # 提取基本信息
        metadata["content_length"] = len(content)
        metadata["word_count"] = len(content.split())
        
        # 提取潜在的主题关键词（简单实现）
        words = content.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # 忽略短词
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 取频率最高的5个词作为关键词
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        metadata["keywords"] = [word for word, _ in top_words]
        
        # 检测语言（简单实现）
        metadata["language"] = self._detect_language(content)
        
        return metadata
    
    def _detect_language(self, text: str) -> str:
        """检测语言（简化实现）"""
        # 简单基于字符范围检测
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if chinese_chars > english_chars:
            return "zh"
        elif english_chars > 0:
            return "en"
        else:
            return "unknown"
    
    def _json_to_text(self, data: Any, indent: int = 0) -> str:
        """将JSON转换为可读文本"""
        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{'  ' * indent}{key}:")
                    lines.append(self._json_to_text(value, indent + 1))
                else:
                    lines.append(f"{'  ' * indent}{key}: {value}")
            return "\n".join(lines)
        elif isinstance(data, list):
            lines = []
            for i, item in enumerate(data):
                lines.append(f"{'  ' * indent}[{i}]: {self._json_to_text(item, indent + 1)}")
            return "\n".join(lines)
        else:
            return str(data)