import os
import asyncio
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from datetime import datetime
import hashlib

# 设置日志
logger = logging.getLogger(__name__)

class SimpleRAGService:
    """简化版RAG知识库服务"""
    
    def __init__(self, working_dir: str = "./simple_rag_storage"):
        self.working_dir = working_dir
        self.is_initialized = False
        self.api_key = None
        self.base_url = None
        self.documents = {}  # 存储文档内容
        self.document_metadata = {}  # 存储文档元数据
        
        # 确保工作目录存在
        os.makedirs(working_dir, exist_ok=True)
        
        # 加载已存在的文档
        self._load_documents()
    
    def _load_documents(self):
        """加载已存储的文档"""
        try:
            docs_file = os.path.join(self.working_dir, "documents.json")
            meta_file = os.path.join(self.working_dir, "metadata.json")
            
            if os.path.exists(docs_file):
                with open(docs_file, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
            
            if os.path.exists(meta_file):
                with open(meta_file, 'r', encoding='utf-8') as f:
                    self.document_metadata = json.load(f)
                    
        except Exception as e:
            logger.warning(f"加载文档失败: {e}")
    
    def _save_documents(self):
        """保存文档到磁盘"""
        try:
            docs_file = os.path.join(self.working_dir, "documents.json")
            meta_file = os.path.join(self.working_dir, "metadata.json")
            
            with open(docs_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(self.document_metadata, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"保存文档失败: {e}")
    
    async def initialize(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        """初始化RAG系统"""
        try:
            self.api_key = api_key
            self.base_url = base_url
            self.is_initialized = True
            
            logger.info("简化RAG服务初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"RAG服务初始化失败: {str(e)}")
            return False
    
    async def add_document(self, content: str, file_path: str = None, metadata: Dict = None) -> bool:
        """添加文档到知识库"""
        if not self.is_initialized:
            logger.error("RAG服务未初始化")
            return False
        
        try:
            # 生成文档ID
            doc_id = hashlib.md5(content.encode()).hexdigest()
            
            # 存储文档内容
            self.documents[doc_id] = {
                "content": content,
                "file_path": file_path,
                "created_at": datetime.now().isoformat()
            }
            
            # 存储元数据
            if metadata:
                self.document_metadata[doc_id] = metadata
            
            # 保存到磁盘
            self._save_documents()
            
            logger.info(f"文档添加成功: {metadata.get('filename', 'unknown') if metadata else 'unknown'}")
            return True
            
        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            return False
    
    async def query_knowledge_base(self, query: str, mode: str = "hybrid") -> Dict[str, Any]:
        """查询知识库"""
        if not self.is_initialized:
            return {
                "success": False,
                "message": "RAG服务未初始化",
                "response": None
            }
        
        try:
            # 简单的关键词匹配搜索
            query_lower = query.lower()
            relevant_docs = []
            
            for doc_id, doc_data in self.documents.items():
                content = doc_data["content"].lower()
                if any(keyword in content for keyword in query_lower.split()):
                    metadata = self.document_metadata.get(doc_id, {})
                    relevant_docs.append({
                        "content": doc_data["content"][:500] + "..." if len(doc_data["content"]) > 500 else doc_data["content"],
                        "filename": metadata.get("filename", "unknown"),
                        "relevance": "medium"
                    })
            
            if relevant_docs:
                # 构建响应
                response = f"基于知识库中的 {len(relevant_docs)} 个相关文档，我找到了以下信息：\n\n"
                for i, doc in enumerate(relevant_docs[:3], 1):  # 最多返回3个文档
                    response += f"文档 {i} ({doc['filename']}):\n{doc['content']}\n\n"
                
                response += f"请根据以上信息回答用户的问题：{query}"
            else:
                response = f"在知识库中没有找到与 '{query}' 相关的信息。请尝试其他关键词或上传相关文档。"
            
            return {
                "success": True,
                "message": "查询成功",
                "response": response,
                "mode": mode,
                "found_docs": len(relevant_docs)
            }
            
        except Exception as e:
            logger.error(f"知识库查询失败: {str(e)}")
            return {
                "success": False,
                "message": f"查询失败: {str(e)}",
                "response": None
            }
    
    async def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        try:
            # 计算存储大小
            total_size = 0
            storage_path = Path(self.working_dir)
            if storage_path.exists():
                for file_path in storage_path.rglob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
            
            # 计算内容长度
            total_content_length = sum(len(doc["content"]) for doc in self.documents.values())
            
            stats = {
                "total_documents": len(self.documents),
                "total_content_length": total_content_length,
                "storage_size": f"{total_size / 1024:.2f} KB",
                "last_updated": datetime.now().isoformat(),
                "working_dir": self.working_dir,
                "is_initialized": self.is_initialized
            }
            
            return {
                "success": True,
                "message": "统计信息获取成功",
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            return {
                "success": False,
                "message": f"获取统计信息失败: {str(e)}",
                "stats": None
            }
    
    async def clear_knowledge_base(self) -> bool:
        """清空知识库"""
        try:
            self.documents = {}
            self.document_metadata = {}
            
            # 删除存储文件
            docs_file = os.path.join(self.working_dir, "documents.json")
            meta_file = os.path.join(self.working_dir, "metadata.json")
            
            if os.path.exists(docs_file):
                os.remove(docs_file)
            if os.path.exists(meta_file):
                os.remove(meta_file)
            
            logger.info("知识库清空成功")
            return True
            
        except Exception as e:
            logger.error(f"清空知识库失败: {str(e)}")
            return False
    
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """从PDF文件内容提取文本"""
        try:
            import PyPDF2
            import io
            
            # 使用BytesIO处理文件内容
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            extracted_text = text.strip()
            if not extracted_text:
                logger.warning("PDF文件中未提取到文本内容")
                return ""
            
            logger.info(f"成功从PDF提取文本，长度: {len(extracted_text)} 字符")
            return extracted_text
            
        except ImportError:
            logger.warning("PyPDF2未安装，无法处理PDF文件")
            return ""
        except Exception as e:
            logger.error(f"PDF文本提取失败: {str(e)}")
            return ""
    
    def extract_text_from_txt(self, file_content: bytes) -> str:
        """从TXT文件内容提取文本"""
        try:
            # 尝试不同的编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            
            for encoding in encodings:
                try:
                    text = file_content.decode(encoding)
                    logger.info(f"成功使用 {encoding} 编码读取TXT文件，长度: {len(text)} 字符")
                    return text.strip()
                except UnicodeDecodeError:
                    continue
            
            logger.warning("无法解码TXT文件，尝试所有编码都失败")
            return ""
            
        except Exception as e:
            logger.error(f"TXT文本提取失败: {str(e)}")
            return ""

# 全局简化RAG服务实例
simple_rag_service = SimpleRAGService()

async def get_simple_rag_service() -> SimpleRAGService:
    """获取简化RAG服务实例"""
    return simple_rag_service