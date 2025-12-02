import chromadb
import uuid
import re
from typing import Any, Dict, List
from llama_index.core import (
    SimpleDirectoryReader, VectorStoreIndex, StorageContext, Document
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from config.settings import RAG_SOURCE_DIR, VECTOR_DB_DIR, COLLECTION_NAME
from utils.logger import get_logger
from utils.tool_events import publish_event

logger = get_logger("RAGTool")


class _LoggingQueryEngine:
    """Light wrapper to log every query routed to the vector store."""

    def __init__(self, query_engine, tool_name: str = "financial_theory_tool"):
        self._query_engine = query_engine
        self._tool_name = tool_name

    def _log_query(self, query_str: str):
        preview = (query_str or "").strip()
        if len(preview) > 80:
            preview = f"{preview[:77]}..."
        logger.info(f"[ToolCall] rag_knowledge_base | query='{preview}'")

    def query(self, *args, **kwargs):
        query_str = kwargs.get("query_str")
        if query_str is None and args:
            query_str = args[0]
        self._log_query(query_str or "")
        call_id = uuid.uuid4().hex
        publish_event(
            {
                "type": "tool_status",
                "call_id": call_id,
                "tool": self._tool_name,
                "status": "running",
                "progress": 15,
                "meta": {"query": (query_str or "")[:200]},
            }
        )
        try:
            result = self._query_engine.query(*args, **kwargs)
            snippet = getattr(result, "response", str(result))
            logger.info(f"[RAGResult] preview='{snippet}'")
            publish_event(
                {
                    "type": "tool_result",
                    "call_id": call_id,
                    "tool": self._tool_name,
                    "status": "succeeded",
                    "progress": 100,
                    "result": "已检索到相关资料，请查看下方详情。", # 移除 LLM 生成的回答，只保留提示语
                    "chunks": self._extract_chunks(result),
                }
            )
            return result
        except Exception as exc:
            publish_event(
                {
                    "type": "tool_status",
                    "call_id": call_id,
                    "tool": self._tool_name,
                    "status": "failed",
                    "progress": 100,
                    "error": str(exc),
                }
            )
            raise

    async def aquery(self, *args, **kwargs):
        query_str = kwargs.get("query_str")
        if query_str is None and args:
            query_str = args[0]
        self._log_query(query_str or "")
        call_id = uuid.uuid4().hex
        publish_event(
            {
                "type": "tool_status",
                "call_id": call_id,
                "tool": self._tool_name,
                "status": "running",
                "progress": 15,
                "meta": {"query": (query_str or "")[:200]},
            }
        )
        try:
            result = await self._query_engine.aquery(*args, **kwargs)
            snippet = getattr(result, "response", str(result))
            logger.info(f"[RAGResult] preview='{snippet}'")
            publish_event(
                {
                    "type": "tool_result",
                    "call_id": call_id,
                    "tool": self._tool_name,
                    "status": "succeeded",
                    "progress": 100,
                    "result": "已检索到相关资料，请查看下方详情。", # 移除 LLM 生成的回答，只保留提示语
                    "chunks": self._extract_chunks(result),
                }
            )
            return result
        except Exception as exc:
            publish_event(
                {
                    "type": "tool_status",
                    "call_id": call_id,
                    "tool": self._tool_name,
                    "status": "failed",
                    "progress": 100,
                    "error": str(exc),
                }
            )
            raise

    def _log_source_nodes(self, result):
        source_nodes = getattr(result, "source_nodes", [])
        for idx, node in enumerate(source_nodes, start=1):
            text = getattr(node, "text", "") or ""
            meta = getattr(node, "metadata", {}) or {}
            score = getattr(node, "score", None)
            node_id = getattr(node, "node_id", "n/a")
            meta_items = []
            for key in ("file_name", "source", "topic"):
                if key in meta:
                    meta_items.append(f"{key}={meta[key]}")
            meta_str = ", ".join(meta_items) if meta_items else "-"

            logger.info(
                "\n".join(
                    [
                        f"[RAGChunk #{idx}] score={score} node_id={node_id} meta={meta_str}",
                        text,
                        "[EndChunk]",
                    ]
                )
            )

    def _extract_chunks(self, result) -> List[Dict[str, Any]]:
        source_nodes = getattr(result, "source_nodes", []) or []
        chunks: List[Dict[str, Any]] = []
        for idx, node in enumerate(source_nodes, start=1):
            text = (getattr(node, "text", "") or "")[:256]
            metadata = self._sanitize_metadata(getattr(node, "metadata", {}) or {})
            chunks.append(
                {
                    "index": idx,
                    "node_id": getattr(node, "node_id", ""),
                    "score": getattr(node, "score", None),
                    "text": text,
                    "metadata": metadata,
                }
            )
        return chunks

    @staticmethod
    def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
        allowed_keys = ("file_name", "source", "topic", "title")
        return {key: meta[key] for key in allowed_keys if key in meta}

    def __getattr__(self, item):
        return getattr(self._query_engine, item)

class FinancialKnowledgeBase:
    def __init__(self):
        self.db_client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        self.collection = self.db_client.get_or_create_collection(COLLECTION_NAME)
        self._query_engine = None
        self.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=96)

    def build_or_load_index(self):
        vector_store = ChromaVectorStore(chroma_collection=self.collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if self.collection.count() == 0:
            logger.info("构建新索引...")
            if not any(RAG_SOURCE_DIR.iterdir()):
                logger.warning(f"数据目录 {RAG_SOURCE_DIR} 为空！")
                return None
            
            documents = []
            # 遍历目录，针对特定文件使用自定义加载逻辑
            for file_path in RAG_SOURCE_DIR.iterdir():
                if file_path.name == "stock_doc_content.txt":
                    logger.info(f"使用自定义加载器处理: {file_path.name}")
                    documents.extend(self._load_stock_docs(file_path))
                elif file_path.is_file() and not file_path.name.startswith("."):
                    # 其他文件使用默认加载器
                    logger.info(f"使用默认加载器处理: {file_path.name}")
                    documents.extend(SimpleDirectoryReader(input_files=[str(file_path)]).load_data())

            nodes = self.node_parser.get_nodes_from_documents(documents)
            
            # Node 级别去重：解决不同逻辑页面包含相同段落导致的内容冗余
            unique_nodes = []
            seen_node_texts = set()
            for node in nodes:
                # 归一化文本：去除首尾空白
                node_text = node.text.strip()
                if node_text not in seen_node_texts:
                    seen_node_texts.add(node_text)
                    unique_nodes.append(node)
            
            logger.info(f"Node去重完成: 原始 {len(nodes)} -> 去重后 {len(unique_nodes)} (移除 {len(nodes) - len(unique_nodes)} 个重复片段)")

            index = VectorStoreIndex(
                unique_nodes,
                storage_context=storage_context,
                show_progress=True,
            )
        else:
            logger.info("加载已有索引...")
            index = VectorStoreIndex.from_vector_store(
                vector_store, storage_context=storage_context
            )
        
        return index

    def _load_stock_docs(self, file_path) -> List[Document]:
        """
        自定义加载器：针对 stock_doc_content.txt 的特殊结构进行两级切分。
        第一级：按 xx_START_PAGE_xx 拆分逻辑页面并提取元数据。
        第二级：返回 Document 对象，后续由 SentenceSplitter 进行语义切分。
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
        except Exception as e:
            logger.error(f"读取文件失败 {file_path}: {e}")
            return []

        # 匹配 xx_START_PAGE_xx ... xx_END_PAGE_xx 之间的内容
        page_pattern = re.compile(r"xx_START_PAGE_xx\n(.*?)\nxx_END_PAGE_xx", re.DOTALL)
        matches = page_pattern.findall(raw_text)
        
        documents = []
        seen_texts = set() # 用于去重

        for i, page_content in enumerate(matches):
            lines = page_content.strip().split("\n")
            metadata = {"source_idx": i, "file_name": file_path.name}
            content_lines = []
            
            for line in lines:
                if line.startswith("URL:"):
                    metadata["url"] = line.replace("URL:", "").strip()
                    metadata["source"] = metadata["url"] # 兼容 source 字段
                elif line.startswith("TITLE:"):
                    metadata["title"] = line.replace("TITLE:", "").strip()
                else:
                    content_lines.append(line)
            
            text = "\n".join(content_lines).strip()
            
            # 去重逻辑：如果正文内容完全一致，则视为重复文档
            if text and text not in seen_texts:
                seen_texts.add(text)
                documents.append(Document(text=text, metadata=metadata))
        
        logger.info(f"自定义加载完成: {len(documents)} 个逻辑文档 (已去重 {len(matches) - len(documents)} 条)")
        return documents

    def _ensure_query_engine(self):
        if self._query_engine is not None:
            return self._query_engine

        index = self.build_or_load_index()
        if not index:
            logger.error("RAG 索引未准备好，无法提供查询功能。")
            return None

        # 策略调整：
        # 1. Top-K = 5: Qwen3-4B 有 8k 上下文，5个 chunk (约2.5k tokens) 完全可控。
        #    增大 K 值有助于同时召回定义、公式和案例。
        # 2. 不设置 similarity_cutoff: 观察到 BGE 模型在此数据集上得分较低(0.6左右)，
        #    设置硬阈值容易导致漏召回。
        query_engine = index.as_query_engine(similarity_top_k=5)
        self._query_engine = _LoggingQueryEngine(
            query_engine, tool_name="financial_theory_tool"
        )
        return self._query_engine

    def query_raw(self, query_text: str) -> str:
        query_engine = self._ensure_query_engine()
        if not query_engine:
            return ""

        try:
            result = query_engine.query(query_text)
            if hasattr(result, "response"):
                return result.response or ""
            return str(result)
        except Exception as e:
            logger.error(f"RAG 查询失败: {e}")
            return ""

    def get_tool(self):
        logging_query_engine = self._ensure_query_engine()
        if not logging_query_engine:
            return None

        return QueryEngineTool(
            query_engine=logging_query_engine,
            metadata=ToolMetadata(
                # 名称必须与 Prompt 严格一致
                name="financial_theory_tool",
                description=(
                    "【金融理论与规则查询】这是你获取金融知识的唯一来源。\n"
                    "无论用户问什么（包括'什么是股票'、'解释RSI'等基础问题），都**必须**先调用此工具。\n"
                    "即使你认为问题很简单，也必须查询此工具以获得标准定义。\n"
                    "即使问题是关于特定股票（如'茅台的RSI是多少'），你也必须先查此工具以获取'RSI的定义和判断标准'，然后再去查数据。\n"
                    "包含：基础概念、交易机制、指标公式、风险模型、策略理论等。"
                )
            ),
        )