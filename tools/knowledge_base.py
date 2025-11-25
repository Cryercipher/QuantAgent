import chromadb
from llama_index.core import (
    SimpleDirectoryReader, VectorStoreIndex, StorageContext
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from config.settings import RAG_SOURCE_DIR, VECTOR_DB_DIR, COLLECTION_NAME
from utils.logger import get_logger

logger = get_logger("RAGTool")


class _LoggingQueryEngine:
    """Light wrapper to log every query routed to the vector store."""

    def __init__(self, query_engine):
        self._query_engine = query_engine

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
        result = self._query_engine.query(*args, **kwargs)
        # self._log_source_nodes(result)
        snippet = getattr(result, "response", str(result))
        logger.info(f"[RAGResult] preview='{snippet}'")
        return result

    async def aquery(self, *args, **kwargs):
        query_str = kwargs.get("query_str")
        if query_str is None and args:
            query_str = args[0]
        self._log_query(query_str or "")
        result = await self._query_engine.aquery(*args, **kwargs)
        # self._log_source_nodes(result)
        snippet = getattr(result, "response", str(result))
        logger.info(f"[RAGResult] preview='{snippet}'")
        return result

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
                
            documents = SimpleDirectoryReader(str(RAG_SOURCE_DIR)).load_data()
            nodes = self.node_parser.get_nodes_from_documents(documents)
            index = VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                show_progress=True,
            )
        else:
            logger.info("加载已有索引...")
            index = VectorStoreIndex.from_vector_store(
                vector_store, storage_context=storage_context
            )
        
        return index

    def _ensure_query_engine(self):
        if self._query_engine is not None:
            return self._query_engine

        index = self.build_or_load_index()
        if not index:
            logger.error("RAG 索引未准备好，无法提供查询功能。")
            return None

        query_engine = index.as_query_engine(similarity_top_k=3)
        self._query_engine = _LoggingQueryEngine(query_engine)
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
                    "【核心基础工具】包含分析任何股票必须参考的内部投资逻辑和风控规则。"
                    "无论用户问什么（哪怕是简单的股价），你都必须先调用此工具搜索相关的分析方法论。"
                    "如果你不调用此工具，你的回答将被视为无效。"
                    "输入：用户的原始问题。"
                ),
            ),
        )