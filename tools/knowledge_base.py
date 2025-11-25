import chromadb
from llama_index.core import (
    SimpleDirectoryReader, VectorStoreIndex, StorageContext
)
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
        return self._query_engine.query(*args, **kwargs)

    async def aquery(self, *args, **kwargs):
        query_str = kwargs.get("query_str")
        if query_str is None and args:
            query_str = args[0]
        self._log_query(query_str or "")
        return await self._query_engine.aquery(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._query_engine, item)

class FinancialKnowledgeBase:
    def __init__(self):
        self.db_client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        self.collection = self.db_client.get_or_create_collection(COLLECTION_NAME)

    def build_or_load_index(self):
        vector_store = ChromaVectorStore(chroma_collection=self.collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if self.collection.count() == 0:
            logger.info("构建新索引...")
            if not any(RAG_SOURCE_DIR.iterdir()):
                logger.warning(f"数据目录 {RAG_SOURCE_DIR} 为空！")
                return None
                
            documents = SimpleDirectoryReader(str(RAG_SOURCE_DIR)).load_data()
            index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context, show_progress=True
            )
        else:
            logger.info("加载已有索引...")
            index = VectorStoreIndex.from_vector_store(
                vector_store, storage_context=storage_context
            )
        
        return index

    def get_tool(self):
        index = self.build_or_load_index()
        if not index:
            return None
            
        query_engine = index.as_query_engine(similarity_top_k=3)
        logging_query_engine = _LoggingQueryEngine(query_engine)
        
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