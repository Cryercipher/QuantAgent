import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# Tushare Token
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "d47f91ee6571a468cd4fae2f7396351260783b2de1370531d3eb5cc0")

# 路径配置
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
RAG_SOURCE_DIR = DATA_DIR / "rules_text"
VECTOR_DB_DIR = DATA_DIR / "chroma_db"

# 模型路径配置
HF_CACHE_DIR = "/root/nas-private/huggingface_cache"
LLM_PATH = os.path.join(HF_CACHE_DIR, "Qwen/Qwen3-4B-instruct")
EMBEDDING_PATH = os.path.join(HF_CACHE_DIR, "BAAI/bge-small-zh-v1.5")

# 向量库集合名称
COLLECTION_NAME = "finance_rules"

# 确保目录存在
for path in [CACHE_DIR, RAG_SOURCE_DIR, VECTOR_DB_DIR]:
    path.mkdir(parents=True, exist_ok=True)