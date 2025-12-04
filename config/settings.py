import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# Tushare Token
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")

# 路径配置
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
CHART_CACHE_DIR = CACHE_DIR / "charts"
RAG_SOURCE_DIR = DATA_DIR / "rules_text"
VECTOR_DB_DIR = DATA_DIR / "chroma_db"

# 模型路径配置
HF_CACHE_DIR = "/root/nas-private/huggingface_cache"
LLM_PATH = os.path.join(HF_CACHE_DIR, "Qwen/Qwen3-4B-instruct")
EMBEDDING_PATH = os.path.join(HF_CACHE_DIR, "BAAI/bge-small-zh-v1.5")

# 向量库集合名称
COLLECTION_NAME = "finance_rules"

# Phoenix 监控配置
PHOENIX_ENABLED = os.getenv("PHOENIX_ENABLED", "true").lower() not in {"false", "0", "no"}
PHOENIX_PORT = int(os.getenv("PHOENIX_PORT", "6006"))
PHOENIX_HOST = os.getenv("PHOENIX_HOST", "127.0.0.1")

# 确保目录存在
for path in [CACHE_DIR, CHART_CACHE_DIR, RAG_SOURCE_DIR, VECTOR_DB_DIR]:
    path.mkdir(parents=True, exist_ok=True)