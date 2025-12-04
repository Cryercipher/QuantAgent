# QuantAgent

QuantAgent 是一个基于大语言模型（LLM）的量化金融智能助手，旨在为投资者提供实时的市场洞察、深度的量化风险分析以及专业的金融理论支持。它集成了 Tushare 行情数据、专业量化模型（GARCH/VaR）、RAG 知识库和交互式图表生成功能。

## ✨ 核心功能

- **📈 深度行情分析**
  - 实时查询股票价格、涨跌幅及成交量。
  - 支持查询指定日期范围的**原始行情数据**（OHLCV），并以结构化表格呈现。
  - 自动模糊匹配股票名称与代码（如“九号电动” -> “九号公司”）。

- **🧮 量化风险评估**
  - 多周期收益率与波动率分析（5/20/60/120日）。
  - **风险指标计算**：历史波动率、GARCH 预测波动率、95% VaR（在险价值）、CVaR（条件在险价值）及最大回撤。

- **📚 金融 RAG 知识库**
  - 基于向量数据库（ChromaDB）的检索增强生成（RAG）。
  - 提供准确的金融术语定义、交易规则解释和理论依据，避免模型幻觉。

- **📊 交互式图表**
  - 自动生成近期 K 线图（支持均线、成交量）。
  - 图表直接嵌入对话流，直观展示市场走势。
 
<img width="472" height="323" alt="image" src="https://github.com/user-attachments/assets/32657286-63de-44a5-ab52-00b768938f0a" />


## 📂 项目结构

```text
QuantAgent/
├── config/             # 全局配置 (路径, Token, 模型参数)
├── core/               # 核心引擎
│   ├── agent_runtime.py    # Agent 运行时与状态管理
│   └── llm_factory.py      # LLM 与 Embedding 模型加载工厂
├── data/               # 数据存储
│   ├── cache/              # 行情与图表缓存
│   ├── chroma_db/          # 向量数据库文件
│   └── rules_text/         # RAG 原始语料
├── frontend/           # 前端界面 (Vanilla JS + SSE)
├── prompts/            # Prompt 工程 (System Prompts)
├── scripts/            # 实用脚本
│   ├── cli_chat.py         # 命令行对话测试工具
│   └── crawl_stock_doc.py  # 知识库爬虫工具
├── tools/              # 工具箱
│   ├── chart_visualizer.py # K线图生成工具
│   ├── knowledge_base.py   # RAG 检索工具
│   └── quant_analysis.py   # 行情与量化分析工具
├── utils/              # 通用工具 (日志, 事件总线)
├── server.py           # 后端启动入口
└── requirements.txt    # 项目依赖
```

## 🚀 快速开始

### 1. 环境准备

确保已安装 Python 3.10+。

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

项目依赖 Tushare 获取行情数据。请确保在 `config/settings.py` 中配置了有效的 Token，或设置环境变量：

```bash
export TUSHARE_TOKEN="your_tushare_token_here"
```

### 3. 启动后端服务

后端基于 FastAPI，提供 SSE 流式接口。

```bash
# 启动服务 (默认端口 8000)
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 启动前端页面

前端为纯静态项目，可直接通过 HTTP 服务器运行。

```bash
cd frontend
python -m http.server 8080
```

访问浏览器：[http://localhost:8080](http://localhost:8080)

## 🛠️ 开发与测试

- **命令行测试**：不依赖前端，直接在终端与 Agent 对话。
  ```bash
  python scripts/cli_chat.py
  ```

- **知识库更新**：
  将新的 `.txt` 或 `.md` 文件放入 `data/rules_text/`，重启服务后系统会自动重建索引。


