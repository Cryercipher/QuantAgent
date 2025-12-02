# QuantAgent

QuantAgent 是一个基于大语言模型的量化金融智能助手，集成了实时行情查询、技术指标分析、金融理论检索（RAG）和图表绘制功能。

## 快速开始

### 1. 环境准备

确保已安装 Python 3.10+。

```bash
pip install -r requirements.txt
```

### 2. 启动后端服务

后端基于 FastAPI，默认运行在 8000 端口。

```bash
# 在项目根目录下运行
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

启动成功后，API 文档地址：[http://localhost:8000/docs](http://localhost:8000/docs)

### 3. 启动前端页面

前端为纯静态 HTML/JS 项目，建议使用 Python 内置 HTTP 服务器启动。

```bash
# 新开一个终端窗口
cd frontend
python -m http.server 8080
```

然后在浏览器中访问：[http://localhost:8080](http://localhost:8080)

> **注意**：前端会自动连接 `http://<当前主机IP>:8000` 的后端接口。请确保后端服务已启动。

## 功能模块

- **行情快照**：查询股票实时价格、涨跌幅。
- **量化分析**：计算波动率、RSI、MACD 等技术指标。
- **K线图表**：生成并展示股票历史走势图（支持中文显示）。
- **理论检索**：基于 RAG 技术回答金融理论问题（如“什么是两级切分法”）。

## 目录结构

- `server.py`: 后端入口文件。
- `core/`: 核心逻辑（LLM 工厂、Agent 运行时）。
- `tools/`: 工具集（行情、RAG、绘图）。
- `frontend/`: 前端静态资源。
- `data/`: 数据存储（ChromaDB、缓存图表）。
