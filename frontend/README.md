# QuantAgent 前端原型

该目录提供一个纯静态的企业级聊天界面原型，用于演示以下交互能力：

- 双列布局（左侧聊天、右侧工具/上下文面板）。
- Markdown 渲染：回答文本使用 `marked` 在浏览器端解析。
- 工具调用动态进度：每个工具行带有状态徽章和实时进度条。
- 结果折叠卡片：行情、量化等长文本默认收起，点击展开查看。
- 会话焦点 & 快捷操作面板，方便扩展上传研报或导出纪要。
- 智能触发逻辑：根据用户输入动态判断是否需要调度行情/量化工具，简单问候不触发工具，指定标的时才展示进度与结果。

## 运行服务

1. 启动 FastAPI 后端（建议在仓库根目录执行）：

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

2. 启动前端静态站点：

```bash
cd /root/nas-private/QuantAgent/frontend
python -m http.server 8080
```

3. 打开 <http://localhost:8080/> 预览界面；若后端端口不同，可在浏览器控制台输入 `window.QUANT_AGENT_API = 'http://localhost:8000'` 后刷新，或在部署时通过 `script` 前注入同名变量以重定向 API 地址。

## 集成指引

1. `/api/chat` 现已提供 SSE 流事件（`status`、`tool_status`、`tool_result`、`final`），前端已默认订阅；接入其他 runtime 时仅需遵守该事件协议。
2. Markdown 内容直接由后端返回，可按需在 UI 端增加 `DOMPurify` 等安全处理。
3. `focusHintEl`、`toolMonitorEl` 的数据来自后端的 `focus_entry` 与工具事件，如需更详细的上下文可扩展响应字段。
4. 若采用 React / Vue，可把当前 `messages`、`toolRuns` 结构迁移为组件状态，复用 `styles.css` 与 SSE 解析逻辑。