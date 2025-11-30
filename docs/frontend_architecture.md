# Enterprise Chat UI Architecture

## Objectives
- Mirror an enterprise chat console where portfolio managers converse with the QuantAgent.
- Surface tool invocation lifecycle in real time (queued → running → completed / failed) with progress indicators to reduce perceived latency.
- Keep responses readable by rendering Markdown and moving verbose analytics into collapsible "deep dive" cards.
- Remain backend-agnostic so the same front-end can consume either the current CLI agent or a future API service.

## Page Layout
1. **Global Shell**: App header with product name, connection indicator, and session timer.
2. **Conversation Pane** (left, ~70% width):
   - Sticky system banner for compliance tips.
   - Scrollable message feed with chat bubbles (user right-aligned, assistant left-aligned).
   - Each assistant message wraps:
     - Markdown-rendered answer.
     - Tool activity stack (real-time progress bars + status chips).
     - Collapsible analytics cards (per tool) that default to collapsed but remember the user toggle state per message.
3. **Context Pane** (right, ~30% width, optional MVP+1):
   - Latest tool cache summaries.
   - Quick actions ("上传财报", "生成投资笔记").
4. **Composer**: Docked bottom bar with text area, file-attachment affordance, and shortcut hints. Enter sends; `Shift+Enter` inserts newline.

## Interaction Flow
1. User submits prompt → front-end immediately appends a pending assistant bubble with spinner and empty tool stack.
2. Backend streams events over WebSocket / SSE using the contract below. UI updates progress bars per tool event and renders partial markdown tokens if available.
3. When `assistant_message.final` event arrives, the UI freezes tool progress, persists collapsible cards, and unlocks composer.

### Event Contract (WebSocket or SSE)
```json
{
  "type": "tool_status",            // | message_chunk | tool_result | error
  "conversation_id": "uuid",
  "message_id": "uuid",
  "tool_name": "quant_analysis_tool",
  "display_name": "量化风险分析",
  "status": "queued|running|succeeded|failed",
  "progress": 42,                     // 0-100
  "metadata": { "ts_code": "600519.SH" }
}
```

`message_chunk` events stream markdown tokens:
```json
{
  "type": "message_chunk",
  "role": "assistant",
  "message_id": "uuid",
  "content": "最新收盘价为 1450.5 元..."
}
```

`tool_result` events deliver the verbose payload for collapsible cards:
```json
{
  "type": "tool_result",
  "tool_name": "quant_analysis_tool",
  "title": "量化风险分析",
  "body_markdown": "## 行情表现..."
}
```

## Component Hierarchy (React / Web Components)
- `<AppShell>`: owns global layout + theme provider.
- `<ChatFeed>`: virtualized scroll list.
  - `<ChatMessage>`:
    - `<MessageHeader>` (avatar, timestamp, token stats placeholder).
    - `<MarkdownBody>` (renders with `marked` / `react-markdown`).
    - `<ToolActivityPanel>`:
      - `<ToolActivityRow>` (name, status badge, animated progress bar).
    - `<ToolResultAccordion>` (map tool results → collapsible cards).
- `<Composer>`: multi-line input, attachments, send CTA.
- `<ToolTimelineDrawer>` (optional) for audit playback.

## Visual Language
- Colors: Navy (#0B132B) for headers, Slate (#1C2541) chat background, Accent Cyan (#3A86FF) for progress, Success Green (#2EC4B6), Warning Amber (#FF9F1C).
- Typography: "Inter", fallback to system.
- Animation: 200 ms ease-out for card expansion, CSS-driven progress smoothing.

## Integration Hooks
1. **Backend Refactor**: Extract agent bootstrap (model + tools + memory) into `core/agent_runtime.py` so both CLI (`app.py`) and HTTP server reuse it.
2. **Event Streaming**: Introduce `utils/tool_events.py` (ContextVar-backed) so every FunctionTool can emit lifecycle events without coupling to UI runtime.
3. **API Layer** (FastAPI suggestion):
   - `POST /api/chat` accepts `{ conversation_id?, message }` and returns `202 Accepted` with stream endpoint.
   - `GET /api/stream/{conversation_id}` uses SSE. Server reads from per-conversation `asyncio.Queue` and emits events defined earlier.
4. **State Persistence**: Store chat history + tool artifacts in Redis (for multi-instance) or server memory (MVP) keyed by conversation id.

## Future Enhancements
- Multi-turn context pane summarizing last tool usage.
- Downloadable PDF briefing generated from latest assistant card.
- Role-based guardrails (compliance view vs analyst view).
- Notification center when long-running backtests finish asynchronously.
