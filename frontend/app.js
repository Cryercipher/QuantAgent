const chatFeed = document.getElementById("chat-feed");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const focusHintEl = document.getElementById("focus-hint");
const toolMonitorEl = document.getElementById("tool-monitor");
const sessionClockEl = document.getElementById("session-clock");

const API_BASE = window.QUANT_AGENT_API || "http://localhost:8000";
const TOOL_NAME_MAP = {
  market_data: "行情快照",
  quant_analysis: "量化风险分析",
  quant_analysis_tool: "量化风险分析",
  financial_theory_tool: "金融理论检索"
};

const messages = [
  {
    id: "msg-0",
    role: "assistant",
    content:
      "欢迎来到 QuantAgent。请描述您的关注标的或策略，我会结合 RAG 知识库与行情工具输出 Markdown 报告。",
    timestamp: Date.now() - 600000,
    status: "ready",
    toolRuns: []
  }
];

function formatTime(ts) {
  const date = new Date(ts);
  return date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
}

function renderMessages() {
  chatFeed.innerHTML = "";
  messages.forEach((msg) => {
    const wrapper = document.createElement("div");
    wrapper.className = `chat-message ${msg.role}`;

    const header = document.createElement("div");
    header.className = "message-header";
    const name = msg.role === "assistant" ? "QuantAgent" : "您";
    header.innerHTML = `<strong>${name}</strong><span>${formatTime(msg.timestamp)}</span>`;
    wrapper.appendChild(header);

    const body = document.createElement("div");
    body.className = "message-body";
    const markdown = msg.content || (msg.status === "running" ? "努力思考中..." : "");
    body.innerHTML = marked.parse(markdown);
    wrapper.appendChild(body);

    if (msg.role === "assistant" && msg.status === "running") {
      const spinner = document.createElement("div");
      spinner.className = "spinner";
      wrapper.appendChild(spinner);
    }

    if (msg.toolRuns && msg.toolRuns.length) {
      wrapper.appendChild(renderToolPanel(msg.toolRuns));
    }

    chatFeed.appendChild(wrapper);
  });
  chatFeed.scrollTop = chatFeed.scrollHeight;
}

function renderToolPanel(toolRuns) {
  const panel = document.createElement("div");
  panel.className = "tool-panel";
  const title = document.createElement("h4");
  title.textContent = "工具执行";
  panel.appendChild(title);

  toolRuns.forEach((run) => {
    const row = document.createElement("div");
    row.className = "tool-row";

    const header = document.createElement("div");
    header.className = "tool-row-header";
    header.innerHTML = `<span>${run.displayName}</span>`;
    const status = document.createElement("span");
    status.className = `tool-status ${run.status}`;
    status.textContent = run.status;
    header.appendChild(status);
    row.appendChild(header);

    const progressTrack = document.createElement("div");
    progressTrack.className = "progress-track";
    const progressBar = document.createElement("div");
    progressBar.className = "progress-bar";
    progressBar.style.width = `${run.progress || 0}%`;
    progressTrack.appendChild(progressBar);
    row.appendChild(progressTrack);

    if (run.resultMarkdown || (run.resultChunks && run.resultChunks.length)) {
      const card = document.createElement("details");
      card.className = "tool-result-card";
      const summary = document.createElement("summary");
      summary.textContent = `${run.displayName} · 结果`;
      card.appendChild(summary);
      const body = document.createElement("div");
      body.className = "tool-result-body";
      if (run.resultChunks && run.resultChunks.length) {
        body.appendChild(renderChunkList(run.resultChunks));
      }
      if (run.resultMarkdown) {
        const markdownBlock = document.createElement("div");
        markdownBlock.innerHTML = marked.parse(run.resultMarkdown);
        body.appendChild(markdownBlock);
      }
      card.appendChild(body);
      row.appendChild(card);
    }

    panel.appendChild(row);
  });
  return panel;
}

function renderChunkList(chunks) {
  const container = document.createElement("div");
  container.className = "chunk-list";
  chunks.forEach((chunk, idx) => {
    const snippet = document.createElement("article");
    snippet.className = "chunk-snippet";

    const meta = document.createElement("div");
    meta.className = "chunk-meta";
    const metaParts = [];
    const indexLabel = chunk.index || chunk.index === 0 ? chunk.index : idx + 1;
    metaParts.push(`Chunk #${indexLabel}`);
    if (chunk.metadata) {
      const { file_name, title, topic } = chunk.metadata;
      if (file_name) metaParts.push(file_name);
      if (title && title !== file_name) metaParts.push(title);
      if (topic) metaParts.push(topic);
    }
    if (typeof chunk.score === "number") {
      metaParts.push(`score ${chunk.score.toFixed(2)}`);
    }
    meta.textContent = metaParts.join(" · ");

    const textBlock = document.createElement("pre");
    textBlock.textContent = chunk.text || "(空片段)";

    snippet.appendChild(meta);
    snippet.appendChild(textBlock);
    container.appendChild(snippet);
  });
  return container;
}

function renderToolMonitor() {
  const latestAssistant = [...messages]
    .reverse()
    .find((msg) => msg.role === "assistant" && msg.status === "running");
  if (!latestAssistant || !latestAssistant.toolRuns.length) {
    toolMonitorEl.innerHTML = "<p>暂无运行中的工具。</p>";
    return;
  }

  toolMonitorEl.innerHTML = "";
  latestAssistant.toolRuns.forEach((run) => {
    const row = document.createElement("div");
    row.className = "monitor-row";
    const name = document.createElement("span");
    name.textContent = run.displayName;
    const state = document.createElement("span");
    state.textContent = `${run.progress || 0}% · ${run.status}`;
    row.appendChild(name);
    row.appendChild(state);
    toolMonitorEl.appendChild(row);
  });
}

function addUserMessage(text) {
  messages.push({
    id: `msg-${Date.now()}`,
    role: "user",
    content: text,
    timestamp: Date.now(),
    status: "ready",
    toolRuns: []
  });
  renderMessages();
}

function upsertToolRun(message, event) {
  message.toolRuns = message.toolRuns || [];
  let run = message.toolRuns.find((item) => item.callId === event.call_id);
  if (!run) {
    run = {
      callId: event.call_id,
      toolId: event.tool,
      displayName: TOOL_NAME_MAP[event.tool] || event.tool,
      status: "queued",
      progress: 0,
      resultMarkdown: null
    };
    message.toolRuns.push(run);
  }
  if (event.status) {
    run.status = event.status;
    run.progress = event.progress ?? statusToProgress(event.status);
  }
  if (event.result) {
    run.resultMarkdown = event.result;
  }
  if (event.chunks) {
    run.resultChunks = event.chunks;
  }
}

function statusToProgress(status) {
  switch (status) {
    case "succeeded":
      return 100;
    case "failed":
      return 100;
    case "running":
      return 60;
    default:
      return 10;
  }
}

function updateFocusHint(entry) {
  if (!entry) {
    focusHintEl.textContent = "尚未聚焦具体标的。";
    return;
  }
  const name = entry.name || entry.ts_code || "未知标的";
  const code = entry.ts_code ? `（${entry.ts_code}）` : "";
  const summary = entry.summary ? `\n${entry.summary}` : "";
  focusHintEl.textContent = `目前聚焦：${name}${code}${summary ? `\n${summary}` : ""}`;
}

async function streamChatResponse(userText, assistantMsg) {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: userText })
  });

  if (!response.ok || !response.body) {
    throw new Error("后端响应异常");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    buffer = processSSEBuffer(buffer, assistantMsg, userText);
  }
}

function processSSEBuffer(buffer, assistantMsg, userText) {
  let idx;
  while ((idx = buffer.indexOf("\n\n")) >= 0) {
    const chunk = buffer.slice(0, idx).trim();
    buffer = buffer.slice(idx + 2);
    if (!chunk.startsWith("data:")) continue;
    const payload = chunk.slice(5).trim();
    if (!payload) continue;
    try {
      const event = JSON.parse(payload);
      handleServerEvent(event, assistantMsg, userText);
    } catch (err) {
      console.error("解析事件失败", err);
    }
  }
  return buffer;
}

function handleServerEvent(event, assistantMsg) {
  switch (event.type) {
    case "status":
      assistantMsg.content = "请求已受理，正在调度模型...";
      break;
    case "tool_status":
      upsertToolRun(assistantMsg, event);
      assistantMsg.content = "智能体正在调用工具...";
      break;
    case "tool_result":
      upsertToolRun(assistantMsg, event);
      break;
    case "message_chunk":
      assistantMsg.content = (assistantMsg.content || "") + (event.content || "");
      break;
    case "final":
      assistantMsg.status = "ready";
      assistantMsg.content = event.answer || "（未返回内容）";
      updateFocusHint(event.focus_entry || null);
      break;
    case "error":
      assistantMsg.status = "ready";
      assistantMsg.content = event.message || "服务器异常，请稍后重试。";
      updateFocusHint(null);
      break;
    case "done":
      break;
    default:
      break;
  }
  renderMessages();
  renderToolMonitor();
}

async function dispatchToBackend(userText) {
  sendBtn.disabled = true;
  const assistantMsg = {
    id: `assistant-${Date.now()}`,
    role: "assistant",
    content: "已接收问题，正在分析上下文...",
    timestamp: Date.now(),
    status: "running",
    toolRuns: []
  };
  messages.push(assistantMsg);
  renderMessages();
  renderToolMonitor();

  try {
    await streamChatResponse(userText, assistantMsg);
  } catch (error) {
    console.error(error);
    assistantMsg.status = "ready";
    assistantMsg.content = "连接后端失败，请检查服务是否启动。";
    renderMessages();
    renderToolMonitor();
    updateFocusHint(null);
  } finally {
    sendBtn.disabled = false;
  }
}

function handleSend() {
  const text = chatInput.value.trim();
  if (!text) return;
  chatInput.value = "";
  addUserMessage(text);
  dispatchToBackend(text);
}

sendBtn.addEventListener("click", handleSend);
chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    handleSend();
  }
});

function startSessionTimer() {
  const start = Date.now();
  setInterval(() => {
    const elapsed = Date.now() - start;
    const minutes = String(Math.floor(elapsed / 60000)).padStart(2, "0");
    const seconds = String(Math.floor((elapsed % 60000) / 1000)).padStart(2, "0");
    sessionClockEl.textContent = `${minutes}:${seconds}`;
  }, 1000);
}

updateFocusHint(null);
renderMessages();
renderToolMonitor();
startSessionTimer();
