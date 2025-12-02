const chatFeed = document.getElementById("chat-feed");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
// const focusHintEl = document.getElementById("focus-hint");
// const toolMonitorEl = document.getElementById("tool-monitor");
const sessionClockEl = document.getElementById("session-clock");

// 自动推断 API 地址：如果当前页面是 localhost，则默认 localhost:8000
// 如果是远程 IP，则尝试连接同 IP 的 8000 端口
const API_BASE = window.QUANT_AGENT_API || `http://${window.location.hostname}:8000`;
const TOOL_NAME_MAP = {
  market_data: "行情快照",
  quant_analysis: "量化风险分析",
  quant_analysis_tool: "量化风险分析",
  financial_theory_tool: "金融理论检索",
  candlestick_chart_tool: "K线图生成",
  stock_search_tool: "股票模糊搜索",
  StockSearchTool: "股票模糊搜索",
  stock_search: "股票模糊搜索"
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

    const chartGallery = renderInlineCharts(msg);
    if (chartGallery) {
      wrapper.appendChild(chartGallery);
    }

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

function renderInlineCharts(msg) {
  if (msg.role !== "assistant") {
    return null;
  }

  const collected = [];
  const seenIds = new Set();
  if (Array.isArray(msg.inlineCharts)) {
    msg.inlineCharts.forEach((payload) => {
      collected.push(payload);
      if (payload && typeof payload === "object" && payload.id) {
        seenIds.add(payload.id);
      }
    });
  }

  if (!collected.length && Array.isArray(msg.toolRuns)) {
    msg.toolRuns.forEach((run) => {
      const isChartTool = run.toolId === "candlestick_chart_tool";
      const hasAsset = run.chartId || run.chartData;
      if (!isChartTool || !hasAsset) {
        return;
      }
      if (run.chartId && seenIds.has(run.chartId)) {
        return;
      }
      collected.push({
        id: run.chartId,
        title: run.chartTitle || run.displayName || "K线图",
        interval: run.chartInterval,
        dataUri: run.chartData,
      });
      if (run.chartId) {
        seenIds.add(run.chartId);
      }
    });
  }

  if (!collected.length) {
    return null;
  }

  console.log("Rendering inline charts:", collected); // Debug log

  const gallery = document.createElement("div");
  gallery.className = "inline-chart-gallery";
  collected.forEach((chartPayload) => {
    if (typeof chartPayload === "string") {
      const fallback = document.createElement("div");
      fallback.className = "chart-card";
      fallback.innerHTML = marked.parse(chartPayload);
      gallery.appendChild(fallback);
      return;
    }
    gallery.appendChild(
      createChartElement({
        chartId: chartPayload.id,
        title: chartPayload.title,
        interval: chartPayload.interval,
        dataUri: chartPayload.dataUri,
      })
    );
  });

  return gallery;
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

    if (
      run.resultMarkdown ||
      (run.resultChunks && run.resultChunks.length) ||
      (run.rawBars && run.rawBars.length) ||
      run.chartId
    ) {
      const card = document.createElement("details");
      card.className = "tool-result-card";
      const summary = document.createElement("summary");
      summary.textContent = `${run.displayName} · 结果`;
      card.appendChild(summary);
      const body = document.createElement("div");
      body.className = "tool-result-body";
      if (run.rawBars && run.rawBars.length) {
        body.appendChild(renderRawBarsTable(run.rawBars));
      }
      if (run.resultChunks && run.resultChunks.length) {
        body.appendChild(renderChunkList(run.resultChunks));
      }
      if (run.chartId) {
        body.appendChild(
          createChartElement({
            chartId: run.chartId,
            title: run.chartTitle,
            interval: run.chartInterval,
            dataUri: run.chartData,
          })
        );
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

function createChartElement({ chartId, title, interval, dataUri }) {
  const card = document.createElement("div");
  card.className = "chart-card";
  const imgSrc = dataUri || buildChartUrl(chartId);
  console.log("Creating chart element:", { chartId, imgSrc }); // Debug log
  if (imgSrc) {
    const img = document.createElement("img");
    img.src = imgSrc;
    img.alt = title || "K线图";
    img.onerror = (e) => console.error("Image load failed:", imgSrc, e); // Debug log
    card.appendChild(img);
  }
  if (interval) {
    const caption = document.createElement("p");
    caption.textContent = interval;
    card.appendChild(caption);
  }
  return card;
}

function buildChartUrl(chartId) {
  if (!chartId) {
    return "";
  }
  return `${API_BASE}/api/charts/${chartId}`;
}

function renderRawBarsTable(bars) {
  const wrapper = document.createElement("div");
  wrapper.className = "raw-bars-wrapper";
  const table = document.createElement("table");
  table.className = "raw-bars-table";
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  ["日期", "开盘", "收盘", "最高", "最低", "成交量"].forEach((label) => {
    const th = document.createElement("th");
    th.textContent = label;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  bars.forEach((bar) => {
    const row = document.createElement("tr");
    const fields = [
      bar.date,
      formatNumberCell(bar.open),
      formatNumberCell(bar.close),
      formatNumberCell(bar.high),
      formatNumberCell(bar.low),
      formatVolumeCell(bar.volume),
    ];
    fields.forEach((value) => {
      const td = document.createElement("td");
      td.textContent = value;
      row.appendChild(td);
    });
    tbody.appendChild(row);
  });
  table.appendChild(tbody);
  wrapper.appendChild(table);
  return wrapper;
}

function formatNumberCell(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "-";
  }
  return value.toFixed(2);
}

function formatVolumeCell(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "-";
  }
  return Math.round(value).toLocaleString();
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
  // 侧栏已移除，此函数不再需要更新 UI
  /*
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
  */
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
  // Debug log to check tool name mapping
  if (!TOOL_NAME_MAP[event.tool]) {
    console.warn(`Tool name not found in map: '${event.tool}'. Available keys:`, Object.keys(TOOL_NAME_MAP));
  }
  
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
  if (event.chart_id) {
    run.chartId = event.chart_id;
    run.chartInterval = event.interval;
    const name = event.metadata && event.metadata.name;
    run.chartTitle = name || run.displayName;
  }
  if (event.chart_data) {
    run.chartData = `data:image/png;base64,${event.chart_data}`;
  }
  if (event.chunks) {
    run.resultChunks = event.chunks;
  }
  if (event.raw_bars) {
    run.rawBars = event.raw_bars;
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
  // 侧栏已移除，此函数不再需要更新 UI
  // if (!entry) {
  //   focusHintEl.textContent = "尚未聚焦具体标的。";
  //   return;
  // }
  // const name = entry.name || entry.ts_code || "未知标的";
  // const code = entry.ts_code ? `（${entry.ts_code}）` : "";
  // const summary = entry.summary ? `\n${entry.summary}` : "";
  // focusHintEl.textContent = `目前聚焦：${name}${code}${summary ? `\n${summary}` : ""}`;
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
  console.log("Event received:", event.type, event); // Debug log
  switch (event.type) {
    case "status":
      assistantMsg.content = "请求已受理，正在调度模型...";
      break;
    case "tool_status":
      upsertToolRun(assistantMsg, event);
      assistantMsg.content = "智能体正在调用工具...";
      break;
    case "tool_result":
      console.log("Tool result:", event); // Debug log
      upsertToolRun(assistantMsg, event);
      if (event.tool === "candlestick_chart_tool" && (event.chart_id || event.result)) {
        assistantMsg.inlineCharts = assistantMsg.inlineCharts || [];
        const dataUri = event.chart_data ? `data:image/png;base64,${event.chart_data}` : null;
        if (event.chart_id) {
          console.log("Adding inline chart:", event.chart_id); // Debug log
          const chartTitle = event.metadata && event.metadata.name ? event.metadata.name : "K线图";
          assistantMsg.inlineCharts.push({
            id: event.chart_id,
            title: chartTitle,
            interval: event.interval,
            dataUri,
          });
        } else if (event.result) {
          // Fallback for markdown result if needed, though we prefer structured data
          // assistantMsg.inlineCharts.push(event.result);
        }
      }
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
    toolRuns: [],
    inlineCharts: []
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
