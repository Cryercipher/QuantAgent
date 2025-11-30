import asyncio
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.agent_runtime import runtime
from utils.logger import get_logger
from utils.tool_events import bind_queue, reset_queue

logger = get_logger("APIServer")

app = FastAPI(title="QuantAgent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


@app.on_event("startup")
async def _startup_event():
    await runtime.ensure_ready()
    logger.info("QuantAgent API 已准备就绪。")


@app.post("/api/chat")
async def chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message 不能为空")

    event_queue: asyncio.Queue = asyncio.Queue()

    async def run_agent_task():
        token = bind_queue(event_queue)
        try:
            result = await runtime.run_chat(message)
            await event_queue.put(
                {
                    "type": "final",
                    "answer": result["answer"],
                    "focus_hint": result.get("focus_hint"),
                    "focus_entry": result.get("focus_entry"),
                }
            )
        except Exception as exc:  # pragma: no cover - unexpected runtime failure
            logger.error("处理消息失败: %s", exc)
            await event_queue.put(
                {
                    "type": "error",
                    "message": "服务器处理失败，请稍后再试。",
                }
            )
        finally:
            reset_queue(token)

    agent_task = asyncio.create_task(run_agent_task())

    async def event_stream():
        await event_queue.put({"type": "status", "state": "accepted"})
        while True:
            event = await event_queue.get()
            data = json.dumps(event, ensure_ascii=False)
            yield f"data: {data}\n\n"
            if event.get("type") in {"final", "error"}:
                break
        await agent_task

    return StreamingResponse(event_stream(), media_type="text/event-stream")
