import asyncio
import logging
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass
class _Binding:
    queue: asyncio.Queue
    loop: asyncio.AbstractEventLoop


_tool_event_queue: ContextVar[Optional[_Binding]] = ContextVar(
    "tool_event_queue", default=None
)
_thread_local = threading.local()
_logger = logging.getLogger("ToolEventBus")


def bind_queue(queue: asyncio.Queue):
    loop = asyncio.get_running_loop()
    binding = _Binding(queue=queue, loop=loop)
    token = _tool_event_queue.set(binding)
    stack = getattr(_thread_local, "queue_stack", [])
    stack.append(binding)
    _thread_local.queue_stack = stack
    return token


def reset_queue(token):
    if token is not None:
        _tool_event_queue.reset(token)
    stack = getattr(_thread_local, "queue_stack", [])
    if stack:
        stack.pop()
        _thread_local.queue_stack = stack


def _get_binding() -> Optional[_Binding]:
    binding = _tool_event_queue.get()
    if binding is not None:
        return binding
    stack = getattr(_thread_local, "queue_stack", [])
    if stack:
        return stack[-1]
    return None


def publish_event(event: dict[str, Any]):
    binding = _get_binding()
    if not binding:
        _logger.debug("tool event dropped (no queue binding): %s", event.get("type"))
        return
    payload = {
        **event,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    try:
        current_loop = asyncio.get_running_loop()
        if current_loop is binding.loop:
            binding.loop.create_task(binding.queue.put(payload))
        else:
            asyncio.run_coroutine_threadsafe(binding.queue.put(payload), binding.loop)
    except RuntimeError:
        asyncio.run_coroutine_threadsafe(binding.queue.put(payload), binding.loop)
