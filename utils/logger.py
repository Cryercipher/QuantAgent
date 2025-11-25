import logging
import sys
from functools import wraps
import json


def _truncate_text(text: str, limit: int = 300) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def log_tool_io(logger: logging.Logger, label: str, preview_chars: int = 300):
    """Decorator to log tool inputs/outputs with truncated previews."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                args_preview = _truncate_text(str(args), preview_chars)
                kwargs_preview = _truncate_text(str(kwargs), preview_chars)
                logger.info(
                    f"[ToolInput] {label} args={args_preview} kwargs={kwargs_preview}"
                )
                result = func(*args, **kwargs)

                if isinstance(result, (dict, list)):
                    try:
                        output_repr = json.dumps(result, ensure_ascii=False)
                    except Exception:
                        output_repr = str(result)
                else:
                    output_repr = str(result)

                logger.info(
                    f"[ToolOutput] {label} result={_truncate_text(output_repr, preview_chars)}"
                )
                return result
            except Exception as exc:
                logger.error(f"[ToolError] {label} err={exc}")
                raise

        return wrapper

    return decorator

def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger