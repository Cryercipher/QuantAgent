import base64
from datetime import datetime, timedelta
from io import BytesIO
from typing import Optional
import uuid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplfinance as mpf  # noqa: E402
import pandas as pd  # noqa: E402
from llama_index.core.tools import FunctionTool  # noqa: E402

from config.settings import CHART_CACHE_DIR  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.tool_events import publish_event  # noqa: E402
from .quant_analysis import MarketDataManager  # noqa: E402

logger = get_logger("ChartTool")

# 配置中文字体 (解决 Matplotlib 中文乱码)
try:
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]
    plt.rcParams["axes.unicode_minus"] = False
    # 创建支持中文的 mplfinance 样式
    MPF_STYLE = mpf.make_mpf_style(
        base_mpf_style="yahoo", rc={"font.family": "WenQuanYi Zen Hei"}
    )
except Exception as exc:
    logger.warning(f"字体配置异常: {exc}")
    MPF_STYLE = "yahoo"


class CandlestickChartTool:
    """Generate recent candlestick charts for a requested stock."""

    def __init__(self, cache_callback=None):
        self.market_manager = MarketDataManager(cache_callback=cache_callback)
        self._cache_callback = cache_callback

    async def generate_candlestick_chart(
        self, stock_name: str, lookback_days: int = 60
    ) -> str:
        logger.info(
            "[ToolCall] candlestick_chart | stock=%s window=%s",
            stock_name,
            lookback_days,
        )
        call_id = uuid.uuid4().hex
        publish_event(
            {
                "type": "tool_status",
                "call_id": call_id,
                "tool": "candlestick_chart_tool",
                "status": "running",
                "progress": 20,
                "meta": {"stock": stock_name, "window": lookback_days},
            }
        )

        # 使用 asyncio.to_thread 运行阻塞的绘图逻辑，同时保留 contextvars
        import asyncio
        return await asyncio.to_thread(
            self._generate_chart_sync, call_id, stock_name, lookback_days
        )

    def _generate_chart_sync(
        self, call_id: str, stock_name: str, lookback_days: int
    ) -> str:
        stock_meta = self.market_manager.resolve_stock(stock_name)
        if not stock_meta:
            error_msg = f"未找到与“{stock_name}”匹配的标的，无法绘制K线。"
            self._emit_failure(call_id, error_msg)
            return error_msg

        ts_code = stock_meta["ts_code"]
        display_name = stock_meta.get("name", stock_name)

        lookback_days = max(20, min(lookback_days, 240))
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days * 2)
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        price_df = self.market_manager.get_price_frame(ts_code, start_str, end_str)
        if price_df is None or price_df.empty:
            error_msg = f"未能获取 {display_name} ({ts_code}) 的行情数据，无法绘制K线。"
            self._emit_failure(call_id, error_msg)
            return error_msg

        price_df = price_df.copy()
        price_df["date"] = pd.to_datetime(price_df["date"])
        column_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "vol": "Volume",
            "amount": "Amount",
        }
        price_df.rename(columns=column_map, inplace=True)
        for required in ("Open", "High", "Low", "Close"):
            if required not in price_df.columns:
                error_msg = f"行情数据缺少 {required} 列，无法绘制K线，请稍后重试。"
                self._emit_failure(call_id, error_msg)
                return error_msg
        if "Volume" not in price_df.columns:
            price_df["Volume"] = float("nan")

        price_df.set_index("date", inplace=True)
        window_df = price_df.tail(min(lookback_days, len(price_df)))

        if window_df.empty:
            error_msg = f"{display_name} ({ts_code}) 在所选区间内缺少有效数据。"
            self._emit_failure(call_id, error_msg)
            return error_msg

        fig, _ = mpf.plot(
            window_df,
            type="candle",
            mav=(5, 20),
            volume=True,
            style=MPF_STYLE,
            figratio=(12, 6),
            figscale=1.15,
            title=f"{display_name} ({ts_code}) 近{len(window_df)}日K线",
            ylabel="价格 (CNY)",
            ylabel_lower="成交量",
            returnfig=True,
        )
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        png_bytes = buffer.getvalue()
        # image_b64 = base64.b64encode(png_bytes).decode("ascii")

        interval_text = (
            f"区间：{window_df.index[0].date()} 至 {window_df.index[-1].date()}"
        )
        chart_id = self._persist_chart_image(ts_code, png_bytes)

        # 仅在摘要中使用文本，避免将 Base64 放入 Markdown 导致 SSE 包过大
        markdown = f"{display_name} K线图已生成（ID: {chart_id}）。\n\n{interval_text}"

        self._cache_chart_summary(ts_code, display_name, window_df)
        publish_event(
            {
                "type": "tool_result",
                "call_id": call_id,
                "tool": "candlestick_chart_tool",
                "status": "succeeded",
                "progress": 100,
                "result": markdown,
                "chart_id": chart_id,
                # "chart_data": image_b64,  # 移除 Base64 以减小 SSE 负载，前端将通过 /api/charts/{id} 加载
                "interval": interval_text,
                "metadata": {"ts_code": ts_code, "name": display_name},
            }
        )
        return f"{display_name} 的近期K线图已生成，可在图表面板查看。"

    @staticmethod
    def _persist_chart_image(ts_code: str, png_bytes: bytes) -> str:
        sanitized = (ts_code or "chart").replace("/", "_").replace(".", "_")
        chart_id = f"{sanitized}_{uuid.uuid4().hex}"
        file_path = CHART_CACHE_DIR / f"{chart_id}.png"
        try:
            file_path.write_bytes(png_bytes)
        except Exception as exc:
            logger.warning(f"写入图表缓存失败: {exc}")
        return chart_id

    @staticmethod
    def _emit_failure(call_id: str, error_msg: str):
        publish_event(
            {
                "type": "tool_status",
                "call_id": call_id,
                "tool": "candlestick_chart_tool",
                "status": "failed",
                "progress": 100,
                "error": error_msg,
            }
        )

    def _cache_chart_summary(
        self, ts_code: Optional[str], display_name: str, df: pd.DataFrame
    ):
        if not self._cache_callback or not ts_code or df.empty:
            return
        latest = df.iloc[-1]
        # 兼容列名大小写
        close_price = latest.get("Close") if "Close" in latest else latest.get("close")
        summary = (
            f"{display_name} 近{len(df)}日K线已生成，最新收盘 {_format_num(close_price)} 元"
        )
        try:
            self._cache_callback(
                ts_code,
                "图表",
                summary,
                metadata={"name": display_name},
            )
        except Exception as exc:
            logger.warning(f"缓存图表摘要失败: {exc}")

    def get_tool(self) -> FunctionTool:
        return FunctionTool.from_defaults(
            fn=self.generate_candlestick_chart,
            name="candlestick_chart_tool",
            description=(
                "绘制指定股票的近期K线图，并返回图像（支持64日内的开收等价数据可视化）。"
                "使用场景：需要快速获得近期走势的视觉化判断。"
            ),
        )


def _format_num(value, placeholder="-"):
    try:
        num = float(value)
    except (TypeError, ValueError):
        return placeholder
    return f"{num:.2f}"
