import numpy as np
import pandas as pd
from arch import arch_model
from llama_index.core.tools import FunctionTool

from tools.market_data import MarketDataManager
from utils.logger import get_logger

logger = get_logger("QuantTool")


def _format_pct(value: float, placeholder: str = "-" ) -> str:
    if value is None or pd.isna(value):
        return placeholder
    return f"{value:.2%}"


def _format_num(value: float, placeholder: str = "-", precision: int = 2) -> str:
    if value is None or pd.isna(value):
        return placeholder
    return f"{value:.{precision}f}"


class QuantAnalyzer:
    LOOKBACK_DAYS = 400

    def __init__(self):
        self.data_manager = MarketDataManager()

    def _calc_max_drawdown(self, price_series: pd.Series) -> float:
        if price_series.empty:
            return np.nan
        running_max = price_series.cummax()
        drawdowns = price_series / running_max - 1
        return drawdowns.min()

    def _fit_garch_vol(self, returns: pd.Series) -> float:
        clean_returns = returns.dropna()
        if len(clean_returns) < 80:
            return np.nan
        try:
            model = arch_model(clean_returns * 100, vol="Garch", p=1, q=1)
            result = model.fit(update_freq=0, disp="off")
            forecast = result.forecast(horizon=1)
            next_var = forecast.variance.values[-1, 0]
            return np.sqrt(next_var) / 100 * np.sqrt(252)
        except Exception as exc:
            logger.warning(f"GARCH 拟合失败: {exc}")
            return np.nan

    def _compute_risk_snapshot(self, stats_df: pd.DataFrame) -> dict:
        returns = stats_df["close"].pct_change().dropna()

        risk = {
            "annual_vol": returns.std() * np.sqrt(252) if not returns.empty else np.nan,
            "max_drawdown": self._calc_max_drawdown(stats_df["close"]),
            "var_95": np.quantile(returns, 0.05) if not returns.empty else np.nan,
        }

        if not returns.empty and not pd.isna(risk["var_95"]):
            tail = returns[returns <= risk["var_95"]]
            risk["cvar_95"] = tail.mean() if not tail.empty else risk["var_95"]
        else:
            risk["cvar_95"] = np.nan

        risk["garch_vol"] = self._fit_garch_vol(returns)
        return risk

    def analyze_asset_risk(self, stock_name: str) -> str:
        """整合多窗口行情 + 风险指标，输出量化快照。"""
        logger.info(f"[ToolCall] quant_analysis | stock_name={stock_name}")

        stock_meta = self.data_manager.resolve_stock(stock_name)
        if not stock_meta:
            return f"未找到与“{stock_name}”匹配的标的，无法执行量化分析。"

        ts_code = stock_meta["ts_code"]
        display_name = stock_meta.get("name", stock_name)

        stats_df = self.data_manager.get_enriched_price_data(
            ts_code, lookback_days=self.LOOKBACK_DAYS
        )

        if stats_df is None or stats_df.empty:
            return f"未能获取 {display_name} ({ts_code}) 的历史行情数据。"

        latest = stats_df.iloc[-1]
        risk = self._compute_risk_snapshot(stats_df)

        report_lines = [
            f"【{display_name} ({ts_code}) 量化分析快照】",
            "\n行情表现：",
            f"- 最新收盘价：{_format_num(latest.get('close'))} 元",
            f"- 近5/20/60/120日收益：{_format_pct(latest.get('ret_5'))} / "
            f"{_format_pct(latest.get('ret_20'))} / {_format_pct(latest.get('ret_60'))} / "
            f"{_format_pct(latest.get('ret_120'))}",
            f"- 近20/60/120日年化波动率：{_format_pct(latest.get('vol_20'))} / "
            f"{_format_pct(latest.get('vol_60'))} / {_format_pct(latest.get('vol_120'))}",
        ]

        report_lines.extend(
            [
                "\n风险评估：",
                f"- 历史年化波动率：{_format_pct(risk['annual_vol'])}",
                f"- 条件波动(GARCH)预测：{_format_pct(risk['garch_vol'])}",
                f"- 95% VaR / CVaR：{_format_pct(risk['var_95'])} / "
                f"{_format_pct(risk['cvar_95'])}",
                f"- 最大回撤：{_format_pct(risk['max_drawdown'])}",
            ]
        )

        report_lines.append(
            "\n解读建议：结合 RAG 理论摘要与上述行情/风险指标，评估估值及仓位安排，必要时搭配止损和分批策略。"
        )

        return "\n".join(report_lines)

    def get_tool(self):
        return FunctionTool.from_defaults(
            fn=self.analyze_asset_risk,
            description="高级金融量化分析工具。仅当用户询问'预测'、'量化分析'、'风险评估'时调用。"
        )