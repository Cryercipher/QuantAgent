import asyncio
import math
import re
import uuid
import difflib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import tushare as ts
from arch import arch_model
from llama_index.core.tools import FunctionTool
from pypinyin import lazy_pinyin

from config.settings import TUSHARE_TOKEN, CACHE_DIR
from utils.logger import get_logger, log_tool_io
from utils.tool_events import publish_event

logger = get_logger("QuantTool")


class MarketDataManager:
    def __init__(self, cache_callback=None):
        ts.set_token(TUSHARE_TOKEN)
        self.pro = ts.pro_api()
        self.stock_list_cache_path = CACHE_DIR / "stock_basic.csv"
        self.price_cache_dir = CACHE_DIR / "prices"
        self.price_cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_callback = cache_callback

    def _get_stock_list(self) -> pd.DataFrame:
        today_str = datetime.now().strftime('%Y%m%d')

        if self.stock_list_cache_path.exists():
            file_date = datetime.fromtimestamp(self.stock_list_cache_path.stat().st_mtime).strftime('%Y%m%d')
            if file_date == today_str:
                cached = pd.read_csv(self.stock_list_cache_path, dtype={'symbol': str})
                return self._enrich_stock_df(cached)

        logger.info("刷新股票列表缓存...")
        try:
            df = self.pro.query(
                'stock_basic',
                exchange='',
                list_status='L',
                fields='ts_code,symbol,name,fullname,enname,area,industry'
            )
            df = self._enrich_stock_df(df)
            df.to_csv(self.stock_list_cache_path, index=False)
            return df
        except Exception as e:
            logger.error(f"Tushare API 调用失败: {e}")
            if self.stock_list_cache_path.exists():
                logger.warning("使用旧缓存数据降级运行")
                cached = pd.read_csv(self.stock_list_cache_path, dtype={'symbol': str})
                return self._enrich_stock_df(cached)
            return pd.DataFrame()

    @log_tool_io(logger, "market_data")
    def get_stock_market_data(
        self,
        stock_name: str,
        days_ago: int = 30,
        ts_code: str | None = None,
        include_raw: bool = False,
    ) -> str | tuple[str, list[dict]]:
        logger.info(
            f"[ToolCall] market_data | stock_name={stock_name} | days_ago={days_ago}"
        )
        df_basic = self._get_stock_list()

        if df_basic.empty:
            payload = "系统错误：无法获取股票列表。"
            return (payload, []) if include_raw else payload

        stock_query = ts_code or stock_name
        stock_meta = self.resolve_stock(stock_query, df_basic=df_basic)
        if not stock_meta:
            payload = f"未找到股票：{stock_name}"
            return (payload, []) if include_raw else payload

        ts_code = stock_meta['ts_code']
        real_name = stock_meta.get('name', stock_name)

        start_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y%m%d')
        end_date = datetime.now().strftime('%Y%m%d')

        try:
            df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df.empty:
                payload = f"暂无 {real_name} 的行情数据。"
                return (payload, []) if include_raw else payload

            df = df.sort_values('trade_date')
            latest_price = df.iloc[-1]['close']
            avg_price = df['close'].mean()

            message = (
                f"【{real_name} ({ts_code})】\n"
                f"最新价: {latest_price}\n"
                f"近{days_ago}日均价: {avg_price:.2f}\n"
                f"近期走势: {df.tail(5)['close'].tolist()}"
            )
            recent = df.tail(3)["close"].tolist()
            summary = (
                f"{real_name} 最新价 {latest_price:.2f}，"
                f"近{days_ago}日均价 {avg_price:.2f}，"
                f"近3日收盘 {recent}"
            )
            self._write_cache_entry(
                ts_code,
                "行情",
                summary,
                metadata={"name": real_name}
            )
            recent = (
                df.tail(min(7, len(df)))
                .copy()
                .sort_values("trade_date")
            )
            raw_payload = []
            for _, row in recent.iterrows():
                raw_payload.append(
                    {
                        "date": pd.to_datetime(row["trade_date"]).strftime("%Y-%m-%d"),
                        "open": _safe_float(row.get("open")),
                        "close": _safe_float(row.get("close")),
                        "high": _safe_float(row.get("high")),
                        "low": _safe_float(row.get("low")),
                        "volume": _safe_float(row.get("vol")),
                    }
                )

            return (message, raw_payload) if include_raw else message
        except Exception as e:
            payload = f"数据查询异常: {str(e)}"
            return (payload, []) if include_raw else payload

    def get_price_frame(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        cache_file = self.price_cache_dir / f"{ts_code}_{start_date}_{end_date}.csv"
        if cache_file.exists():
            try:
                cached = pd.read_csv(cache_file, parse_dates=["trade_date"])
                return cached
            except Exception:
                logger.warning("价格缓存读取失败，重新拉取 Tushare 数据。")

        try:
            df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        except Exception as exc:
            logger.error(f"获取日线数据失败: {exc}")
            if cache_file.exists():
                return pd.read_csv(cache_file, parse_dates=["trade_date"])
            return pd.DataFrame()

        if df.empty:
            return df

        df = df.sort_values("trade_date").reset_index(drop=True)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        numeric_cols = [
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df.rename(columns={"trade_date": "date"}, inplace=True)
        df.to_csv(cache_file, index=False)
        return df

    def get_multi_window_stats(
        self, price_df: pd.DataFrame, windows: list[int] | None = None
    ) -> pd.DataFrame:
        if price_df is None or price_df.empty:
            return price_df

        df = price_df.copy()
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        if windows is None:
            windows = [5, 20, 60, 120]

        daily_returns = df["close"].pct_change()

        for window in windows:
            col_suffix = str(window)
            df[f"ret_{col_suffix}"] = df["close"].pct_change(periods=window)
            df[f"vol_{col_suffix}"] = (
                daily_returns.rolling(window=window).std() * math.sqrt(252)
            )
            df[f"ma_{col_suffix}"] = df["close"].rolling(window=window).mean()

        df["rolling_max_20"] = df["close"].rolling(window=20).max()
        df["rolling_min_20"] = df["close"].rolling(window=20).min()

        return df

    def resolve_stock(self, query: str, df_basic: pd.DataFrame | None = None) -> dict | None:
        df_basic = df_basic if df_basic is not None else self._get_stock_list()
        if df_basic.empty or not query:
            return None

        query = query.strip()
        query_ascii = _normalize_ascii(query)

        def _match(mask):
            matches = df_basic[mask]
            if not matches.empty:
                return matches.iloc[0].to_dict()
            return None

        for column in ("ts_code", "symbol"):
            if column in df_basic.columns:
                mask = df_basic[column].astype(str).str.contains(
                    query, case=False, na=False, regex=False
                )
                result = _match(mask)
                if result:
                    return result

        for column in ("name", "fullname"):
            if column in df_basic.columns:
                mask = df_basic[column].astype(str).str.contains(
                    query, case=False, na=False
                )
                result = _match(mask)
                if result:
                    return result

        if query_ascii:
            if "name_pinyin" in df_basic.columns:
                mask = df_basic["name_pinyin"].str.contains(query_ascii, na=False)
                result = _match(mask)
                if result:
                    return result
            if "enname_norm" in df_basic.columns:
                mask = df_basic["enname_norm"].str.contains(query_ascii, na=False)
                result = _match(mask)
                if result:
                    return result

        # 4. 模糊匹配 (Fuzzy Match)
        # 解决用户输入别名或不完整名称的问题 (如 "九号汽车" -> "九号公司-WD")
        candidates = []
        if "name" in df_basic.columns:
            candidates.extend(df_basic["name"].dropna().astype(str).tolist())
        # 也可以加入 fullname，但可能会增加干扰，视情况而定。这里先加上。
        if "fullname" in df_basic.columns:
            candidates.extend(df_basic["fullname"].dropna().astype(str).tolist())
            
        # cutoff=0.4: 允许较低的相似度以捕获 "九号汽车" vs "九号公司" 这种差异
        # n=1: 只取最相似的一个
        matches = difflib.get_close_matches(query, candidates, n=1, cutoff=0.4)
        if matches:
            best_match = matches[0]
            logger.info(f"模糊匹配成功: '{query}' -> '{best_match}'")
            # 重新查找对应的行
            mask = pd.Series(False, index=df_basic.index)
            if "name" in df_basic.columns:
                mask |= (df_basic["name"] == best_match)
            if "fullname" in df_basic.columns:
                mask |= (df_basic["fullname"] == best_match)
            
            result = _match(mask)
            if result:
                return result

        return None

        return None

    def get_enriched_price_data(
        self, ts_code: str, lookback_days: int = 400, windows: list[int] | None = None
    ) -> pd.DataFrame:
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y%m%d")
        price_df = self.get_price_frame(ts_code, start_date, end_date)
        if price_df.empty:
            return price_df
        return self.get_multi_window_stats(price_df, windows)

    def _write_cache_entry(self, ts_code: str, category: str, summary: str, metadata=None):
        if not self._cache_callback or not ts_code or not summary:
            return
        try:
            self._cache_callback(ts_code, category, summary, metadata or {})
        except Exception as exc:
            logger.warning(f"缓存行情摘要失败: {exc}")

    def _enrich_stock_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if "name" in df.columns:
            df["name_pinyin"] = df["name"].fillna("").apply(_to_pinyin)
        if "enname" in df.columns:
            df["enname_norm"] = df["enname"].fillna("").apply(_normalize_ascii)
        else:
            df["enname_norm"] = ""
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str)
        return df


def _format_pct(value: float, placeholder: str = "-" ) -> str:
    if value is None or pd.isna(value):
        return placeholder
    return f"{value:.2%}"


def _format_num(value: float, placeholder: str = "-", precision: int = 2) -> str:
    if value is None or pd.isna(value):
        return placeholder
    return f"{value:.{precision}f}"


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_pinyin(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return "".join(lazy_pinyin(text)).lower()


def _normalize_ascii(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", value.lower())


class QuantAnalyzer:
    LOOKBACK_DAYS = 400

    def __init__(self, cache_callback=None, data_manager: MarketDataManager | None = None):
        self.data_manager = data_manager or MarketDataManager()
        self._cache_callback = cache_callback

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

    @log_tool_io(logger, "quant_analysis")
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
        report = "\n".join(report_lines)
        summary = (
            f"{display_name} 收盘 {_format_num(latest.get('close'))} 元，"
            f"年化波动 {_format_pct(risk['annual_vol'])}，"
            f"最大回撤 {_format_pct(risk['max_drawdown'])}，"
            f"VaR95 {_format_pct(risk['var_95'])}，"
            f"GARCH波动 {_format_pct(risk['garch_vol'])}"
        )
        self._write_cache_entry(ts_code, summary, metadata={"name": display_name})
        return report

    def _write_cache_entry(self, ts_code: str, summary: str, metadata=None):
        if not self._cache_callback or not ts_code or not summary:
            return
        try:
            self._cache_callback(ts_code, "量化", summary, metadata or {})
        except Exception as exc:
            logger.warning(f"缓存量化摘要失败: {exc}")

class MarketInsightTool:
    """Unified tool that returns both recent market data and advanced quant analysis."""

    def __init__(self, cache_callback=None):
        self.market_manager = MarketDataManager(cache_callback=cache_callback)
        self.quant_analyzer = QuantAnalyzer(
            cache_callback=cache_callback,
            data_manager=self.market_manager,
        )

    def _analyze_stock_sync(self, stock_name: str, days_ago: int) -> str:
        stock_meta = self.market_manager.resolve_stock(stock_name)
        if not stock_meta:
            return f"未找到与“{stock_name}”匹配的标的，请输入更精确的名称或代码。"

        ts_code = stock_meta["ts_code"]
        display_name = stock_meta.get("name", stock_name)

        market_payload = self.market_manager.get_stock_market_data(
            display_name,
            days_ago=days_ago,
            ts_code=ts_code,
            include_raw=True,
        )
        if isinstance(market_payload, tuple):
            market_section, raw_bars = market_payload
        else:
            market_section, raw_bars = market_payload, []
        quant_section = self.quant_analyzer.analyze_asset_risk(display_name)

        segments = [
            "【基础行情速览】",
            (market_section or "暂无行情数据").strip(),
            "",
            "【量化风险分析】",
            (quant_section or "暂无量化分析结果").strip(),
        ]

        report = "\n".join(seg for seg in segments if seg)
        return report, {"raw_bars": raw_bars}

    async def analyze_stock(self, stock_name: str, days_ago: int = 30) -> str:
        call_id = uuid.uuid4().hex
        publish_event(
            {
                "type": "tool_status",
                "call_id": call_id,
                "tool": "quant_analysis_tool",
                "status": "running",
                "progress": 20,
                "meta": {"stock": stock_name, "window": days_ago},
            }
        )
        loop = asyncio.get_running_loop()
        try:
            report, extras = await loop.run_in_executor(
                None, self._analyze_stock_sync, stock_name, days_ago
            )
        except Exception as exc:
            publish_event(
                {
                    "type": "tool_status",
                    "call_id": call_id,
                    "tool": "quant_analysis_tool",
                    "status": "failed",
                    "progress": 100,
                    "error": str(exc),
                }
            )
            raise

        raw_bars = (extras or {}).get("raw_bars", []) if isinstance(extras, dict) else []
        publish_event(
            {
                "type": "tool_result",
                "call_id": call_id,
                "tool": "quant_analysis_tool",
                "status": "succeeded",
                "progress": 100,
                "result": report[:2000],
                "raw_bars": raw_bars,
            }
        )
        return report

    def get_tool(self):
        return FunctionTool.from_defaults(
            async_fn=self.analyze_stock,
            name="quant_analysis_tool",
            description=(
                "整合行情快照与量化风险指标的综合分析工具。"
                "输入股票名称后，自动输出当日价格、均线等关键信息并附带年度波动、VaR、回撤等高级量化结论。"
            ),
        )

    def get_search_tool(self):
        return FunctionTool.from_defaults(
            fn=self.search_stock_info,
            name="stock_search_tool",
            description=(
                "股票代码/名称模糊搜索工具。"
                "当用户输入的名称不准确（如使用产品名'九号电动车'、别名'宁王'）或不确定具体股票代码时，"
                "**必须**先调用此工具进行搜索，获取准确的股票名称和代码（ts_code）。"
            ),
        )

    def search_stock_info(self, query: str) -> str:
        """
        搜索股票信息，返回匹配列表。
        """
        call_id = uuid.uuid4().hex
        publish_event(
            {
                "type": "tool_status",
                "call_id": call_id,
                "tool": "stock_search_tool",
                "status": "running",
                "progress": 10,
                "meta": {"query": query},
            }
        )
        
        logger.info(f"[ToolCall] stock_search | query={query}")
        try:
            df_basic = self.market_manager._get_stock_list()
            if df_basic.empty:
                raise Exception("无法获取股票列表")

            query = query.strip()
            if not query:
                raise ValueError("搜索关键词为空")

            matches = []
            
            # 1. 精确/包含匹配
            mask = pd.Series(False, index=df_basic.index)
            for col in ["ts_code", "symbol", "name", "fullname", "enname", "name_pinyin"]:
                if col in df_basic.columns:
                    mask |= df_basic[col].astype(str).str.contains(query, case=False, na=False)
            
            exact_hits = df_basic[mask].head(5)
            for _, row in exact_hits.iterrows():
                matches.append(f"{row['name']} ({row['ts_code']}) - {row.get('fullname', '')}")

            # 2. 模糊匹配 (如果精确匹配少于3个)
            if len(matches) < 3:
                candidates = []
                if "name" in df_basic.columns:
                    candidates.extend(df_basic["name"].dropna().astype(str).tolist())
                if "fullname" in df_basic.columns:
                    candidates.extend(df_basic["fullname"].dropna().astype(str).tolist())
                
                # 使用 difflib 查找相似名称
                # 降低阈值到 0.25 以适配 "九号电动车" vs "九号公司-WD" (score~0.33) 这种情况
                fuzzy_names = difflib.get_close_matches(query, candidates, n=5, cutoff=0.25)
                
                for fname in fuzzy_names:
                    # 反查对应的行
                    fmask = (df_basic["name"] == fname) | (df_basic.get("fullname") == fname)
                    frows = df_basic[fmask]
                    for _, row in frows.iterrows():
                        item = f"{row['name']} ({row['ts_code']}) - {row.get('fullname', '')}"
                        if item not in matches:
                            matches.append(item)

            if not matches:
                result_text = f"未找到与 '{query}' 相关的股票。请尝试使用更准确的名称。"
            else:
                result_text = f"找到以下相关股票，请根据需要选择准确的名称调用分析工具：\n" + "\n".join(matches[:10])
            
            publish_event(
                {
                    "type": "tool_result",
                    "call_id": call_id,
                    "tool": "stock_search_tool",
                    "status": "succeeded",
                    "progress": 100,
                    "result": result_text,
                }
            )
            return result_text

        except Exception as e:
            publish_event(
                {
                    "type": "tool_status",
                    "call_id": call_id,
                    "tool": "stock_search_tool",
                    "status": "failed",
                    "progress": 100,
                    "error": str(e),
                }
            )
            return f"搜索出错: {str(e)}"