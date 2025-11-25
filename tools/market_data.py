import math
from datetime import datetime, timedelta

import pandas as pd
import tushare as ts
from llama_index.core.tools import FunctionTool

from config.settings import TUSHARE_TOKEN, CACHE_DIR
from utils.logger import get_logger, log_tool_io

logger = get_logger("MarketDataTool")

class MarketDataManager:
    def __init__(self):
        ts.set_token(TUSHARE_TOKEN)
        self.pro = ts.pro_api()
        self.stock_list_cache_path = CACHE_DIR / "stock_basic.csv"
        self.price_cache_dir = CACHE_DIR / "prices"
        self.price_cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_stock_list(self) -> pd.DataFrame:
        """获取股票列表（带文件缓存）"""
        today_str = datetime.now().strftime('%Y%m%d')
        
        # 检查缓存有效性
        if self.stock_list_cache_path.exists():
            file_date = datetime.fromtimestamp(self.stock_list_cache_path.stat().st_mtime).strftime('%Y%m%d')
            if file_date == today_str:
                return pd.read_csv(self.stock_list_cache_path, dtype={'symbol': str})

        logger.info("刷新股票列表缓存...")
        try:
            df = self.pro.query('stock_basic', exchange='', list_status='L', 
                              fields='ts_code,symbol,name,area,industry')
            df.to_csv(self.stock_list_cache_path, index=False)
            return df
        except Exception as e:
            logger.error(f"Tushare API 调用失败: {e}")
            if self.stock_list_cache_path.exists():
                logger.warning("使用旧缓存数据降级运行")
                return pd.read_csv(self.stock_list_cache_path, dtype={'symbol': str})
            return pd.DataFrame()

    @log_tool_io(logger, "market_data")
    def get_stock_market_data(self, stock_name: str, days_ago: int = 30) -> str:
        """
        Tool Function: 查询股票近期行情。
        """
        logger.info(
            f"[ToolCall] market_data | stock_name={stock_name} | days_ago={days_ago}"
        )
        df_basic = self._get_stock_list()
        
        if df_basic.empty:
            return "系统错误：无法获取股票列表。"

        # 模糊匹配
        matched = df_basic[df_basic['name'].str.contains(stock_name)]
        if matched.empty:
            return f"未找到股票：{stock_name}"
        
        ts_code = matched.iloc[0]['ts_code']
        real_name = matched.iloc[0]['name']
        
        start_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y%m%d')
        end_date = datetime.now().strftime('%Y%m%d')
        
        try:
            df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df.empty:
                return f"暂无 {real_name} 的行情数据。"
            
            df = df.sort_values('trade_date')
            latest_price = df.iloc[-1]['close']
            avg_price = df['close'].mean()
            
            return (f"【{real_name} ({ts_code})】\n"
                    f"最新价: {latest_price}\n"
                    f"近{days_ago}日均价: {avg_price:.2f}\n"
                    f"近期走势: {df.tail(5)['close'].tolist()}")
        except Exception as e:
            return f"数据查询异常: {str(e)}"

    def get_price_frame(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch standardized daily price data for downstream quant analysis."""
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
        """Enrich price frame with multi-window returns, volatility, and moving averages."""
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

    def resolve_stock(self, query: str) -> dict | None:
        """模糊匹配股票名称/代码，返回基础信息。"""
        df_basic = self._get_stock_list()
        if df_basic.empty:
            return None

        query = query.strip()

        exact_code = df_basic[
            df_basic["ts_code"].str.contains(query, case=False, na=False, regex=False)
        ]
        if not exact_code.empty:
            return exact_code.iloc[0].to_dict()

        fuzzy = df_basic[
            df_basic["name"].str.contains(query, na=False, regex=False)
        ]
        if fuzzy.empty:
            return None
        return fuzzy.iloc[0].to_dict()

    def get_enriched_price_data(
        self, ts_code: str, lookback_days: int = 400, windows: list[int] | None = None
    ) -> pd.DataFrame:
        """Convenience wrapper: 拉取指定区间行情并追加多窗口统计。"""
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y%m%d")
        price_df = self.get_price_frame(ts_code, start_date, end_date)
        if price_df.empty:
            return price_df
        return self.get_multi_window_stats(price_df, windows)

    def get_tool(self):
        return FunctionTool.from_defaults(
            fn=self.get_stock_market_data,
            description="用于查询中国A股股票的实时/历史行情数据。输入股票中文名即可（如'招商银行'）。"
        )