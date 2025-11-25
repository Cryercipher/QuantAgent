import tushare as ts
import pandas as pd
from datetime import datetime, timedelta
from llama_index.core.tools import FunctionTool
from config.settings import TUSHARE_TOKEN, CACHE_DIR
from utils.logger import get_logger

logger = get_logger("MarketDataTool")

class MarketDataManager:
    def __init__(self):
        ts.set_token(TUSHARE_TOKEN)
        self.pro = ts.pro_api()
        self.stock_list_cache_path = CACHE_DIR / "stock_basic.csv"

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

    def get_stock_market_data(self, stock_name: str, days_ago: int = 30) -> str:
        """
        Tool Function: 查询股票近期行情。
        """
        logger.info(f"查询股票: {stock_name}")
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

    def get_tool(self):
        return FunctionTool.from_defaults(
            fn=self.get_stock_market_data,
            description="用于查询中国A股股票的实时/历史行情数据。输入股票中文名即可（如'招商银行'）。"
        )