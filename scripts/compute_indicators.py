import talib
import pandas as pd
import duckdb
import os
import json
import requests
from bs4 import BeautifulSoup
from pykrx import stock
from datetime import datetime
from io import StringIO
import yfinance as yf
from datetime import timedelta
from multiprocessing import Pool, cpu_count
import time  # 추가: sleep용

# 수정: 환경 변수로 DB_PATH 동적화
DATA_DIR = os.getenv('DATA_DIR', './data')  # 로컬 기본: ./data, 배포: /tmp/data
os.makedirs(DATA_DIR, exist_ok=True)
META_DIR = os.path.join(DATA_DIR, 'meta')
os.makedirs(META_DIR, exist_ok=True)
DB_PATH = os.path.join(META_DIR, 'universe.db')

def get_last_trading_date():
    today = datetime.now()
    while True:
        if today.weekday() < 5:
            return today.strftime('%Y%m%d')
        today -= timedelta(days=1)

def get_kr_tickers():
    today = get_last_trading_date()
    df_kr = stock.get_market_cap_by_ticker(today)
    return df_kr.sort_values('시가총액', ascending=False).head(500).index.tolist(), df_kr

def get_us_symbols():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    df_us = pd.read_html(StringIO(str(table)))[0]
    return df_us['Symbol'].tolist()[:1000]

def compute_indicators_wrapper(args):
    symbol, market, df_kr = args
    return compute_indicators(symbol, market, df_kr)

def compute_indicators(symbol, market='US', df_kr=None):
    try:
        # 수정: base_dir 환경 변수로 동적화
        base_dir = DATA_DIR
        if market == 'US':
            daily_path = os.path.join(base_dir, 'us_daily', f"{symbol}.parquet")
            weekly_path = os.path.join(base_dir, 'us_weekly', f"{symbol}.parquet")
            close_col = 'Close'
            vol_col = 'Volume'
        else:
            daily_path = os.path.join(base_dir, 'kr_daily', f"{symbol}.parquet")
            weekly_path = os.path.join(base_dir, 'kr_weekly', f"{symbol}.parquet")
            close_col = '종가'
            vol_col = '거래량'
        
        if not os.path.exists(daily_path) or not os.path.exists(weekly_path):
            print(f"{symbol} 데이터 없음 – 스킵")
            return None
        
        df_daily = pd.read_parquet(daily_path)
        df_weekly = pd.read_parquet(weekly_path)
        
        df_daily['RSI_D'] = talib.RSI(df_daily[close_col], timeperiod=14)
        
        macd, signal, hist = talib.MACD(df_daily[close_col], fastperiod=12, slowperiod=26, signalperiod=9)
        df_daily['MACD'] = macd
        df_daily['MACD_SIGNAL'] = signal
        
        df_daily['OBV'] = talib.OBV(df_daily[close_col], df_daily[vol_col])
        obv_signal = talib.SMA(df_daily['OBV'], timeperiod=9)
        df_daily['OBV_SIGNAL'] = obv_signal
        
        recent_d_rsi = df_daily['RSI_D'].tail(3).round(2).tolist()
        recent_macd = df_daily['MACD'].tail(3).round(4).tolist()
        recent_signal = df_daily['MACD_SIGNAL'].tail(3).round(4).tolist()
        recent_obv = df_daily['OBV'].tail(3).round(0).tolist()
        recent_obv_signal = df_daily['OBV_SIGNAL'].tail(3).round(0).tolist()
        
        if market == 'KR':
            market_cap = df_kr.loc[symbol, '시가총액'] if symbol in df_kr.index else 0
        else:
            ticker = yf.Ticker(symbol)
            market_cap = ticker.info.get('marketCap', 0) or 0
        
        df_daily['TradingValue'] = df_daily[close_col] * df_daily[vol_col]
        avg_20d = df_daily['TradingValue'].tail(20).mean()
        today_trading = df_daily['TradingValue'].iloc[-1]
        turnover = today_trading / market_cap if market_cap > 0 else 0
        
        return (symbol, market, json.dumps(recent_d_rsi),
                json.dumps(recent_macd), json.dumps(recent_signal),
                json.dumps(recent_obv), json.dumps(recent_obv_signal),
                float(market_cap), float(avg_20d), float(today_trading), float(turnover))
    except Exception as e:
        print(f"{symbol} 에러: {e} – 스킵")
        return None

if __name__ == '__main__':
    # 수정: DB 연결 재시도 + 자동 생성 로직
    def ensure_db_exists():
        if not os.path.exists(DB_PATH):
            con_temp = duckdb.connect(DB_PATH, read_only=False)
            con_temp.execute("""
                CREATE TABLE IF NOT EXISTS indicators (
                    symbol VARCHAR PRIMARY KEY,
                    market VARCHAR,
                    rsi_d TEXT,
                    macd_d TEXT,
                    signal_d TEXT,
                    obv_d TEXT,
                    signal_obv_d TEXT,
                    market_cap DOUBLE,
                    avg_trading_value_20d DOUBLE,
                    today_trading_value DOUBLE,
                    turnover DOUBLE
                )
            """)
            con_temp.close()
            print(f"DB 생성 완료: {DB_PATH}")
    
    con = None
    try:
        ensure_db_exists()  # 자동 생성
        con = duckdb.connect(DB_PATH)
    except duckdb.IOException:
        print("DB 파일 잠김 – 5초 대기 후 재시도")
        time.sleep(5)
        ensure_db_exists()
        con = duckdb.connect(DB_PATH)
    
    con.execute("DROP TABLE IF EXISTS indicators")
    con.execute("""
        CREATE TABLE IF NOT EXISTS indicators (
            symbol VARCHAR PRIMARY KEY,
            market VARCHAR,
            rsi_d TEXT,
            macd_d TEXT,
            signal_d TEXT,
            obv_d TEXT,
            signal_obv_d TEXT,
            market_cap DOUBLE,
            avg_trading_value_20d DOUBLE,
            today_trading_value DOUBLE,
            turnover DOUBLE
        )
    """)
    
    kr_tickers, df_kr = get_kr_tickers()
    us_symbols = get_us_symbols()
    print(f"KR 상위 500: {len(kr_tickers)}개 로드")
    print(f"US 상위 1000: {len(us_symbols)}개 로드")
    
    num_processes = min(cpu_count(), 8)
    print(f"멀티프로세싱 시작: {num_processes} 프로세스 사용")
    
    us_args = [(sym, 'US', None) for sym in us_symbols]
    with Pool(num_processes) as pool:
        us_results = pool.map(compute_indicators_wrapper, us_args)
    
    kr_args = [(ticker, 'KR', df_kr) for ticker in kr_tickers]
    with Pool(num_processes) as pool:
        kr_results = pool.map(compute_indicators_wrapper, kr_args)
    
    all_results = [r for r in us_results + kr_results if r is not None]
    print(f"유효 결과: {len(all_results)}개")
    
    for row in all_results:
        con.execute(
            "INSERT OR REPLACE INTO indicators VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            row
        )
    
    print("전체 완료! DB 확인: ", con.execute("SELECT COUNT(*) FROM indicators").fetchone()[0])
    con.close()