import duckdb
import pandas as pd
import os
import json
from pykrx import stock
import yfinance as yf
import sys
from datetime import datetime
import time  # 추가: sleep용

# 수정: 환경 변수로 DB_PATH 동적화
DATA_DIR = os.getenv('DATA_DIR', './data')
os.makedirs(DATA_DIR, exist_ok=True)
META_DIR = os.path.join(DATA_DIR, 'meta')
os.makedirs(META_DIR, exist_ok=True)
DB_PATH = os.path.join(META_DIR, 'universe.db')

con = None

# 수정: DB 연결 재시도 + 자동 생성
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

try:
    ensure_db_exists()
    con = duckdb.connect(DB_PATH)
except duckdb.IOException:
    print("DB 파일 잠김 – 5초 대기 후 재시도")
    time.sleep(5)
    ensure_db_exists()
    con = duckdb.connect(DB_PATH)

def run_screener(top_n=50, use_us=True, use_kr=True):
    market_filter = "market = 'US'" if use_us and not use_kr else "market = 'KR'" if use_kr and not use_us else "market IN ('US', 'KR')"
    query = f"""
    WITH parsed AS (
        SELECT symbol, market,
            rsi_d, obv_d, signal_obv_d, market_cap, avg_trading_value_20d, today_trading_value, turnover,
            CAST(json_extract_string(rsi_d, '$[0]') AS DOUBLE) AS rsi_d_2ago,
            CAST(json_extract_string(rsi_d, '$[1]') AS DOUBLE) AS rsi_d_1ago,
            CAST(json_extract_string(rsi_d, '$[2]') AS DOUBLE) AS rsi_d_latest,
            CAST(json_extract_string(obv_d, '$[1]') AS DOUBLE) AS obv_1ago,
            CAST(json_extract_string(obv_d, '$[0]') AS DOUBLE) AS obv_latest,
            CAST(json_extract_string(signal_obv_d, '$[1]') AS DOUBLE) AS signal_obv_1ago,
            CAST(json_extract_string(signal_obv_d, '$[0]') AS DOUBLE) AS signal_obv_latest
        FROM indicators
    )
    SELECT symbol, market,
        rsi_d AS rsi_d_array,
        obv_d AS obv_array,
        signal_obv_d AS signal_obv_array,
        market_cap, avg_trading_value_20d, today_trading_value, turnover,
        rsi_d_2ago, rsi_d_1ago, rsi_d_latest,
        obv_latest, signal_obv_latest,
        (obv_latest > signal_obv_latest AND obv_1ago <= signal_obv_1ago) AS obv_bullish_cross,
        (rsi_d_2ago > rsi_d_1ago AND rsi_d_1ago > rsi_d_latest) AS rsi_d_3down
    FROM parsed
    WHERE {market_filter}
      AND (obv_latest > signal_obv_latest AND obv_1ago <= signal_obv_1ago)
      AND (rsi_d_2ago > rsi_d_1ago AND rsi_d_1ago > rsi_d_latest)
      AND market_cap >= CASE WHEN market = 'US' THEN 2000000000.0 ELSE 200000000000.0 END
      AND avg_trading_value_20d >= CASE WHEN market = 'US' THEN 30000000.0 ELSE 5000000000.0 END
      AND turnover >= 0.005
    ORDER BY rsi_d_latest ASC
    """
    results = con.execute(query).fetchdf()
    
    def get_name(row):
        try:
            if row['market'] == 'KR':
                return stock.get_market_ticker_name(row['symbol'])
            else:
                ticker = yf.Ticker(row['symbol'])
                return ticker.info.get('longName', 'N/A')
        except:
            return 'N/A'
    
    results['name'] = results.apply(get_name, axis=1)
    numeric_cols = results.select_dtypes(include=['float64']).columns
    for col in numeric_cols:
        results[col] = results[col].round(2)
    
    results = results.head(top_n)
    
    # 수정: results_path 환경 변수로
    results_path = os.path.join(META_DIR, 'screener_results.parquet')
    results.to_parquet(results_path)
    print(f"스크리너 완료: {len(results)}개 후보")
    print(results.round(2))
    return results

if __name__ == "__main__":
    use_us = sys.argv[1] == 'True' if len(sys.argv) > 1 else True
    use_kr = sys.argv[2] == 'True' if len(sys.argv) > 2 else True
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    run_screener(top_n, use_us, use_kr)
    if con:
        con.close()  # 추가: 연결 종료