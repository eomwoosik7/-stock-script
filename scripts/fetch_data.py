import yfinance as yf
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
from pykrx import stock  # 시총 로드용만 유지
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count

def get_last_trading_date():
    today = datetime.now()
    while True:
        if today.weekday() < 5:  # 월~금 (0-4)
            return today.strftime('%Y%m%d')
        today -= timedelta(days=1)

def get_kr_tickers():
    """KR 티커 로드 (pykrx 시총용)"""
    today = get_last_trading_date()
    df_kr = stock.get_market_cap_by_ticker(today)
    kr_tickers = df_kr.sort_values('시가총액', ascending=False).head(500).index.tolist()
    print(f"KR 상위 500: {len(kr_tickers)}개 로드 (날짜: {today})")
    return kr_tickers

def get_us_symbols():
    """US 심볼 로드"""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    df_us = pd.read_html(str(table))[0]
    us_symbols = df_us['Symbol'].tolist()[:1000]
    print(f"US 상위 1000: {len(us_symbols)}개 로드")
    return us_symbols

def fetch_us_wrapper(args):
    """Pool용: (symbol, start_date) 튜플"""
    symbol, start_date = args
    return fetch_us_single(symbol, start_date)

def fetch_us_single(symbol, start_date):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date)
        if data.empty:
            print(f"데이터 다운로드 실패: {symbol}. 네트워크 확인하세요.")
            return
        print(f"{symbol} 컬럼 확인: {data.columns.tolist()}")
        daily_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'us_daily'))
        os.makedirs(daily_dir, exist_ok=True)
        data.to_parquet(os.path.join(daily_dir, f"{symbol}.parquet"))
        data_tz = data.tz_localize(None)
        weekly = data_tz.resample('W-FRI').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'})
        weekly_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'us_weekly'))
        os.makedirs(weekly_dir, exist_ok=True)
        weekly.to_parquet(os.path.join(weekly_dir, f"{symbol}.parquet"))
        print(f"{symbol} 주봉 저장 완료.")
    except Exception as e:
        print(f"{symbol} 에러: {e} – 스킵")

def fetch_kr_wrapper(args):
    """Pool용: (ticker, start_date) 튜플"""
    ticker, start_date = args
    return fetch_kr_single(ticker, start_date)

def fetch_kr_single(ticker, start_date):
    try:
        symbol_ks = f"{ticker}.KS"
        end_date = datetime.now().strftime('%Y-%m-%d')
        data = yf.download(symbol_ks, start=start_date, end=end_date, auto_adjust=False)  # 옵션 추가
        if data.empty:
            print(f"{ticker} 데이터 없음 – 스킵")
            return
        # MultiIndex 제거
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)  # 두 번째 레벨(티커) 드롭
        # 컬럼 매핑
        data['시가'] = data['Open']
        data['고가'] = data['High']
        data['저가'] = data['Low']
        data['종가'] = data['Close']
        data['거래량'] = data['Volume']
        data = data[['시가', '고가', '저가', '종가', '거래량']]
        
        print(f"{ticker} KR 컬럼: {data.columns.tolist()}")
        daily_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'kr_daily'))
        os.makedirs(daily_dir, exist_ok=True)
        data.to_parquet(os.path.join(daily_dir, f"{ticker}.parquet"))
        
        # 주봉
        weekly = data.resample('W-FRI').agg({'시가':'first', '고가':'max', '저가':'min', '종가':'last', '거래량':'sum'})
        weekly_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'kr_weekly'))
        os.makedirs(weekly_dir, exist_ok=True)
        weekly.to_parquet(os.path.join(weekly_dir, f"{ticker}.parquet"))
        print(f"{ticker} 저장 완료.")
    except Exception as e:
        print(f"{ticker} 에러: {e} – 스킵")

if __name__ == '__main__':  # 워커 보호
    start = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2년 데이터
    
    # 데이터 로드
    kr_tickers = get_kr_tickers()
    us_symbols = get_us_symbols()
    
    # 멀티프로세싱 설정
    num_processes = min(cpu_count(), 4)  # 4개로 제한 (네트워크 안정)
    print(f"멀티프로세싱 시작: {num_processes} 프로세스 사용")
    
    # US 병렬 다운로드
    us_args = [(symbol, start) for symbol in us_symbols]
    with Pool(num_processes) as pool:
        pool.map(fetch_us_wrapper, us_args)
    
    # KR 병렬 다운로드
    kr_args = [(ticker, start) for ticker in kr_tickers]
    with Pool(num_processes) as pool:
        pool.map(fetch_kr_wrapper, kr_args)
    
    print("전체 다운로드 완료!")