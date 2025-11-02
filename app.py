import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
import subprocess
import talib
from pykrx import stock
import yfinance as yf
from datetime import datetime
import numpy as np  # np.where용

st.set_page_config(page_title="Trading Copilot", layout="wide")

# 수정: 환경 변수로 DATA_DIR 동적화 (Cloud Run용)
DATA_DIR = os.getenv('DATA_DIR', '/tmp/data')
os.makedirs(DATA_DIR, exist_ok=True)
META_DIR = os.path.join(DATA_DIR, 'meta')
os.makedirs(META_DIR, exist_ok=True)
DB_PATH = os.path.join(META_DIR, 'universe.db')

@st.cache_data
def load_data():
    con = duckdb.connect(DB_PATH, read_only=True)  # 추가: 읽기 전용
    df_ind = con.execute("SELECT * FROM indicators").fetchdf()
    con.close()
    return df_ind

# 수정: 캐시 제거 - 매번 새 연결 생성 (잠김 방지)
def get_db_connection():
    if not os.path.exists(DB_PATH):
        # 파일 없으면 생성 (read_only=False)
        con = duckdb.connect(DB_PATH, read_only=False)
        # 테이블 생성 (기존 스키마 맞춰)
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
        con.close()  # 임시 연결 종료
        print(f"DB 생성 완료: {DB_PATH}")  # 디버그용
    
    # 이제 read_only=True로 연결
    return duckdb.connect(DB_PATH, read_only=True)

def run_screener_query(con, filter_condition="all", use_us=True, use_kr=True, top_n=None):
    # 추가: con close 체크 후 재연결
    try:
        con.execute("SELECT 1").fetchone()  # 테스트 쿼리
    except:
        con = get_db_connection()  # 재연결
        st.session_state.con = con  # 세션 업데이트
    
    market_filter = "market = 'US'" if use_us and not use_kr else "market = 'KR'" if use_kr and not use_us else "market IN ('US', 'KR')"
    
    if filter_condition == "obv":
        condition = "(obv_latest > signal_obv_latest AND obv_1ago <= signal_obv_1ago)"
    elif filter_condition == "rsi":
        condition = "(rsi_d_2ago > rsi_d_1ago AND rsi_d_1ago > rsi_d_latest)"
    else:  # all
        condition = "(obv_latest > signal_obv_latest AND obv_1ago <= signal_obv_1ago) AND (rsi_d_2ago > rsi_d_1ago AND rsi_d_1ago > rsi_d_latest)"
    
    liquidity = """
    AND market_cap >= CASE WHEN market = 'US' THEN 2000000000.0 ELSE 200000000000.0 END
    AND avg_trading_value_20d >= CASE WHEN market = 'US' THEN 30000000.0 ELSE 5000000000.0 END
    AND turnover >= 0.005
    """
    
    query = f"""
    WITH parsed AS (
        SELECT symbol, market,
            rsi_d, macd_d, signal_d, obv_d, signal_obv_d, market_cap, avg_trading_value_20d, today_trading_value, turnover,
            CAST(json_extract(rsi_d, '$[0]') AS DOUBLE) AS rsi_d_2ago,  -- [0]=2ago
            CAST(json_extract(rsi_d, '$[1]') AS DOUBLE) AS rsi_d_1ago,  -- [1]=1ago
            CAST(json_extract(rsi_d, '$[2]') AS DOUBLE) AS rsi_d_latest, -- [2]=latest
            CAST(json_extract(macd_d, '$[2]') AS DOUBLE) AS macd_latest,
            CAST(json_extract(signal_d, '$[2]') AS DOUBLE) AS signal_latest,
            CAST(json_extract(obv_d, '$[1]') AS DOUBLE) AS obv_1ago,
            CAST(json_extract(obv_d, '$[0]') AS DOUBLE) AS obv_latest,
            CAST(json_extract(signal_obv_d, '$[1]') AS DOUBLE) AS signal_obv_1ago,
            CAST(json_extract(signal_obv_d, '$[0]') AS DOUBLE) AS signal_obv_latest
        FROM indicators
    )
    SELECT symbol, market,
        rsi_d AS rsi_d_array,
        macd_d AS macd_array,
        signal_d AS signal_array,
        obv_d AS obv_array,
        signal_obv_d AS signal_obv_array,
        market_cap, avg_trading_value_20d, today_trading_value, turnover,
        rsi_d_2ago, rsi_d_1ago, rsi_d_latest,
        macd_latest, signal_latest,
        obv_latest, signal_obv_latest,
        (obv_latest > signal_obv_latest AND obv_1ago <= signal_obv_1ago) AS obv_bullish_cross,
        (rsi_d_2ago > rsi_d_1ago AND rsi_d_1ago > rsi_d_latest) AS rsi_d_3down,
        CASE WHEN avg_trading_value_20d >= CASE WHEN market = 'US' THEN 30000000.0 ELSE 5000000000.0 END THEN 1 ELSE 0 END AS avg_vol_ok,
        CASE WHEN turnover >= 0.005 THEN 1 ELSE 0 END AS turnover_ok
    FROM parsed
    WHERE {market_filter}
      AND {condition}
      {liquidity}
    ORDER BY rsi_d_latest ASC
    """
    df = con.execute(query).fetchdf()
    if top_n:
        df = df.head(top_n)
    return df

def format_dataframe(df, market_type):
    def market_cap_fmt(x):
        if pd.isna(x) or x == 0: return 'N/A'
        x = float(x)
        prefix = 'KRW ' if market_type == 'KR' else 'USD '
        if market_type == 'KR': return f"{prefix}{x / 1e8:,.0f}억원"
        else: return f"{prefix}{x / 1e9:,.1f}B"
    
    def trading_value_fmt(x):
        if pd.isna(x) or x == 0: return 'N/A'
        x = float(x)
        if market_type == 'KR': return f"{x / 1e8:,.0f}억원"
        else: return f"{x / 1e6:,.0f}M USD"
    
    def turnover_fmt(x):
        if pd.isna(x) or x == 0: return 'N/A'
        x = float(x) * 100
        return f"{x:.2f}%"
    
    def rsi_fmt(x):
        if pd.isna(x): return 'N/A'
        if isinstance(x, str):
            try:
                vals = json.loads(x)
                if isinstance(vals, list):
                    return ', '.join(f"{v:.2f}" for v in vals)  # RSI: 3일치 전체
                else:
                    return f"{float(vals):.2f}"
            except:
                return str(x)
        else:
            return f"{float(x):.2f}"
    
    def macd_fmt(x):
        if pd.isna(x): return 'N/A'
        if isinstance(x, str):
            try:
                vals = json.loads(x)
                if isinstance(vals, list):
                    return f"{vals[2]:.2f}"  # MACD: 최근 1일치
                else:
                    return f"{float(vals):.4f}"
            except:
                return str(x)
        else:
            return f"{float(x):.4f}"
    
    def obv_fmt(x):
        if pd.isna(x): return 'N/A'
        if isinstance(x, str):
            try:
                vals = json.loads(x)
                if isinstance(vals, list):
                    return f"{int(vals[2]):,}"  # OBV: 최근 1일치
                else:
                    return f"{int(vals):,}"
            except:
                return str(x)
        else:
            return f"{int(x):,}"
    
    def bool_fmt(x):
        return '✅' if x else '❌'  # 추가: bool 체크 표시
    
    # 적용
    if '시가총액' in df.columns:
        df['시가총액'] = df['시가총액'].apply(market_cap_fmt)
    if '20일평균거래대금' in df.columns:
        df['20일평균거래대금'] = df['20일평균거래대금'].apply(trading_value_fmt)
    if '오늘거래대금' in df.columns:
        df['오늘거래대금'] = df['오늘거래대금'].apply(trading_value_fmt)
    if '회전율' in df.columns:
        df['회전율'] = df['회전율'].apply(turnover_fmt)
    if 'RSI_3일' in df.columns:
        df['RSI_3일'] = df['RSI_3일'].apply(rsi_fmt)
    if 'MACD' in df.columns:
        df['MACD'] = df['MACD'].apply(macd_fmt)
    if 'MACD_SIGNAL' in df.columns:
        df['MACD_SIGNAL'] = df['MACD_SIGNAL'].apply(macd_fmt)
    if 'OBV' in df.columns:
        df['OBV'] = df['OBV'].apply(obv_fmt)
    if 'OBV_SIGNAL' in df.columns:
        df['OBV_SIGNAL'] = df['OBV_SIGNAL'].apply(obv_fmt)
    if 'OBV_상승' in df.columns:
        df['OBV_상승'] = df['OBV_상승'].apply(bool_fmt)
    if 'RSI_3하락' in df.columns:
        df['RSI_3하락'] = df['RSI_3하락'].apply(bool_fmt)
    if '거래대금_최소' in df.columns:
        df['거래대금_최소'] = df['거래대금_최소'].apply(bool_fmt)
    if '회전율_최소' in df.columns:
        df['회전율_최소'] = df['회전율_최소'].apply(bool_fmt)
    
    return df

def show_graphs(symbol, market):
    # 수정: base_dir 환경 변수로 동적화
    base_dir = DATA_DIR
    daily_path = os.path.join(base_dir, ('us_daily' if market == 'US' else 'kr_daily'), f"{symbol}.parquet")
    if os.path.exists(daily_path):
        df_chart = pd.read_parquet(daily_path)
        close_col = 'Close' if market == 'US' else '종가'
        vol_col = 'Volume' if market == 'US' else '거래량'
        
        if close_col in df_chart.columns:
            df_chart[close_col] = df_chart[close_col].round(2)
        
        fig_price = px.line(df_chart, x=df_chart.index, y=close_col, title=f"{symbol} 종가")
        fig_price.update_layout(height=400)
        # 수정: 키에 탭 이름 추가 (중복 방지)
        st.plotly_chart(fig_price, width='stretch', key=f"{st.session_state.current_tab}_{symbol}_price_chart")
        
        macd, signal, hist = talib.MACD(df_chart[close_col], fastperiod=12, slowperiod=26, signalperiod=9)
        df_macd = pd.DataFrame({'Date': df_chart.index, 'MACD': macd, 'Signal': signal, 'Hist': hist}).dropna()
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df_macd['Date'], y=df_macd['MACD'], name='MACD', line=dict(color='blue')))
        fig_macd.add_trace(go.Scatter(x=df_macd['Date'], y=df_macd['Signal'], name='Signal', line=dict(color='red')))
        fig_macd.add_trace(go.Bar(x=df_macd['Date'], y=df_macd['Hist'], name='Histogram', marker_color='gray', opacity=0.6))
        fig_macd.update_layout(title=f"{symbol} MACD 분석", height=400)
        # 수정: 키에 탭 이름 추가
        st.plotly_chart(fig_macd, width='stretch', key=f"{st.session_state.current_tab}_{symbol}_macd_chart")
        
        df_chart['OBV'] = talib.OBV(df_chart[close_col], df_chart[vol_col])
        obv_signal = talib.SMA(df_chart['OBV'], timeperiod=9)
        df_chart['OBV_SIGNAL'] = obv_signal
        df_obv = df_chart[['OBV', 'OBV_SIGNAL']].dropna()
        fig_obv = go.Figure()
        fig_obv.add_trace(go.Scatter(x=df_obv.index, y=df_obv['OBV'], name='OBV', line=dict(color='green')))
        fig_obv.add_trace(go.Scatter(x=df_obv.index, y=df_obv['OBV_SIGNAL'], name='OBV Signal', line=dict(color='orange')))
        fig_obv.update_layout(title=f"{symbol} OBV 분석", height=400)
        # 수정: 키에 탭 이름 추가
        st.plotly_chart(fig_obv, width='stretch', key=f"{st.session_state.current_tab}_{symbol}_obv_chart")

def get_filtered_symbols(df, search_term):
    symbol_col = 'symbol' if 'symbol' in df.columns else '종목코드'
    if search_term:
        search_upper = search_term.upper()
        mask = df[symbol_col].str.contains(search_upper, na=False)
        if 'name' in df.columns or '회사명' in df.columns:
            name_col = 'name' if 'name' in df.columns else '회사명'
            mask = mask | df[name_col].str.contains(search_upper, na=False)
        return df[mask][symbol_col].tolist()
    return df[symbol_col].tolist()

@st.cache_data
def add_names_cached(symbols, markets):
    name_dict = {}
    for sym, mkt in zip(symbols, markets):
        try:
            if mkt == 'KR':
                name_dict[sym] = stock.get_market_ticker_name(sym)
            else:
                ticker = yf.Ticker(sym)
                name_dict[sym] = ticker.info.get('longName', 'N/A')
        except:
            name_dict[sym] = 'N/A'
    return name_dict

def add_names(df):
    df = df.copy()
    if 'symbol' not in df.columns:
        return df
    symbols = df['symbol'].tolist()
    markets = df['market'].tolist()
    name_dict = add_names_cached(symbols, markets)
    df.loc[:, 'name'] = df['symbol'].map(name_dict)
    return df

def prepare_tab_df(df, is_total=False):
    df = df.copy()
    
    # 컬럼 확인: 스크리너 vs Total
    if 'rsi_d_array' in df.columns:
        rsi_col = 'rsi_d_array'
        macd_col = 'macd_array'
        signal_col = 'signal_array'
        obv_col = 'obv_array'
        signal_obv_col = 'signal_obv_array'
    else:
        rsi_col = 'rsi_d'
        macd_col = 'macd_d'
        signal_col = 'signal_d'
        obv_col = 'obv_d'
        signal_obv_col = 'signal_obv_d'
    
    # JSON 파싱 (최근 1일치만: [0] 인덱스)
    df.loc[:, 'rsi_d_latest'] = df[rsi_col].apply(lambda x: json.loads(x)[2] if pd.notna(x) and len(json.loads(x)) > 2 else None)  # [2]=latest
    df.loc[:, 'rsi_d_1ago'] = df[rsi_col].apply(lambda x: json.loads(x)[1] if pd.notna(x) and len(json.loads(x)) > 1 else None)    # [1]=1ago
    df.loc[:, 'rsi_d_2ago'] = df[rsi_col].apply(lambda x: json.loads(x)[0] if pd.notna(x) and len(json.loads(x)) > 0 else None)    # [0]=2ago
    df.loc[:, 'macd_latest'] = df[macd_col].apply(lambda x: json.loads(x)[2] if pd.notna(x) and len(json.loads(x)) > 2 else None)
    df.loc[:, 'signal_latest'] = df[signal_col].apply(lambda x: json.loads(x)[2] if pd.notna(x) and len(json.loads(x)) > 2 else None)
    df.loc[:, 'obv_latest'] = df[obv_col].apply(lambda x: json.loads(x)[0] if pd.notna(x) and len(json.loads(x)) > 0 else None)
    df.loc[:, 'obv_1ago'] = df[obv_col].apply(lambda x: json.loads(x)[1] if pd.notna(x) and len(json.loads(x)) > 1 else None)
    df.loc[:, 'signal_obv_latest'] = df[signal_obv_col].apply(lambda x: json.loads(x)[0] if pd.notna(x) and len(json.loads(x)) > 0 else None)
    df.loc[:, 'signal_obv_1ago'] = df[signal_obv_col].apply(lambda x: json.loads(x)[1] if pd.notna(x) and len(json.loads(x)) > 1 else None)
    
    # 조건 컬럼 추가 (영어 이름으로: 중복 방지)
    df.loc[:, 'obv_bullish'] = (df['obv_latest'] > df['signal_obv_latest']) & (df['obv_1ago'] <= df['signal_obv_1ago'])
    df.loc[:, 'rsi_3down'] = (df['rsi_d_2ago'] > df['rsi_d_1ago']) & (df['rsi_d_1ago'] > df['rsi_d_latest'])
    
    # 유동성 조건 추가
    threshold = np.where(df['market'] == 'US', 30000000.0, 5000000000.0)
    df.loc[:, '거래대금_최소'] = df['avg_trading_value_20d'] >= threshold
    df.loc[:, '회전율_최소'] = df['turnover'] >= 0.005
    
    if not is_total:
        # 컬럼 rename (조건 컬럼도 한국어로, MACD 추가)
        col_map = {
            'symbol': '종목코드',
            'market': '시장',
            'name': '회사명',
            'rsi_d_array': 'RSI_3일',
            'macd_array': 'MACD',
            'signal_array': 'MACD_SIGNAL',
            'obv_array': 'OBV',
            'signal_obv_array': 'OBV_SIGNAL',
            'market_cap': '시가총액',
            'avg_trading_value_20d': '20일평균거래대금',
            'today_trading_value': '오늘거래대금',
            'turnover': '회전율',
            'obv_bullish': 'OBV_상승',
            'rsi_3down': 'RSI_3하락'
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    
    return df

# 메인 앱: con을 세션 상태로 관리 (에러 방지)
if 'con' not in st.session_state:
    st.session_state.con = get_db_connection()
con = st.session_state.con

df_ind = load_data()

st.sidebar.header("필터 & 실행 (Home용)")
use_kr = st.sidebar.checkbox("KR", True)
use_us = st.sidebar.checkbox("US", True)
top_n = st.sidebar.slider("Top N", 5, 50, 50)

if st.sidebar.button("데이터 업데이트"):
    if 'con' in st.session_state and st.session_state.con:
        st.session_state.con.close()  # 임시 close
        del st.session_state.con
    import time
    subprocess.run(["python", "scripts/batch.py", str(use_kr), str(use_us), str(top_n)])
    time.sleep(3)  # 대기
    st.cache_data.clear()
    st.session_state.con = get_db_connection()  # 재연결
    st.rerun()

tab1, tab2, tab3, tab4 = st.tabs(["Home (전체 스크리너)", "OBV 상승", "RSI 하락 지속", "Total"])

with tab1:
    st.session_state.current_tab = "Home"
    st.header("Home (OBV 상승 + RSI 하락 + 유동성)")
    df_full = run_screener_query(con, "all", use_us, use_kr, top_n=None)
    df = df_full.head(top_n)  # Home만 TOP N 적용
    df = add_names(df)
    df = prepare_tab_df(df)
    
    if not df_full.empty:
        df_kr_temp = df_full[df_full['market'] == 'KR']
        df_us_temp = df_full[df_full['market'] == 'US']
        total_candidates = len(df_kr_temp) + len(df_us_temp)
        st.metric("후보 수", total_candidates)
        
        col_map = {'symbol': '종목코드', 'market': '시장', 'name': '회사명', 'rsi_d_array': 'RSI_3일', 
                   'macd_array': 'MACD', 'signal_array': 'MACD_SIGNAL',
                   'obv_array': 'OBV', 'signal_obv_array': 'OBV_SIGNAL', 'market_cap': '시가총액', 
                   'avg_trading_value_20d': '20일평균거래대금', 'today_trading_value': '오늘거래대금', 'turnover': '회전율',
                   'obv_bullish': 'OBV_상승', 'rsi_3down': 'RSI_3하락'}
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        
        df_kr_results = df[df['시장'] == 'KR'] if '시장' in df.columns else pd.DataFrame()
        df_us_results = df[df['시장'] == 'US'] if '시장' in df.columns else pd.DataFrame()
        
        if not df_kr_results.empty:
            cols_kr = ['종목코드', '회사명', '시장', 'RSI_3일', 'MACD', 'MACD_SIGNAL', 'OBV', 'OBV_SIGNAL', '시가총액', '20일평균거래대금', '오늘거래대금', '회전율', 'OBV_상승', 'RSI_3하락', '거래대금_최소', '회전율_최소']
            df_kr_results = df_kr_results[[col for col in cols_kr if col in df_kr_results.columns]]
            df_kr_results = df_kr_results.sort_values('시가총액', ascending=False)
            df_kr_results = format_dataframe(df_kr_results, 'KR')
            # 수정: 가운데 정렬
            df_kr_results = df_kr_results.style.set_properties(**{'text-align': 'center'})
            st.subheader("국내 (KR) - 시총: KRW 억원")
            st.dataframe(df_kr_results)
        if not df_us_results.empty:
            cols_us = ['종목코드', '회사명', '시장', 'RSI_3일', 'MACD', 'MACD_SIGNAL', 'OBV', 'OBV_SIGNAL', '시가총액', '20일평균거래대금', '오늘거래대금', '회전율', 'OBV_상승', 'RSI_3하락', '거래대금_최소', '회전율_최소']
            df_us_results = df_us_results[[col for col in cols_us if col in df_us_results.columns]]
            df_us_results = df_us_results.sort_values('시가총액', ascending=False)
            df_us_results = format_dataframe(df_us_results, 'US')
            # 수정: 가운데 정렬
            df_us_results = df_us_results.style.set_properties(**{'text-align': 'center'})
            st.subheader("해외 (US) - 시총: USD B")
            st.dataframe(df_us_results)
        
        search_term = st.text_input("종목 검색 (Home)", placeholder="코드/회사명 입력", key="search_home")
        filtered_symbols = get_filtered_symbols(df, search_term)
        if filtered_symbols:
            selected_symbol = st.selectbox("종목 선택 (Home)", filtered_symbols, key="select_home")
            if selected_symbol:
                market = df[df['종목코드'] == selected_symbol]['시장'].iloc[0] if '시장' in df.columns else df[df['symbol'] == selected_symbol]['market'].iloc[0] if 'market' in df.columns else 'US'
                show_graphs(selected_symbol, market)
    else:
        st.info("후보 없음")

with tab2:
    st.session_state.current_tab = "OBV 상승"
    st.header("OBV 상승 (조건 1 + 유동성)")
    df_obv_full = run_screener_query(con, "obv", use_us, use_kr, top_n=None)  # TOP N 없음
    df_obv = df_obv_full  # 전체
    df_obv = add_names(df_obv)
    df_obv = prepare_tab_df(df_obv)
    
    if not df_obv_full.empty:
        df_kr_temp = df_obv_full[df_obv_full['market'] == 'KR']
        df_us_temp = df_obv_full[df_obv_full['market'] == 'US']
        total_candidates = len(df_kr_temp) + len(df_us_temp)
        st.metric("후보 수", total_candidates)
        
        col_map = {'symbol': '종목코드', 'market': '시장', 'name': '회사명', 'rsi_d_array': 'RSI_3일', 
                   'macd_array': 'MACD', 'signal_array': 'MACD_SIGNAL',
                   'obv_array': 'OBV', 'signal_obv_array': 'OBV_SIGNAL', 'market_cap': '시가총액', 
                   'avg_trading_value_20d': '20일평균거래대금', 'today_trading_value': '오늘거래대금', 'turnover': '회전율',
                   'obv_bullish': 'OBV_상승', 'rsi_3down': 'RSI_3하락'}
        df_obv = df_obv.rename(columns={k: v for k, v in col_map.items() if k in df_obv.columns})
        
        df_kr_results = df_obv[df_obv['시장'] == 'KR'] if '시장' in df_obv.columns else pd.DataFrame()
        df_us_results = df_obv[df_obv['시장'] == 'US'] if '시장' in df_obv.columns else pd.DataFrame()
        
        if not df_kr_results.empty:
            cols_kr = ['종목코드', '회사명', '시장', 'RSI_3일', 'MACD', 'MACD_SIGNAL', 'OBV', 'OBV_SIGNAL', '시가총액', '20일평균거래대금', '오늘거래대금', '회전율', 'OBV_상승', 'RSI_3하락', '거래대금_최소', '회전율_최소']
            df_kr_results = df_kr_results[[col for col in cols_kr if col in df_kr_results.columns]]
            df_kr_results = df_kr_results.sort_values('시가총액', ascending=False)
            df_kr_results = format_dataframe(df_kr_results, 'KR')
            # 수정: 가운데 정렬
            df_kr_results = df_kr_results.style.set_properties(**{'text-align': 'center'})
            st.subheader("국내 (KR) - 시총: KRW 억원")
            st.dataframe(df_kr_results)
        if not df_us_results.empty:
            cols_us = ['종목코드', '회사명', '시장', 'RSI_3일', 'MACD', 'MACD_SIGNAL', 'OBV', 'OBV_SIGNAL', '시가총액', '20일평균거래대금', '오늘거래대금', '회전율', 'OBV_상승', 'RSI_3하락', '거래대금_최소', '회전율_최소']
            df_us_results = df_us_results[[col for col in cols_us if col in df_us_results.columns]]
            df_us_results = df_us_results.sort_values('시가총액', ascending=False)
            df_us_results = format_dataframe(df_us_results, 'US')
            # 수정: 가운데 정렬
            df_us_results = df_us_results.style.set_properties(**{'text-align': 'center'})
            st.subheader("해외 (US) - 시총: USD B")
            st.dataframe(df_us_results)
        
        search_term = st.text_input("종목 검색 (OBV)", placeholder="코드/회사명 입력", key="search_obv")
        filtered_symbols = get_filtered_symbols(df_obv, search_term)
        if filtered_symbols:
            selected_symbol = st.selectbox("종목 선택 (OBV)", filtered_symbols, key="select_obv")
            if selected_symbol:
                market = df_obv[df_obv['종목코드'] == selected_symbol]['시장'].iloc[0] if '시장' in df_obv.columns else 'US'
                show_graphs(selected_symbol, market)
    else:
        st.info("OBV 후보 없음")

with tab3:
    st.session_state.current_tab = "RSI 하락 지속"
    st.header("RSI 하락 지속 (조건 2 + 유동성)")
    df_rsi_full = run_screener_query(con, "rsi", use_us, use_kr, top_n=None)  # TOP N 없음
    df_rsi = df_rsi_full  # 전체
    df_rsi = add_names(df_rsi)
    df_rsi = prepare_tab_df(df_rsi)
    
    if not df_rsi_full.empty:
        df_kr_temp = df_rsi_full[df_rsi_full['market'] == 'KR']
        df_us_temp = df_rsi_full[df_rsi_full['market'] == 'US']
        total_candidates = len(df_kr_temp) + len(df_us_temp)
        st.metric("후보 수", total_candidates)
        
        col_map = {'symbol': '종목코드', 'market': '시장', 'name': '회사명', 'rsi_d_array': 'RSI_3일', 
                   'macd_array': 'MACD', 'signal_array': 'MACD_SIGNAL',
                   'obv_array': 'OBV', 'signal_obv_array': 'OBV_SIGNAL', 'market_cap': '시가총액', 
                   'avg_trading_value_20d': '20일평균거래대금', 'today_trading_value': '오늘거래대금', 'turnover': '회전율',
                   'obv_bullish': 'OBV_상승', 'rsi_3down': 'RSI_3하락'}
        df_rsi = df_rsi.rename(columns={k: v for k, v in col_map.items() if k in df_rsi.columns})
        
        df_kr_results = df_rsi[df_rsi['시장'] == 'KR'] if '시장' in df_rsi.columns else pd.DataFrame()
        df_us_results = df_rsi[df_rsi['시장'] == 'US'] if '시장' in df_rsi.columns else pd.DataFrame()
        
        if not df_kr_results.empty:
            cols_kr = ['종목코드', '회사명', '시장', 'RSI_3일', 'MACD', 'MACD_SIGNAL', 'OBV', 'OBV_SIGNAL', '시가총액', '20일평균거래대금', '오늘거래대금', '회전율', 'OBV_상승', 'RSI_3하락', '거래대금_최소', '회전율_최소']
            df_kr_results = df_kr_results[[col for col in cols_kr if col in df_kr_results.columns]]
            df_kr_results = df_kr_results.sort_values('시가총액', ascending=False)
            df_kr_results = format_dataframe(df_kr_results, 'KR')
            # 수정: 가운데 정렬
            df_kr_results = df_kr_results.style.set_properties(**{'text-align': 'center'})
            st.subheader("국내 (KR) - 시총: KRW 억원")
            st.dataframe(df_kr_results)
        if not df_us_results.empty:
            cols_us = ['종목코드', '회사명', '시장', 'RSI_3일', 'MACD', 'MACD_SIGNAL', 'OBV', 'OBV_SIGNAL', '시가총액', '20일평균거래대금', '오늘거래대금', '회전율', 'OBV_상승', 'RSI_3하락', '거래대금_최소', '회전율_최소']
            df_us_results = df_us_results[[col for col in cols_us if col in df_us_results.columns]]
            df_us_results = df_us_results.sort_values('시가총액', ascending=False)
            df_us_results = format_dataframe(df_us_results, 'US')
            # 수정: 가운데 정렬
            df_us_results = df_us_results.style.set_properties(**{'text-align': 'center'})
            st.subheader("해외 (US) - 시총: USD B")
            st.dataframe(df_us_results)
        
        search_term = st.text_input("종목 검색 (RSI)", placeholder="코드/회사명 입력", key="search_rsi")
        filtered_symbols = get_filtered_symbols(df_rsi, search_term)
        if filtered_symbols:
            selected_symbol = st.selectbox("종목 선택 (RSI)", filtered_symbols, key="select_rsi")
            if selected_symbol:
                market = df_rsi[df_rsi['종목코드'] == selected_symbol]['시장'].iloc[0] if '시장' in df_rsi.columns else 'US'
                show_graphs(selected_symbol, market)
    else:
        st.info("RSI 후보 없음")

with tab4:
    st.session_state.current_tab = "Total"
    st.header("Total (전체 종목 목록)")
    if not df_ind.empty:
        # df_ind = add_names(df_ind)  # 제거: 회사명 로드 안 함
        df_ind = prepare_tab_df(df_ind, is_total=True)
        
        col_map_total = {'symbol': '종목코드', 'market': '시장',  # 'name': '회사명' 제거
                         'rsi_d': 'RSI_3일', 'macd_d': 'MACD', 
                         'signal_d': 'MACD_SIGNAL', 'obv_d': 'OBV', 'signal_obv_d': 'OBV_SIGNAL', 
                         'market_cap': '시가총액', 'avg_trading_value_20d': '20일평균거래대금', 
                         'today_trading_value': '오늘거래대금', 'turnover': '회전율'}
        df_ind_renamed = df_ind.rename(columns={k: v for k, v in col_map_total.items() if k in df_ind.columns})
        df_ind_renamed = df_ind_renamed.sort_values('시가총액', ascending=False).reset_index(drop=True)
        
        df_kr_ind = df_ind_renamed[df_ind_renamed['시장'] == 'KR'] if '시장' in df_ind_renamed.columns else pd.DataFrame()
        df_us_ind = df_ind_renamed[df_ind_renamed['시장'] == 'US'] if '시장' in df_ind_renamed.columns else pd.DataFrame()
        
        if not df_kr_ind.empty:
            cols_kr_total = ['종목코드', '시장', 'RSI_3일', 'MACD', 'MACD_SIGNAL', 'OBV', 'OBV_SIGNAL', '시가총액', '20일평균거래대금', '오늘거래대금', '회전율']  # '회사명' 제거
            df_kr_ind = df_kr_ind[[col for col in cols_kr_total if col in df_kr_ind.columns]]
            df_kr_ind = format_dataframe(df_kr_ind, 'KR')
            # 수정: 가운데 정렬
            df_kr_ind = df_kr_ind.style.set_properties(**{'text-align': 'center'})
            st.subheader("국내 (KR) - 시총: KRW 억원")
            st.dataframe(df_kr_ind)
        if not df_us_ind.empty:
            cols_us_total = ['종목코드', '시장', 'RSI_3일', 'MACD', 'MACD_SIGNAL', 'OBV', 'OBV_SIGNAL', '시가총액', '20일평균거래대금', '오늘거래대금', '회전율']  # '회사명' 제거
            df_us_ind = df_us_ind[[col for col in cols_us_total if col in df_us_ind.columns]]
            df_us_ind = format_dataframe(df_us_ind, 'US')
            # 수정: 가운데 정렬
            df_us_ind = df_us_ind.style.set_properties(**{'text-align': 'center'})
            st.subheader("해외 (US) - 시총: USD B")
            st.dataframe(df_us_ind)
        
        search_term = st.text_input("종목 검색 (Total)", placeholder="코드 입력", key="search_total")
        filtered_symbols = get_filtered_symbols(df_ind_renamed, search_term)
        if filtered_symbols:
            selected_symbol = st.selectbox("종목 선택 (Total)", filtered_symbols, key="select_total")
            if selected_symbol:
                market = df_ind[df_ind['symbol'] == selected_symbol]['market'].iloc[0] if 'market' in df_ind.columns else 'US'
                show_graphs(selected_symbol, market)
        else:
            st.info("검색 결과 없음")
    else:
        st.info("데이터 없음 – 배치 실행하세요.")

# 앱 끝에서 con close (세션 상태로 관리되므로 안전)
if hasattr(st.session_state, 'con') and st.session_state.con:
    st.session_state.con.close()

# 수정: LOG_DIR 환경 변수로 동적화
LOG_DIR = os.getenv('LOG_DIR', '/tmp/logs')
os.makedirs(LOG_DIR, exist_ok=True)
log_time_file = os.path.join(LOG_DIR, 'batch_time.txt')
if os.path.exists(log_time_file):
    with open(log_time_file, "r") as f:
        last_time = f.read().strip()
    st.sidebar.info(f"마지막 갱신: {last_time}")
else:
    st.sidebar.info("로그 없음 – 배치 실행하세요.")