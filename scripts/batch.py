import schedule
import time
import subprocess
from datetime import datetime
import requests
import os
import pandas as pd
import sys

def run_batch(use_kr=True, use_us=True, top_n=20):
    print(f"배치 시작: {datetime.now()}")
    
    # 수정: SCRIPT_DIR 환경 변수로 (배포 시 scripts 폴더 경로)
    SCRIPT_DIR = os.getenv('SCRIPT_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
    
    subprocess.run(["python", os.path.join(SCRIPT_DIR, "fetch_data.py")])
    subprocess.run(["python", os.path.join(SCRIPT_DIR, "compute_indicators.py")])
    time.sleep(1)  # 추가: DB 락 해제 지연
    
    subprocess.run(["python", os.path.join(SCRIPT_DIR, "screener.py"), str(use_us), str(use_kr), str(top_n)])
    time.sleep(1)  # 추가: DB 락 해제 지연
    
    # 수정: results_path 환경 변수로
    DATA_DIR = os.getenv('DATA_DIR', './data')
    results_path = os.path.join(DATA_DIR, 'meta', 'screener_results.parquet')
    if os.path.exists(results_path):
        df = pd.read_parquet(results_path)
        numeric_cols = df.select_dtypes(include=['float64']).columns
        for col in numeric_cols:
            df[col] = df[col].round(2)
        msg = f"Top {len(df)} 후보: {df['symbol'].tolist()}" if len(df) > 0 else "후보 없음"
    else:
        msg = "배치 완료 – 결과 확인하세요"
    
    token = "YOUR_TELEGRAM_BOT_TOKEN"
    chat_id = "YOUR_CHAT_ID"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, data={'chat_id': chat_id, 'text': msg})
    
    # 수정: log_dir 환경 변수로
    LOG_DIR = os.getenv('LOG_DIR', './logs')
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, 'batch_time.txt'), "w") as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("배치 완료!")

if __name__ == "__main__":
    use_kr = sys.argv[1] == 'True' if len(sys.argv) > 1 else True
    use_us = sys.argv[2] == 'True' if len(sys.argv) > 2 else True
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    run_batch(use_kr, use_us, top_n)