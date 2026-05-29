# 필요한 라이브러리 설치 및 임포트
# !pip install mplfinance yfinance pandas matplotlib numpy

import datetime
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import yfinance as yf

# 1. 설정 변수 지정
start_date = "2026-04-01"
end_date = "2026-05-29"  # 원하는 마지막 날짜에 + 1일 더해줘야 합니다.

# 분석할 티커 설정 (야후 파인낸스 기준)
# 국내 주식의 경우 종목코드 뒤에 .KS(코스피) 또는 .KQ(코스닥)를 붙여줍니다. (예: '069500.KS')
ticker = "122630.KS"

# 2. 데이터 불러오기
stock_data = yf.download(ticker, start=start_date, end=end_date)
if isinstance(stock_data.columns, pd.MultiIndex) and stock_data.columns.nlevels > 1:
    stock_data.columns = stock_data.columns.get_level_values(0)

# 필요한 컬럼만 선택하고 숫자형으로 변환
data = stock_data[["Open", "High", "Low", "Close"]].astype(float)
data = data.dropna(how="any")

# 지수화된 주가 계산 (첫 거래일을 100으로 기준 추출)
data = data / data.iloc[0] * 100


# 3. 분석 함수 정의
# 이격도 계산 함수
def calculate_disparity(data, ticker, moving_average_period):
    moving_average = data.rolling(window=moving_average_period).mean()
    disparity = (data / moving_average) * 100
    return disparity


# 변동성 계산 함수
def calculate_volatility(data, ticker, period=20, annual_factor=252):
    # 로그 수익률 계산
    log_returns = np.log(data / data.shift(1))
    # 변동성 계산
    volatility = log_returns.rolling(window=period).std() * np.sqrt(annual_factor) * 100
    return volatility


# 4. 지표 계산
# x축 범위 설정을 위한 시작 및 종료 날짜
x_start_date = data.index.min()
x_end_date = data.index.max()

# 1달 단위 날짜 추출 (수직선 표시용)
monthly_dates = pd.date_range(start=start_date, end=end_date, freq="MS")

# 5일 및 20일 이격도 계산
disparity_5d = calculate_disparity(data["Close"], ticker, 5)
disparity_20d = calculate_disparity(data["Close"], ticker, 20)

# 변동성 계산
volatility = calculate_volatility(data["Close"], ticker)


# 5. 그래프 그리기
fig, axes = plt.subplots(
    3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
)

# 캔들차트를 위한 스타일 설정 (상승 빨강, 하락 파랑)
mc = mpf.make_marketcolors(
    up="red", down="blue", edge="black", wick="black", inherit=True
)
s = mpf.make_mpf_style(marketcolors=mc)

# 양옆 2일의 여백 설정
margin = datetime.timedelta(days=2)

# [Subplot 1] 캔들차트 및 이동평균선(5일, 20일)
mpf.plot(
    data,
    ax=axes[0],
    type="candle",
    style=s,
    ylabel="Price",
    mav=(5, 20),
    mavcolors=["red", "black"],  # 5일 평선: 빨강, 20일 평선: 검정
    show_nontrading=True,
    scale_width_adjustment=dict(candle=0.8, volume=0.8),
    xlim=(x_start_date - margin, x_end_date + margin),
)
axes[0].set_title(f"{ticker} Candlestick Chart")

# 월단위 수직선 그리기
for date in monthly_dates:
    axes[0].axvline(x=date, color="gray", linestyle="--", alpha=0.5)

# [Subplot 2] 이격도 그래프 (5일은 바 차트, 20일은 라인 차트)
axes[1].bar(
    disparity_5d.index,
    disparity_5d - 100,
    color="red",
    label=f"{ticker} 5-Day Disparity",
    alpha=0.7,
)
axes[1].plot(
    disparity_20d - 100, color="black", label=f"{ticker} 20-Day Disparity", alpha=0.7
)
axes[1].axhline(0, color="black", linestyle="--")
axes[1].set_title(f"{ticker} 5-Day and 20-Day Disparity")
axes[1].legend(loc="upper left")
axes[1].set_xlim(x_start_date - margin, x_end_date + margin)

for date in monthly_dates:
    axes[1].axvline(x=date, color="gray", linestyle="--", alpha=0.5)

# [Subplot 3] 변동성 그래프
axes[2].plot(volatility, label=f"{ticker} Volatility", color="green")
axes[2].set_title(f"{ticker} Rolling 20-Day Annualized Volatility")
axes[2].legend(loc="upper left")
axes[2].set_xlim(x_start_date - margin, x_end_date + margin)

for date in monthly_dates:
    axes[2].axvline(x=date, color="gray", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()