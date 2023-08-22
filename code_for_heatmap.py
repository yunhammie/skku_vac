# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 17:25:20 2023

@author: 82107
"""

# 히트맵을 그리기 위해 히트맵과 동일하게 생긴 데이터프레임 hour_map_1을 채워넣는 과정 설명

# 1. 데이터프레임 hour_map_1 준비

# 1) 'Date' 열에서 겹치지 않는 날짜만 추출하고, 원하는 형식으로 변환
DF = pd.read_csv('C:/Users/82107/OneDrive/BIST/5/DF.csv', encoding='CP949')
all_Date = DF.iloc[1:, 1]
all_dates = pd.to_datetime(all_Date, format='%d.%m.%Y').dt.strftime('%Y-%m-%d')
map_dates = list(set(all_dates))
map_dates = sorted(map_dates)

# 2) 시간 단위 'Hour' 열 만들기
map_hour = [str(hour).zfill(2) for hour in range(0, 24)]

# 3) 데이터프레임 hour_map_1의 열과 행 지정
hour_map_1 = pd.DataFrame(index=map_dates, columns=map_hour)


# 2. hour_map_1 채워 넣기

# 1) 데이터가 존재하는 날짜&시간의 경우, 우선 0 집어넣기
for i in range(0, sample1_1.shape[0]):
    date = sample1_1.iloc[i, 0]
    time = sample1_1.iloc[i, 1].split(':')[0]
    hour_map_1.loc[date, time] = 0

# 2) 이상거동이 나타난 날짜&시간의 경우, 해당 현상이 나타난 기간(분 단위)에 해당하는 value 집어넣기  
for i in range(0, df_neg.shape[0]):
    date = merged_df.iloc[i, 0]
    time = merged_df.iloc[i, 1]
    value = merged_df.iloc[i, 2]
    hour_map_1.loc[date, time] = value
    
# 3) 누락된 값은 차별화하기 위해 -10으로 채우기
# * 1), 2) 이후 여전히 NaN값인 것은 주어진 데이터에 없는 값임
hour_map_1_filled = hour_map_1.fillna(-10)


# 3. 히트맵 그리기
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))

# vmin=-10 ; 누락된 값이 존재하는 블럭의 채도를 가장 낮게 표시
# vamx=60; 1시간동안 60분 내내 이상거동이 나타난 블럭의 채도를 가장 높게 표시
sns.heatmap(hour_map_1_filled, cmap='YlOrRd', linewidths=0.5, linecolor="lightgray", annot=False, vmin=-10, vmax=60) 

plt.title("chiller pump - abnormal behavior #type 1 <heat-map>") # 예시
plt.xlabel("Hour")
plt.ylabel("Date")
plt.show()