# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 15:57:04 2023

@author: LAB_2
"""

#%% (1): Basic settings|


# 자주 사용되는 Python library Import
import os
import numpy as np
import pandas as pd
from numpy import genfromtxt
from numpy import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.dates as mdates
import sklearn as sk
import scipy.stats as stats
import matplotlib.font_manager as fm
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 글꼴 경로 설정
font_path = "C:/Users/LAB_2/Desktop/글씨/NanumFontSetup_OTF_SQUARE/NanumSquareB.otf"
font_prop = FontProperties(fname=font_path)

# 현재 파이썬 파일이 저장되어 있는 위치로 Directory 변경
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Baseline 디렉토리 
Base_dir = os.path.dirname(os.path.realpath(__file__))

# Data가 저장된 위치 정의
Data_dir = "/DATA/"

# Data 불러오기 df: 엑셀 파일 원본
df = pd.read_csv(os.path.join(Base_dir+Data_dir +"/all.csv"), index_col = 0,encoding = 'CP949')

# (함수 A): 새로운 열 추가할 때마다 ; 열의 이름을 0부터 시작하는 정수열로 변환하는 함수 정의
def adjust_dataframe_index(df):
    current_column_names = df.columns.tolist()
    new_column_names = list(range(len(current_column_names)))
    column_name_mapping = dict(zip(current_column_names, new_column_names))
    df.rename(columns=column_name_mapping, inplace=True)


# (함수 B): 날짜 형식 바꾸기 ('%d.%m.%Y' => '%Y-%m-%d')
def change_date_format(df, date_column_name):
    Date = df.loc[:, date_column_name]
    Date = Date[1:]
    dates = pd.to_datetime(Date, format='%d.%m.%Y').dt.strftime('%Y-%m-%d')
    dates = pd.concat([pd.Series(['dates']), dates], ignore_index=True)
    dates_list = list(dates)
    df.insert(0, 'date', dates_list)
    df.drop(date_column_name, axis = 1, inplace=True)

# (함수 E) 분단위 시계열을 ‘열’로 가지고 날짜를 ‘행’으로 가지는 빈 <분단위 데이터프레임> 만들기
def minute_abnormal(df, df_adapt): # df : 원본데이터에서 가공만 거친 데이터프레임, 
                                   # df_adapt : off 시간대 & 적응기간까지 모두 제거하여서, 이상거동 검출이 가능한 데이터프레임
    #(1) 열 (=분단위 데이터) 만들기 
    # Time 열에서 겹치는 것 없이 시간만 뽑아와서 Times 리스트 만들기
    Times = list(set(df_adapt.iloc[:, 1])) 
    # 시간 문자열을 시간 형식으로 변환
    import datetime
    times_datetime = [datetime.datetime.strptime(time, '%H:%M') for time in Times] 
    # 시간을 오름차순으로 정렬
    times_sorted = sorted(times_datetime)
    # 정렬된 시간을 문자열 형태로 변환
    times_sorted_str = [time.strftime('%H:%M') for time in times_sorted] 
    
    # (2) 행 (=날짜) 만들기
    map_dates = sorted(list(set(df.iloc[1:, 0])))
    
    minute_abnormal_df = pd.DataFrame(index=map_dates, columns=times_sorted_str)
    return minute_abnormal_df


# (함수 F): 해당 유형의 이상거동이 발견된 시간대 & 연달아 나타난 지속 시간 표시하는 데프 만들기 함수
columns = ["Date", "Start_time", "End_time", "Duration", "Code"]
def abnormal_df(code):  
    consecutive_count = 0
    start_time = None
    severity = None
    results = []
    
    for idx, row in problems.iterrows():
        date = idx
        time_values = row.values
        time_indices = row.index
        for i in range(len(time_values)):
            if time_values[i] == code:
                if consecutive_count == 0:
                    start_time = time_indices[i]
                consecutive_count += 1
            else:
                if consecutive_count > 0:
                    end_time = time_indices[i - 1]
                    duration = consecutive_count
                    hour_sums = []
                    hour_start = int(start_time.split(":")[0])
                    hour_end = int(end_time.split(":")[0])
                    hour_sum = 0
                    for hour in range(hour_start, hour_end + 1):
                        for t in range(len(time_values)):
                            if time_indices[t].startswith(f"{hour:02d}:") and time_values[t] == code:
                                hour_sum += (time_values[t])
                            hour_sums.append(hour_sum)
                    if hour_sums and not np.all(np.isnan(hour_sums)):
                        max_hour_sum = np.nanmax(hour_sums)
                    else:
                        max_hour_sum = None
                    results.append([date, start_time, end_time, duration, code])
                consecutive_count = 0
                start_time = None
    return results


# (함수 G) 이상거동 정리한 데이터프레임의 단위를 "날짜 & 분"에서 "날짜 & 시간"로 바꾼 데이터프레임을 리턴하는 함수
def min_to_hour_new(df):
    df = df.reset_index()
    dates = []
    hours = []
    durations = []
    for i in range(df.shape[0]):
        dates.append(df.iloc[i, 0])
        start_time = df.iloc[i, 1]
        end_time = df.iloc[i, 2]
        duration = df.iloc[i,3]
        start_hour = int(start_time.split(":")[0])
        end_hour = int(end_time.split(":")[0])
        hours.append(start_hour)
        if end_hour != start_hour:
            diff_hour = end_hour - start_hour
            j = 0
            while diff_hour >= 0:
                if  j == 0: #1트 때
                    duration_1 = 60 - int(start_time.split(":")[1])
                    durations.append(duration_1)
                    duration -= duration_1
                elif diff_hour == 0: #막트 때
                    dates.append(df.iloc[i, 0])
                    hours.append(end_hour)
                    durations.append(duration)
                else:
                    duration_n = 60
                    duration -= duration_n
                    dates.append(df.iloc[i, 0])
                    hours.append(start_hour + j)
                    durations.append(duration_n)
                diff_hour -= 1
                j += 1
        else:
            durations.append(duration)
            
    df_grouped = pd.DataFrame()
    df_grouped['Date'] = dates
    df_grouped['Hours'] = hours
    df_grouped['Durations'] = durations
    df_grouped = df_grouped.groupby(['Date', 'Hours'])['Durations'].sum().reset_index()
    return df_grouped


#%% (2) 데이터 가공: 원본데이터 가공해서 DF 만들기, 날짜/시간 분리, 날짜 형식 바꾸기, 초기화 등등

# Raw_Data에서 필요한 정보 만들기, DF: 전체데이터
DF = pd.DataFrame(df)
DF = DF.iloc[1:]  # 필요한 행만 선택 (2번째 행부터)

# 날짜와 시간 분리
DF['Times'] = DF.index
DF[['dates', 'times']] = DF['Times'].str.split(' ', expand=True)
DF.insert(0, 'Date', DF['dates'])
DF.insert(1, 'Time', DF['times'])
DF = DF.drop('Times', axis=1)
DF = DF.drop('dates', axis=1)
DF = DF.drop('times', axis=1)

# 날짜 형식 바꾸기
change_date_format(DF, 'Date')

# 열의 이름을 0부터 시작하는 정수열로 변환
adjust_dataframe_index(DF) 

# 행의 index를 정수로 초기화
DF.reset_index(drop=True, inplace=True)

# DF 안의 요소들 자료형을 숫자형으로 바꾸기
DF.iloc[1:,2:] = DF.iloc[1:,2:].apply(pd.to_numeric, errors='coerce')


#%% (3): 필요한 열 추가: 냉수_공급/환수_dT, 냉수_공급/설정_dT, 냉동기_1분간_전력사용량(kWh), SA_공급/설정_dT, Evaporator_dT, SA 온도 기울기

# 새로운 열 추가 ('냉수_공급/환수_dT')
new_col1 = []
new_col1.append('냉수_공급/환수_dT')
new_col1.extend(DF.iloc[1:,2] - DF.iloc[1:,3])
DF.insert(4, 'new_col1', new_col1)
adjust_dataframe_index(DF) # 새로운 열 추가할 때마다 열의 이름을 0부터 시작하는 정수열로 변환


# 새로운 열 추가 ('냉수_공급/설정_dT')
new_col2 = []
new_col2.append('냉수_공급/설정_dT')
new_col2.extend(DF.iloc[1:,3] - 7)
DF.insert(5, 'new_col2', new_col2)
adjust_dataframe_index(DF) # 새로운 열 추가할 때마다 열의 이름을 0부터 시작하는 정수열로 변환


# 새로운 열 추가 ('냉동기_1분간_전력사용량(kWh)')
chiller_h = DF.iloc[1:,34]
new_col3 = []
new_col3.append('냉동기_1분간_전력사용량(kWh)')
new_col3.append(float(0))
for i in range(2, DF.shape[0]):
    diff = chiller_h[i] - chiller_h[i-1]
    new_col3.append(round(diff, 2))
DF.insert(35, 'new_col3', new_col3)
adjust_dataframe_index(DF) # 새로운 열 추가할 때마다 열의 이름을 0부터 시작하는 정수열로 변환


# 새로운 열 추가 ('SA_공급/설정_dT')
new_col4 = []
new_col4.append('SA_공급/설정_dT')
new_col4.extend(DF.iloc[1:,43] - DF.iloc[1:,42])
DF.insert(44, 'new_col4', new_col4)
adjust_dataframe_index(DF) # 새로운 열 추가할 때마다 열의 이름을 0부터 시작하는 정수열로 변환


# 새로운 열 추가 ('Evaporator_dT')
new_col5 = []
new_col5.append('Evaporator_dT')
new_col5.extend(DF.iloc[1:,94] - DF.iloc[1:,95])
DF.insert(96, 'new_col5', new_col5)
adjust_dataframe_index(DF) # 새로운 열 추가할 때마다 열의 이름을 0부터 시작하는 정수열로 변환


# (2) & (3) 이후의 DF를 따로 저장하기
#DF.to_csv('C:/Users/82107/OneDrive/BIST/5/DF.csv', encoding='CP949')

# 새로운 열 추가 ('SA 온도 기울기')                                              
SA_temp = DF.iloc[1:,43]
SA_gradient =[]
SA_gradient.append('SA_온도_기울기')
SA_gradient.append(float(0))
for i in range(2,DF.shape[0]):
    diff = SA_temp[i] - SA_temp[i-1]
    SA_gradient.append(round(diff,5))
DF.insert(97, 'SA_gradient', SA_gradient)
adjust_dataframe_index(DF) # 새로운 열 추가할 때마다 열의 이름을 0부터 시작하는 정수열로 변환

#%% (4): HVAC 시스템 전체 off를 판단하기 위한 데이터 고르기

# HVAC 시스템 전체 off이기 위해 만족해야 하는 조건 설명(with index)
# 시계열 데이터 ; 'Date' & 'Time' -> [0] & [1]
# 1번 냉각탑 ; '냉각탑 작동 여부' -> [91] == 0
# 2번 냉각수 펌프 ; '냉각수펌프 작동여부' -> [92] == 0
# 4번 냉수 펌프 ; '냉수펌프 작동여부' -> [93] == 0
# (미완성인 5번) AHU 팬 ; 'Fan 회전속도(%)' -> [37] == 0
# 3번-1) '냉동기 전력량(kW)' -> [36] <= 0.1
# 3번-2) '냉동기_1분간_전력사용량(kWh)' -> [35] <= 0.01
# 3번-3) 'Evaporator_dT' -> [96] <= 0       
                      
# ==> 3번-1,2,3 중에서 2개 이상 만족하면 off
# ==> 1, 2, 3, 4, 5번 모두 만족해야 off


# 필요한 데이터 7개 + 시계열 데이터 2개 선택하기
Selected_Data = DF.iloc[:, [0, 1, 91, 92, 93, 37, 36, 35, 96]]
Selected_Data.columns = ['Date', 'Time', '1', '2', '4', '(5)', '3-1', '3-2', '3-3']


# 3번 ; 냉동기 작동여부 판단하는 새로운 열 추가
chiller_off = []
chiller_off.append('냉동기 작동여부')
count = 0
for i in range(1, Selected_Data.shape[0]):
    if Selected_Data.loc[i, '3-1'] <= 0.1:
        count += 1
    if Selected_Data.loc[i, '3-2'] <= 0.01:
        count += 1
    if Selected_Data.loc[i, '3-3'] <= 0:
        count += 1
    if count >= 2:
        chiller_off.append(0)
    else:
        chiller_off.append(1)
    count = 0 # count 초기화
Selected_Data.insert(4, '3', chiller_off)  # Selected_Data 데프에 추가하기
DF['냉동기 작동 여부'] = chiller_off  # DF에 냉동기 작동 여부(0 or 1)에 관한 열도 추가하기


# 5번 ; AHU 팬 작동 여부 판단하는 새로운 열 추가
ahu_off = []
ahu_off.append('AHU 팬 작동 여부')

for i in range(1, Selected_Data.shape[0]):
    if Selected_Data.loc[i, '(5)'] != 0: 
        ahu_off.append(1)
    else:
        ahu_off.append(0)
Selected_Data.insert(6, '5', ahu_off)  # Selected_Data 데프에 추가하기
DF['AHU 팬 작동 여부'] = ahu_off  # DF에 ahu 작동 여부(0 or 1)에 관한 열도 추가하기


# 열 이름 1~5의 요소들 합계가 0일 때 전체 시스템 off라고 판단
DT_sum = Selected_Data.iloc[:, 2:7].copy().sum(axis=1)
DT_sum.iloc[0] = 'sum'
Selected_Data.insert(2, 'Sum', DT_sum)
sys_off = []  # sys_off ; 전체 시스템 off면 0, 전체 시스템 on이면 1이라고 표시하는 열 추가
sys_off.append('전체 시스템 ON/OFF')
idx_sys_off = []  # idx_sys_off ; 전체 시스템 off인 인덱스는 drop하기 위해 필요한 리스트
idx_tower_off = []  # 냉각타워 off인 인덱스는 drop하기 위해 필요한 리스트
idx_chiller_off = []  # 냉동기 off인 인덱스는 drop하기 위해 필요한 리스트
idx_ahu_off = []  # AHU 팬 off인 인덱스는 drop하기 위해 필요한 리스트

for i in range(1, Selected_Data.shape[0]):
    if Selected_Data.loc[i, 'Sum'] == 0:
        sys_off.append(0)
        idx_sys_off.append(i)
    else:
        sys_off.append(1)
        if Selected_Data.loc[i, '1'] == 0:
            idx_tower_off.append(i)
        if Selected_Data.loc[i, '3'] == 0:
            idx_chiller_off.append(i)
        if Selected_Data.loc[i, '5'] == 0:
            idx_ahu_off.append(i)
Selected_Data.insert(2, 'sys_off', sys_off)

#%% (5): 분 단위 / 시간 단위(4가지 종류) 히트맵을 위한 데이터프레임 만들기


# 1. 1분 단위로 on/off 표시하는 데이터프레임 4개 만들기
map_dates = sorted(list(set(Selected_Data.iloc[1:, 0])))  # 'Date' 열에서 겹치지 않는 날짜만 추출하고, 원하는 형식으로 변환
map_times = sorted(list(set(Selected_Data.iloc[1:, 1])))  # 분단위 'Time' 열에서 겹치지 않는 시간만 추출하고, 원하는 형식으로 변환
map_sys_off = pd.DataFrame(index=map_dates, columns=map_times)  # hvac 시스템 전체
map_tower = pd.DataFrame(index=map_dates, columns=map_times)  # 냉각타워
map_chiller = pd.DataFrame(index=map_dates, columns=map_times)  # 냉동기
map_ahu = pd.DataFrame(index=map_dates, columns=map_times)  # AHU 팬


# 2. 1시간 단위로 on/off 표시하는 데이터프레임 만들기 (히트맵 그리기용)
# <1-1> ~ <1-4>
map_hour = [str(hour).zfill(2) for hour in range(0, 24)]  # 시간 단위 'Hour' 열 만들기
hour_map_sys_off = pd.DataFrame(index=map_dates, columns=map_hour)  # <1-1> 시스템 전체 ON 히트맵
hour_map_tower = pd.DataFrame(index=map_dates, columns=map_hour)  # <1-2> 냉각타워 ON 히트맵
hour_map_chiller = pd.DataFrame(index=map_dates, columns=map_hour)  # <1-3> 냉동기 ON 히트맵
hour_map_ahu = pd.DataFrame(index=map_dates, columns=map_hour)  # <1-4> AHU 팬 ON 히트맵

# <2-1> ~ <2-3>
hour_map_tower_off = pd.DataFrame(index=map_dates, columns=map_hour)  # <2-1> 전체 시스템 ON인데 냉각타워는 OFF인 거동 히트맵
idx_sys_on_tower_off = []  # 냉각타워는 off
hour_map_chiller_off = pd.DataFrame(index=map_dates, columns=map_hour)  # <2-2> 전체 시스템 ON인데 냉동기는 OFF인 거동 히트맵
idx_sys_on_chiller_off = []  # 냉동기는 off
hour_map_ahu_off = pd.DataFrame(index=map_dates, columns=map_hour)  # <2-3> 전체 시스템 ON인데 AHU 팬는 OFF인 거동 히트맵
idx_sys_on_ahu_off = []  # AHU 팬은 off

# <3-1> ~ <3-3>
hour_map_only_tower_off = pd.DataFrame(index=map_dates, columns=map_hour)  # <3-1> 나머지는 다 ON인데 냉각타워만 OFF인 거동 히트맵
idx_only_tower_off = []  # 냉각타워만 off
hour_map_only_chiller_off = pd.DataFrame(index=map_dates, columns=map_hour)  # <3-2> 나머지는 다 ON인데 냉동기만 OFF인 거동 히트맵
idx_only_chiller_off = []  # 냉동기만 off
hour_map_only_ahu_off = pd.DataFrame(index=map_dates, columns=map_hour)  # <3-3> 나머지는 다 ON인데 AHU 팬만 OFF인 거동 히트맵
idx_only_ahu_off = []  # AHU 팬만 off

# <4-1> ~ <4-3>
# 외기냉방 ; 냉동기, 냉각타워 꺼져있고, ahu는 켜져있고, 펌프 상관 없고
# 정상 ; 펌프가 둘다 꺼져있는 경우
# 비정상 ; 펌프 둘 중에 하나라도 켜져있는 경우
hour_map_oac = pd.DataFrame(index=map_dates, columns=map_hour)   # <4-1> 외기냉방
idx_oac = [] # 외기냉방
hour_map_oac_normal = pd.DataFrame(index=map_dates, columns=map_hour)   # <4-2> 외기냉방 정상
idx_oac_normal = [] # 외기냉방 정상
hour_map_oac_abnormal = pd.DataFrame(index=map_dates, columns=map_hour)   # <4-3> 외기냉방 비정상
idx_oac_abnormal = [] # 외기냉방 비정상


# 함수 data_exist_mark_0
def data_exist_mark_0(df_heatmap, df): # df_heatmap : 히트맵을 그리기 위한 데이터프레임, df : 기존 데이터프레임
    for i in range(1, df.shape[0]):    
        date = df.iloc[i, 0] # df의 0번째 열이 날짜 열이어야 함
        time = df.iloc[i, 1] # df의 1번째 열이 시간 열이어야 함
        hours_minutes = time.split(':') # 시간과 분 분리
        hour = hours_minutes[0] # 시간만 뽑아내기
        #print(date, hour)
        if np.isnan(df_heatmap.loc[date, hour]) == 1: # 기존 데이터프레임 df에 데이터가 존재하고, 해당 시간대가 df_heatmap에 nan이라고 표시되어 있다면,
            df_heatmap.loc[date, hour] = 0 # df_heatmap에 우선 0으로 표시하기


#  
# for i in range(1, Selected_Data.shape[0]):
#     date = Selected_Data.iloc[i, 0]
#     time = Selected_Data.iloc[i, 1]
#     hours_minutes = time.split(':')
#     hour = hours_minutes[0]
#     minutes = [int(hm[1]) for hm in hours_minutes]
    
#     value_sys = Selected_Data.iloc[i, 2]
#     value_tower = Selected_Data.iloc[i, 4]
#     value_cooling_water = Selected_Data.iloc[i, 5]
#     value_chiller = Selected_Data.iloc[i, 6]
#     value_cold_water = Selected_Data.iloc[i, 7]
#     value_ahu = Selected_Data.iloc[i, 8]
    
#     # 1. 분단위 on/off 표시 데프
#     map_sys_off.loc[date, time] = value_sys
#     map_tower.loc[date, time] = value_tower
#     map_chiller.loc[date, time] = value_chiller
#     map_ahu.loc[date, time] = value_ahu
    
#     # 2. 시간단위 on/off 표시 데프
#     # <1-1> 시스템 전체 ON 히트맵
#     hour_map_sys_off.loc[date, hour] += value_sys

#     # <1-2> 냉각타워 ON 히트맵
#     hour_map_tower.loc[date, hour] += value_tower
    
#     # <1-3> 냉동기 ON 히트맵
#     hour_map_chiller.loc[date, hour] += value_chiller
    
#     # <1-4> AHU 팬 ON 히트맵
#     hour_map_ahu.loc[date, hour] += value_ahu
    
#     # <2-1> 전체 시스템 ON인데 냉각타워는 OFF인 거동 히트맵
#     if value_tower == 0 and value_sys != 0:
#         #print(value_sys, date, hour)
#         hour_map_tower_off.loc[date, hour] += value_sys
#         idx_sys_on_tower_off.append(i)

#     # <2-2> 전체 시스템 ON인데 냉동기는 OFF인 거동 히트맵
#     if value_chiller == 0 and value_sys != 0:
#         #print(value_sys, date, hour)
#         hour_map_chiller_off.loc[date, hour] += value_sys
#         idx_sys_on_chiller_off.append(i)
    
#     # <2-3> 전체 시스템 ON인데 AHU 팬만 OFF인 거동 히트맵
#     if value_ahu == 0 and value_sys != 0:
#         #print(value_sys, date, hour)
#         hour_map_ahu_off.loc[date, hour] += value_sys
#         idx_sys_on_ahu_off.append(i)
        
#     # <3-1> 냉각타워만 OFF인 거동 히트맵
#     if value_tower == 0 and value_sys != 0 and value_cooling_water != 0 and value_chiller != 0 and value_cold_water != 0 and value_ahu != 0:
#         hour_map_only_tower_off.loc[date, hour] += 1
#         idx_only_tower_off.append(i)
    
#     # <3-2> 냉동기만 OFF인 거동 히트맵
#     if value_chiller == 0 and value_sys != 0 and value_cooling_water != 0 and value_tower != 0 and value_cold_water != 0 and value_ahu != 0:
#         hour_map_only_chiller_off.loc[date, hour] += 1
#         idx_only_chiller_off.append(i)
        
#     # <3-3> AHU 팬만 OFF인 거동 히트맵
#     if value_ahu == 0 and value_sys != 0 and value_cooling_water != 0 and value_chiller != 0 and value_cold_water != 0 and value_tower != 0:
#         hour_map_only_ahu_off.loc[date, hour] += 1
#         idx_only_ahu_off.append(i)
        
#     # <4-1> 외기냉방 히트맵
#     if value_tower == 0 and value_chiller == 0 and value_ahu != 0:
#         hour_map_oac.loc[date, hour] += 1
#         idx_oac.append(i)
    
#     # <4-2> 외기냉방 정상 히트맵
#     if value_tower == 0 and value_chiller == 0 and value_ahu != 0 and value_cooling_water == 0 and value_cold_water == 0:
#         hour_map_oac_normal.loc[date, hour] += 1
#         idx_oac_normal.append(i)
        
#     # <4-3> 외기냉방 비정상 히트맵
#     if i in idx_oac and not i in idx_oac_normal:
#         hour_map_oac_abnormal.loc[date, hour] += 1
#         idx_oac_abnormal.append(i)


#%% (6) HVAC off 시간대 제거하기 - 전체 시스템 off, Tower off

# HVAC 시스템 자체가 off인 시간대 삭제할 데이터프레임 ; df_sys
df_sys = DF.copy()

# 전체 시스템 off 시간대 제거
df_sys = df_sys.drop(idx_sys_off)

# 냉각탑 off 시간대 제거
df_sys_tower = df_sys.copy()
df_sys_tower = df_sys_tower.drop(idx_tower_off)


#%% (7) 루프1: 이상거동 진단 시작

# 원하는 열 선택하기

Tower_number = [0, 1, 6, 31, 32, 91] 
# (0)Date,(1)Time,(2)냉각수 공급온도,(3)냉각타워 팬 주파수,(4)냉각타워 팬 동력,(5)냉동기 작동 여부

Tower_df = df_sys_tower.iloc[:,Tower_number] # Tower_df: 이상거동 진단 시 기본 데이터프레임
# Tower_df 열 이름 바꾸기 (0번 행을 열 이름으로 올리기)
Tower_df_new_column_names = Tower_df.iloc[0]
Tower_df = Tower_df[1:]
Tower_df.columns = Tower_df_new_column_names
Tower_df.reset_index(drop=True, inplace=True)

#%% (7-1) 이상거동 진단: 1,2,3
# 이상거동1 - 알고리즘: 냉각수 공급온도 < 28℃ + 냉각탑 팬 주파수 > 60% 
# 이상거동2 - 알고리즘: 냉각수 공급온도 > 32℃ + 냉각탑 팬 주파수 < 100% 
# 이상거동3 - 알고리즘: 냉각수 공급온도 > 32℃ + 냉각탑 팬 주파수 = 100% 
hp_supplytemp_1=28
hp_frequency_1=60
hp_supplytemp_2=32
hp_frequency_2=100
# 이상거동 유형 표시하는 열 SA_df_adaptX에 추가 (전수조사)
def Tower_abnormal(df):
    
    df_df = df.copy()
    
    abnormal_123 = []
    
    # 조건따라 이상거동 유형 분류 & 리스트에 판단요소 넣기
    for i in range(df.shape[0]):
        condition1 = df.iloc[i,2] < hp_supplytemp_1 and df.iloc[i,3] > hp_frequency_1
        condition2 = df.iloc[i,2] > hp_supplytemp_2 and df.iloc[i,3] < hp_frequency_2
        condition3 = df.iloc[i,2] > hp_supplytemp_2 and df.iloc[i,3] == hp_frequency_2

        if condition1:
            abnormal_123.append(1)
        elif condition2:
            abnormal_123.append(2)
        elif condition3:
            abnormal_123.append(3)
        else:
            abnormal_123.append(0)
    
    df_df['이상거동 유형 1,2,3'] = abnormal_123

    return df_df

Tower_Abnormal_df_all = Tower_abnormal(Tower_df)

#%% (7-2) 전수조사 요약본

Columns = ["Date", "Start_time", "End_time", "Duration", "Code"]
def abnormal_all_summary(df, code): #code는 유형. 1,2,3,4 입력 가능.
    
    df_df = df.copy()
    consecutive_count = 0
    start_time = None
    results = []
    
    if code == 1 or code == 2 or code == 3:
        ab_row = df_df['이상거동 유형 1,2,3']
        
    for i in range(df_df.shape[0]):
        
        if  ab_row[i] == code:
            if consecutive_count == 0:
                date = df_df.iloc[i,0]
                start_time = df_df.iloc[i,1]
                if i == df_df.shape[0] - 1:
                    time_check = df_df.iloc[i-5,1] #Time이 다르면 문제 없으므로 임의로 5칸 뒤 시간으로 설정.
                else:
                    time_check = df_df.iloc[i+1,1]
                if int(start_time.split(":")[1]) + 1 != int(time_check.split(":")[1]):#11:30 다음, 11:50인데 둘 다 이상거동일 때 기록하는 예외상황
                    consecutive_count += 1
                    end_time = df_df.iloc[i,1]
                    duration = consecutive_count
                    results.append([date, start_time, end_time, duration, code])
                    consecutive_count = 0
                    start_time = None
                    continue
                        
            if i == df_df.shape[0] - 1:
                date_check = date #같으면 문제X
            else:
                date_check = df_df.iloc[i+1,0]
            if date != date_check: #날짜 바뀌었을 때
                if consecutive_count > 0:
                    end_time = df_df.iloc[i,1]
                    duration = consecutive_count + 1
                    results.append([date, start_time, end_time, duration, code])
                consecutive_count = 0
                start_time = None
                continue
                
            consecutive_count += 1
            
        else:
            if consecutive_count > 0:
                end_time = df_df.iloc[i-1,1]
                duration = consecutive_count
                results.append([date, start_time, end_time, duration, code])
            consecutive_count = 0
            start_time = None
    
    return results
Tower_Abnormal_1_all = pd.DataFrame(abnormal_all_summary(Tower_Abnormal_df_all, 1), columns = Columns)
Tower_Abnormal_2_all = pd.DataFrame(abnormal_all_summary(Tower_Abnormal_df_all, 2), columns = Columns)
Tower_Abnormal_3_all = pd.DataFrame(abnormal_all_summary(Tower_Abnormal_df_all, 3), columns = Columns)

#%% (7-3) 이상거동 1,2,3 지속시간 기준 적용
hp_duration_type1=0
hp_duration_type2=0
hp_duration_type3=0
def Tower_real_abnormal(df):
    
    df_df = df.copy()
    not_real = []
    if df_df.loc[0,'Code'] == 1:
        for i in range(df_df.shape[0]):
            if df_df.loc[i,'Duration'] < hp_duration_type1: #20분 이상일 때만 진짜 이상거동으로 취급
                not_real.append(i)
    if df_df.loc[0,'Code'] == 2:
        for i in range(df_df.shape[0]):
            if df_df.loc[i,'Duration'] < hp_duration_type2: #20분 이상일 때만 진짜 이상거동으로 취급
                not_real.append(i)
    if df_df.loc[0,'Code'] == 3:
        for i in range(df_df.shape[0]):
            if df_df.loc[i,'Duration'] < hp_duration_type3: #20분 이상일 때만 진짜 이상거동으로 취급
                not_real.append(i)
    
    df_df = df_df.drop(not_real)
    
    return df_df

Tower_Abnormal_1 = Tower_real_abnormal(Tower_Abnormal_1_all)
Tower_Abnormal_1.set_index("Date", inplace=True)
Tower_Abnormal_2 = Tower_real_abnormal(Tower_Abnormal_2_all)
Tower_Abnormal_2.set_index("Date", inplace=True)
Tower_Abnormal_3 = Tower_real_abnormal(Tower_Abnormal_3_all)
Tower_Abnormal_3.set_index("Date", inplace=True)


#%% (7-4) 시간 단위로 올리기

Tower_Abnormal_1_grouped = min_to_hour_new(Tower_Abnormal_1)
Tower_Abnormal_2_grouped = min_to_hour_new(Tower_Abnormal_2)
Tower_Abnormal_3_grouped = min_to_hour_new(Tower_Abnormal_3)

#%% (8) 히트맵 그리기

# <<히트맵 그리는 과정 총 4단계>>

# EX) 히트맵 그리기 위한 데이터프레임의 이름은 "df_heatmap", 기존 데이터프레임은 "DF"일 때

# [히트맵 그리기 1단계]: 1시간 단위의 히트맵 전용 데이터프레임 df_heatmap 만들기
map_dates = sorted(list(set(DF.iloc[1:, 0])))  # 원본 데이터프레임을 가공한 DF의 'Date' 열에서 겹치지 않는 날짜만 추출
map_hour = [str(hour).zfill(2) for hour in range(0, 24)]  # 00시~23시까지 시간 단위 'Hour' 열 만들기
df_heatmap = pd.DataFrame(index=map_dates, columns=map_hour)  # 히트맵


# [히트맵 그리기 2단계]: 기존 데이터프레임 df에 데이터가 있으면, df_heatmap에 무조건 0으로 표시, 없으면 -10 표시
def data_exist_makr_0(df_heatmap, df): # df_heatmap : 히트맵을 그리기 위한 데이터프레임, df : 기존 데이터프레임
    for i in range(1, df.shape[0]):    
        date = df.iloc[i, 0] # df의 0번째 열이 날짜 열이어야 함
        time = df.iloc[i, 1] # df의 1번째 열이 시간 열이어야 함
        hours_minutes = time.split(':') # 시간과 분 분리
        hour = hours_minutes[0] # 시간만 뽑아내기
        #print(date, hour)
        if np.isnan(df_heatmap.loc[date, hour]) == 1: # 기존 데이터프레임 df에 데이터가 존재하고, 해당 시간대가 df_heatmap에 nan이라고 표시되어 있다면,
            df_heatmap.loc[date, hour] = 0 # df_heatmap에 우선 0으로 표시하기
    return df_heatmap
            
def data_nonexist_mark_minus10(df_heatmap):
    df_heatmap = df_heatmap.fillna(-10)
    return df_heatmap
    
df_heatmap = data_exist_makr_0(df_heatmap, DF)
df_heatmap = data_nonexist_mark_minus10(df_heatmap)


# [히트맵 그리기 3단계]: 이상 거동 히트맵 그리기
# (각자 dt_heatmap에 유효한 값 채우기 ; 1시간당 이상거동 유형 *번이 *분 발생하는지)


# [히트맵 그리기 4단계]: 이상 거동 히트맵 그리기
# (라이브러리 불러오기)
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib.font_manager import FontProperties

# (글꼴 경로 설정)
font_path = "C:/Users/LAB_2/Desktop/글씨/NanumFontSetup_OTF_SQUARE/NanumSquareB.otf"
font_prop = FontProperties(fname=font_path)

# (빨간색 히트맵 그리는 함수)
def abnormal_heat_map(df_heatmap, df_name): # df : 히트맵 전용 데이터프레임, df_name : 히트맵의 제목
    plt.figure(figsize=(8, 10))
    sns.heatmap(df_heatmap, cmap='YlOrRd', linewidths=0.5, linecolor="lightgray", annot=False, vmin=-10, vmax=60)
    plt.xlabel("Hour", fontsize=10, fontproperties=font_prop)
    plt.ylabel("Date", fontsize=10, fontproperties=font_prop)
    plt.title(df_name, fontsize=10, fontproperties=font_prop)
    plt.xticks(rotation=0)
    plt.show()
    
# (파란색 히트맵 그리는 함수)
def heat_map(df_heatmap, df_name): # df : 히트맵 전용 데이터프레임, df_name : 히트맵의 제목
    plt.figure(figsize=(8, 10))
    sns.heatmap(df_heatmap, cmap='Blues', linewidths=0.5, linecolor="lightgray", annot=False, vmin=-10, vmax=60)
    plt.xlabel("Hour", fontsize=10, fontproperties=font_prop)
    plt.ylabel("Date", fontsize=10, fontproperties=font_prop)
    plt.title(df_name, fontsize=10, fontproperties=font_prop)
    plt.xticks(rotation=0)
    plt.show()


# 히트맵 그리기 - 이상거동 유형1: Tower_map_1

# [히트맵 그리기 1단계]: 1시간 단위의 히트맵 전용 데이터프레임 df_heatmap 만들기
Tower_map_1 = pd.DataFrame()
Tower_map_1 = pd.DataFrame(index=map_dates, columns=map_hour)  # 히트맵

# [히트맵 그리기 2단계]: 기존 데이터프레임 df에 데이터가 있으면, df_heatmap에 무조건 0으로 표시, 없으면 -10 표시
Tower_map_1 = data_exist_makr_0(Tower_map_1, Tower_df)
Tower_map_1 = data_nonexist_mark_minus10(Tower_map_1)

# [히트맵 그리기 3단계]: 이상 거동 히트맵 그리기
# (각자 dt_heatmap에 유효한 값 채우기 ; 1시간당 이상거동 유형 *번이 *분 발생하는지)
for i in range(0, Tower_Abnormal_1_grouped.shape[0]):
    date = str(Tower_Abnormal_1_grouped.iloc[i, 0])
    time = "{:02d}".format(Tower_Abnormal_1_grouped.iloc[i, 1])
    value = Tower_Abnormal_1_grouped.iloc[i, 2]
    Tower_map_1.loc[date, time] = value

# [히트맵 그리기 4단계]: 이상 거동 히트맵 그리기
abnormal_heat_map(Tower_map_1, f"loopA #type 1 (Incidence in 1h >= {hp_duration_type1}m) <heat-map>")


# 히트맵 그리기 - 이상거동 유형2: Tower_map_2

# [히트맵 그리기 1단계]: 1시간 단위의 히트맵 전용 데이터프레임 df_heatmap 만들기
Tower_map_2 = pd.DataFrame()
Tower_map_2 = pd.DataFrame(index=map_dates, columns=map_hour)  # 히트맵

# [히트맵 그리기 2단계]: 기존 데이터프레임 df에 데이터가 있으면, df_heatmap에 무조건 0으로 표시, 없으면 -10 표시
Tower_map_2 = data_exist_makr_0(Tower_map_2, Tower_df)
Tower_map_2 = data_nonexist_mark_minus10(Tower_map_2)

# [히트맵 그리기 3단계]: 이상 거동 히트맵 그리기
# (각자 dt_heatmap에 유효한 값 채우기 ; 1시간당 이상거동 유형 *번이 *분 발생하는지)
for i in range(0, Tower_Abnormal_2_grouped.shape[0]):
    date = str(Tower_Abnormal_2_grouped.iloc[i, 0])
    time = "{:02d}".format(Tower_Abnormal_2_grouped.iloc[i, 1])
    value = Tower_Abnormal_2_grouped.iloc[i, 2]
    Tower_map_2.loc[date, time] = value

# [히트맵 그리기 4단계]: 이상 거동 히트맵 그리기
abnormal_heat_map(Tower_map_2, f"loopA #type 2 (Incidence in 1h >= {hp_duration_type2}m) <heat-map>")


# 히트맵 그리기 - 이상거동 유형3: Tower_map_3

# [히트맵 그리기 1단계]: 1시간 단위의 히트맵 전용 데이터프레임 df_heatmap 만들기
Tower_map_3 = pd.DataFrame()
Tower_map_3 = pd.DataFrame(index=map_dates, columns=map_hour)  # 히트맵

# [히트맵 그리기 2단계]: 기존 데이터프레임 df에 데이터가 있으면, df_heatmap에 무조건 0으로 표시, 없으면 -10 표시
Tower_map_3 = data_exist_makr_0(Tower_map_3, Tower_df)
Tower_map_3 = data_nonexist_mark_minus10(Tower_map_3)

# [히트맵 그리기 3단계]: 이상 거동 히트맵 그리기
# (각자 dt_heatmap에 유효한 값 채우기 ; 1시간당 이상거동 유형 *번이 *분 발생하는지)
for i in range(0, Tower_Abnormal_3_grouped.shape[0]):
    date = str(Tower_Abnormal_3_grouped.iloc[i, 0])
    time = "{:02d}".format(Tower_Abnormal_3_grouped.iloc[i, 1])
    value = Tower_Abnormal_3_grouped.iloc[i, 2]
    Tower_map_3.loc[date, time] = value

# [히트맵 그리기 4단계]: 이상 거동 히트맵 그리기
abnormal_heat_map(Tower_map_3, f"loopA #type 3 (Incidence in 1h >= {hp_duration_type3}m) <heat-map>")    
    
#%% (8-1) 비율 조사

# 이상거동 유형 1 
# 전체 데이터 ; 45853분
sum_of_time = int(45853)
sum_of_abnormal_1 = int(Tower_Abnormal_1_grouped.iloc[:, 2].sum())
print("Tower - abnormal behavior #type 1 (Incidence in 1h >= 20m) <Incidence Rate>: " + str(round((sum_of_abnormal_1/sum_of_time)*100, 2)) + "%")

# 이상거동 유형 2
# 전체 데이터 ; 45853분
sum_of_time = int(45853)
sum_of_abnormal_2 = int(Tower_Abnormal_2_grouped.iloc[:, 2].sum())
print("Tower - abnormal behavior #type 2 (Incidence in 1h > 20m) <Incidence Rate>: " + str(round((sum_of_abnormal_2/sum_of_time)*100, 2)) + "%")

# 이상거동 유형 3
# 전체 데이터 ; 45853분
sum_of_time = int(45853)
sum_of_abnormal_3 = int(Tower_Abnormal_3_grouped.iloc[:, 2].sum())
print("Tower - abnormal behavior #type 3 (Incidence in 1h > 20m) <Incidence Rate>: " + str(round((sum_of_abnormal_3/sum_of_time)*100, 2)) + "%")
