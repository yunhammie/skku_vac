

#%%
# (1): 자주 사용되는 Python library Import & 함수 지정

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
font_path = r"C:\Users\82107\OneDrive\BIST\5\NANUMSQUAREB.TTF"
font_prop = FontProperties(fname=font_path)



#%% (2): 데이터 불러오고 기본적인 가공하기

# 데이터 불러오기, df: 엑셀파일 원본
df = pd.read_csv(r"C:\Users\82107\OneDrive\BIST\5\all.csv", index_col=0, encoding='CP949')

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

#%% (3): 분단위의 각 루프별 이상거동 유형(1~3) 진단 결과를 하나의 데이터프레임 틀 만들기

# 세로축이 될 시계열 정보 뽑아내기 (데이터에 존재하는 것만)
time_info = []
for i in range(1, DF.shape[0]):
    date = DF.iloc[i, 0]
    time = DF.iloc[i, 1]
    info = date + " " + time
    time_info.append(info)

# 가로축이 될 루프 정보 담기
loops = ["A", "B", "C", "D", "E", "F", "G-1", "G-2", "G-3"]

# 모든 루프의 이상거동 유형(1~3) 진단 결과 & 핵심 장치 on/off 정보를 담은 데이터프레임 all_loop_df 만들기
all_loop_df = pd.DataFrame(index=time_info, columns=loops)


#%% (4) : 전체 시스템 및 장치별로 on/off 정보가 담긴 파일들 불러오고 가공하기

# on/off 정보가 담긴 csv 파일들 불러오기
map_sys = pd.read_csv('C:/Users/82107/OneDrive/BIST/5/sys_info.csv', encoding='CP949')
map_tower = pd.read_csv('C:/Users/82107/OneDrive/BIST/5/tower_info.csv', encoding='CP949')
map_chiller = pd.read_csv('C:/Users/82107/OneDrive/BIST/5/chiller_info.csv', encoding='CP949')
map_ahu = pd.read_csv('C:/Users/82107/OneDrive/BIST/5/ahu_info.csv', encoding='CP949')

# 인덱스 재지정
index_value = map_sys.columns[0]
map_sys = map_sys.set_index(index_value)
map_tower = map_tower.set_index(index_value)
map_chiller = map_chiller.set_index(index_value)
map_ahu = map_ahu.set_index(index_value)

#%% (5) : 전체 시스템 및 장치별로 on/off 정보 담기

def off_minus_1(df_info, df_result, loop_list):
    for idx, row in df_info.iterrows():
        values = row.values
        times = row.index
        date = idx
        for j in range(len(times)):
            if df_info.loc[date, times[j]] == 0:
                time_info = date + " " + times[j]
                for k in df_result.columns:
                    if k in loop_list:
                        df_result.loc[time_info, k] = -1
            elif df_info.loc[date, times[j]] == 1:
                time_info = date + " " + times[j]
                for k in df_result.columns:
                    if k in loop_list:
                        df_result.loc[time_info, k] = 0

# 모든 루프에 대하여 전체 system 꺼진 것 -1로 표시하기
off_minus_1(map_sys, all_loop_df, ["A", "B", "C", "D", "E", "F", "G-1", "G-2", "G-3"])
# 루프 A, B 냉각타워 꺼진 것 -1로 표시하기
off_minus_1(map_tower, all_loop_df, ["A", "B"])
# 루프 C, D 냉동기 꺼진 것 -1로 표시하기
off_minus_1(map_chiller, all_loop_df, ["C", "D"])
# 루프 E, F AHU 팬 꺼진 것 -1로 표시하기
off_minus_1(map_ahu, all_loop_df, ["E", "F"])

#%% (6) : 이상거동 유형(1~3) 진단 결과 정보가 담긴 csv 파일들 불러오고 가공하기

# 이상거동 유형(1~3) 진단 결과 정보가 담긴 csv 파일들 불러오기
result_loopA = pd.read_csv('C:/Users/82107/OneDrive/BIST/5/result_loopA.csv', encoding='CP949')
result_loopB = pd.read_csv('C:/Users/82107/OneDrive/BIST/5/result_loopB.csv', encoding='CP949')
result_loopC = pd.read_csv('C:/Users/82107/OneDrive/BIST/5/result_loopC.csv', encoding='CP949') 
result_loopD = pd.read_csv('C:/Users/82107/OneDrive/BIST/5/result_loopD.csv', encoding='CP949') 
result_loopE = pd.read_csv('C:/Users/82107/OneDrive/BIST/5/result_loopE.csv', encoding='CP949')
result_loopF = pd.read_csv('C:/Users/82107/OneDrive/BIST/5/result_loopF.csv', encoding='CP949') 
result_loopG_1 = pd.read_csv('C:/Users/82107/OneDrive/BIST/5/result_loopG_1.csv', encoding='CP949')
result_loopG_2 = pd.read_csv('C:/Users/82107/OneDrive/BIST/5/result_loopG_2.csv', encoding='CP949')
result_loopG_3 = pd.read_csv('C:/Users/82107/OneDrive/BIST/5/result_loopG_3.csv', encoding='CP949')


# 인덱스 재지정
def set_index_and_return(column_name, dataframe):
    index_value = dataframe.columns[0]
    return dataframe.set_index(index_value)

result_loopA = set_index_and_return(result_loopA.columns[0], result_loopA)
result_loopB = set_index_and_return(result_loopB.columns[0], result_loopB)
result_loopC = set_index_and_return(result_loopC.columns[0], result_loopC)
result_loopD = set_index_and_return(result_loopD.columns[0], result_loopD)
result_loopE = set_index_and_return(result_loopE.columns[0], result_loopE)
result_loopF = set_index_and_return(result_loopF.columns[0], result_loopF)
result_loopG_1 = set_index_and_return(result_loopG_1.columns[0], result_loopG_1)
result_loopG_2 = set_index_and_return(result_loopG_2.columns[0], result_loopG_2)
result_loopG_3 = set_index_and_return(result_loopG_3.columns[0], result_loopG_3)


#%% (7) : 분단위의 각 루프별 이상거동 유형(1~3) 진단 결과 담기

# 분단위의 각 루프별 이상거동 유형(1~3) 진단 결과를 하나의 데이터프레임에 담아내기
def abnormal_value(df_info, df_result, loop_list):
    for idx, row in df_info.iterrows():
        values = row.values
        times = row.index
        date = idx
        for j in range(len(times)):
            if df_info.loc[date, times[j]] == 1:
                time_info = date + " " + times[j]
                for k in df_result.columns:
                    if k in loop_list:
                        df_result.loc[time_info, k] = 1
            elif df_info.loc[date, times[j]] == 2:
                time_info = date + " " + times[j]
                for k in df_result.columns:
                    if k in loop_list:
                        df_result.loc[time_info, k] = 2
            elif df_info.loc[date, times[j]] == 3:
                time_info = date + " " + times[j]
                for k in df_result.columns:
                    if k in loop_list:
                        df_result.loc[time_info, k] = 3
                        
abnormal_value(result_loopA, all_loop_df, ["A"])
abnormal_value(result_loopB, all_loop_df, ["B"])
abnormal_value(result_loopC, all_loop_df, ["C"])
abnormal_value(result_loopD, all_loop_df, ["D"])
abnormal_value(result_loopE, all_loop_df, ["E"])
abnormal_value(result_loopF, all_loop_df, ["F"])
abnormal_value(result_loopG_1, all_loop_df, ["G-1"])
abnormal_value(result_loopG_2, all_loop_df, ["G-2"])
abnormal_value(result_loopG_3, all_loop_df, ["G-3"])

#%% (8) : 결과를 csv 파일로 저장
all_loop_df.to_csv('C:/Users/82107/OneDrive/BIST/5/all_loop_df(min).csv')