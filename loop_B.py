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

#%%
#하이퍼 파라미터 설정하기
#hp_B_Res 랑 히트맵 이름 바꾸기, 0 이상 비율 되는 것들 추가하기

hp_loop = "B" 
hp_heatmap1_time = 0  # print로 출력시 사용
hp_heatmap2_time= 10
hp_heatmap2_timeall = 0 # timeall = 전수조사용
hp_heatmap3_time = 0
hp_heatmap4_time = 15
hp_heatmap4_timeall = 0

hp_temp_diff = 1 
hp_Res = 5 
hp_frequency_max = 98 
hp_frequency_efficiency = 60 
hp_frequency_off = 0 



# 글꼴 경로 설정
font_path = r"C:\Users\Lenovo\OneDrive - 성균관대학교\바탕 화면\Code\NANUMSQUAREB.TTF"
font_prop = FontProperties(fname=font_path)


# 자주 사용되는 코드 함수화

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
    
    
# (함수 C): 날짜 카운팅하는 열 'calendar' 추가하는 함수
def new_column_calendar(df):
    calendar = [] # 날짜 카운팅 리스트
    date_count = 1
    calendar.append(date_count)
    calendar.append(date_count)
    for i in range(2, df.shape[0]):
        if df.iloc[i-1, 0] != df.iloc[i, 0]:
            date_count += 1
        calendar.append(date_count)
    df["calendar"] = calendar
    return df


# (함수 D): 적응기간 삭제 함수
def adapt_delete(df):
    adapt = [] # 적응 기간에 속하는 행의 index 모은 리스트
    date_count = 1
    time_count= 0
    i = 1
    while i < df.shape[0]:
        time_count += 1
        adapt.append(i)
        if (time_count >= 3) or (df.loc[i, "calendar"] != df.loc[i+1, "calendar"]):
            date_count += 1
            restart = df[df["calendar"] == date_count].index.min()
            i = restart
            time_count = 0
            continue
        i+=1
    df_adapt = df.copy()
    df_adapt = df_adapt.drop(adapt)
    df_adapt.reset_index(drop=True, inplace=True)
    return df_adapt


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
def min_to_hour(df):
    df = df.sort_values(by=["Date", df.columns[0]], ascending=[True,True])
    hours = []
    for i in range(df.shape[0]):
        time = df.iloc[i, 0]
        hour = int(time.split(":")[0])
        hours.append(hour)
    df.insert(0, "Hours", hours)
    df_grouped = pd.DataFrame()
    df_grouped = df.groupby(['Date', 'Hours'])['Duration'].sum().reset_index()
    return df_grouped



#%% (2): 데이터 불러오기

df_4 = pd.read_csv(r"C:\Users\Lenovo\OneDrive - 성균관대학교\바탕 화면\Code\df_tower_on(07.31).csv", index_col=0, encoding='CP949')

# 해당 루프 이상거동 진단에 필요한 변수들 & 라벨링된 숫자들
# 0: 날짜, 1: 시간, 
Selected_Data_4 = df_4.iloc[:, [0, 1, 6, 7, 12, 14, 15, 16, 91]]

Selected_Data_4.loc[:, '6'] = pd.to_numeric(Selected_Data_4.loc[:, '6'], errors='coerce')
Selected_Data_4.loc[:, '7'] = pd.to_numeric(Selected_Data_4.loc[:, '7'], errors='coerce')

# 숫자형으로 변환한 후에 차이값 계산하여 새로운 열 추가
dt_list = []
dt_list = Selected_Data_4.loc[:, '7'] - Selected_Data_4.loc[:, '6']
Selected_Data_4.insert(4, "4", dt_list)

#%%

# 날짜, 시간 열 다듬기 
Selected_Data_4_copy = Selected_Data_4.copy()
Selected_Data_4_copy.loc[1:,'4'] = pd.to_numeric(Selected_Data_4_copy.loc[1:,'4'], errors='coerce')
Selected_Data_4_copy.loc[0, '0'] = 'Date'
Selected_Data_4_copy.loc[0, '1'] = 'Time'
Selected_Data_4_copy.loc[0, '6'] = '공급 온도'
Selected_Data_4_copy.loc[0, '7'] = '환수 온도'
Selected_Data_4_copy.loc[0, '4'] = '공급/환수 온도'

# 날짜 형식 바꾸기
# change_date_format(Selected_Data_4_copy, '0')



#%% (3): 히트맵 그리는 함수 미리 넣어두기

# <<히트맵 그리는 과정 총 4단계>>

# EX) 히트맵 그리기 위한 데이터프레임의 이름은 "df_heatmap", 기존 데이터프레임은 "DF"일 때


# [히트맵 그리기 1단계]: 1시간 단위의 히트맵 전용 데이터프레임 df_heatmap 만들기
map_dates = sorted(list(set(Selected_Data_4_copy.iloc[1:, 0])))  # 원본 데이터프레임을 가공한 DF의 'Date' 열에서 겹치지 않는 날짜만 추출
map_hour = [str(hour).zfill(2) for hour in range(0, 24)]  # 00시~23시까지 시간 단위 'Hour' 열 만들기
df_heatmap = pd.DataFrame(index=map_dates, columns=map_hour)  # 히트맵


# [히트맵 그리기 2단계]: 기존 데이터프레임 df에 데이터가 있으면, df_heatmap에 무조건 0으로 표시, 없으면 -10 표시
def data_exist_makr_0(df_heatmap, df): # df_heatmap : 히트맵을 그리기 위한 데이터프레임, df : 기존 데이터프레임
    for i in range(1, df.shape[0]):    
        date = df.iloc[i, 0] # df의 0번째 열이 날짜 열이어야 함
        time = df.iloc[i, 1] # df의 1번째 열이 시간 열이어야 함
        hours_minutes = time.split(':') # 시간과 분 분리
        hour = hours_minutes[0] # 시간만 뽑아내기
        print(date, hour)
        if np.isnan(df_heatmap.loc[date, hour]) == 1: # 기존 데이터프레임 df에 데이터가 존재하고, 해당 시간대가 df_heatmap에 nan이라고 표시되어 있다면,
            df_heatmap.loc[date, hour] = 0 # df_heatmap에 우선 0으로 표시하기
    return df_heatmap
            
def data_nonexist_mark_minus10(df_heatmap):
    df_heatmap = df_heatmap.fillna(-10)
    return df_heatmap
    
df_heatmap = data_exist_makr_0(df_heatmap, Selected_Data_4_copy)
df_heatmap = data_nonexist_mark_minus10(df_heatmap)


# [히트맵 그리기 3단계]: 이상 거동 히트맵 그리기
# (각자 dt_heatmap에 유효한 값 채우기 ; 1시간당 이상거동 유형 *번이 *분 발생하는지)


# [히트맵 그리기 4단계]: 이상 거동 히트맵 그리기
# (라이브러리 불러오기)
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib.font_manager import FontProperties

# (글꼴 경로 설정)
font_path = r"C:\Users\Lenovo\OneDrive - 성균관대학교\바탕 화면\Code\NANUMSQUAREB.TTF" 
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


# (초록색 히트맵 그리는 함수)
def energy_heat_map(df_heatmap, df_name): # df : 히트맵 전용 데이터프레임, df_name : 히트맵의 제목
    plt.figure(figsize=(8, 10))
    sns.heatmap(df_heatmap, cmap='Greens', linewidths=0.5, linecolor="lightgray", annot=False, vmin=-10, vmax=60)
    plt.xlabel("Hour", fontsize=10, fontproperties=font_prop)
    plt.ylabel("Date", fontsize=10, fontproperties=font_prop)
    plt.title(df_name, fontsize=10, fontproperties=font_prop)
    plt.xticks(rotation=0)
    plt.show()

    
    
#%% (4): 적응기간 초반 3분 제거

data_4 = Selected_Data_4_copy.copy()
data_4 = data_4.reset_index(drop=True)

# 날짜 카운팅하는 열 'calendar' 추가
data_4 = new_column_calendar(data_4)

# 적응기간 삭제
data_4_adapt = adapt_delete(data_4)


#%% (5): 이상거동 유형1 진단 과정

# 1-1) 유형1 - 데이터 준비 단계
sample1 = data_4_adapt.copy()

# 열 이름 변경
col = sample1.iloc[0, :]
sample1.columns = col
sample1 = sample1.iloc[1:, :]
sample1.reset_index(drop=True, inplace=True)

# 냉각수 온도차, 냉각수 펌프 차압 설정값 각각 기울기 구하고 sample1의 열에 추가하기
# 날짜 
calendar = sample1.iloc[:, -1]
#%%
# 냉각수 온도차 => DTI
valve =  sample1.iloc[:, 4]
valve_incline = []
valve_incline.append(float(0))

# 냉각수 펌프 차압 설정값 => SPI
set_point =  sample1.iloc[:, 6]
set_point_incline = []
set_point_incline.append(float(0))

# 기울기(DTI, SPI) 구하고 sample1의 열에 추가하기
for i in range(1, len(valve)):
    if calendar[i] != calendar[i-1]:
        valve_incline.append(0)
        set_point_incline.append(0)
        continue
    DTI = float(valve[i]) - float(valve[i-1])
    valve_incline.append(DTI)
    spi = float(set_point[i]) - float(set_point[i-1])
    set_point_incline.append(spi)

sample1.insert(2, "DTI", valve_incline)
sample1.insert(3, "SPI", set_point_incline)
#%%

# 이상거동1 진단을 위해 필요한 변수들만 남긴 sample1_1 만들기 
sample1_1 = sample1.copy()
sample1_1 = sample1_1.drop(sample1_1.columns[4:-1], axis=1)
#%%


# 5-2) 유형1 - 냉각수 온도차의 기울기과 셋포인트 기울기 두 변수간의 correlation 분석
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 기존 데이터 - SPI & DTI간의 correlation
df_corr = pd.DataFrame()
df_corr = sample1_1.copy().iloc[:, 2:4]
df_corr.columns = ['SPI', 'DTI']

corr_matrix = df_corr.corr()

sns.heatmap(corr_matrix, annot=True, vmin=0, vmax=1)
plt.show()

# 변형 데이터 - SPI & DTI(delayed by 1m)간의 correlation
valve = []
valve.append(0)
valve.extend(sample1_1.copy().iloc[:, 3])
valve = valve[:-1]

df_corr2 = pd.DataFrame()
df_corr2["SPI"] = sample1_1.copy().iloc[:, 2]
df_corr2["DTI(delayed by 1 m)"] = valve

corr_matrix2 = df_corr2.corr()

sns.heatmap(corr_matrix2, annot=True, vmin=0, vmax=1)
plt.show()


# 5-3) 유형1 - DTI 음수/양수인 경우 범위를 인덱스로 모으기 => idx_neg3, idx_post3

# DTI 절대값이 1보다 큰 경우(= 냉각수 온도차에 변화가 생긴 경우)의 인덱스 모으기

idx_neg = []  # 음수의 인덱스를 저장할 리스트
idx_post = []  # 양수의 인덱스를 저장할 리스트

for idx, value in sample1_1.iloc[:, 2].items():
    if (value > hp_temp_diff or value < hp_temp_diff * (-1)):
        if value < 0:
            idx_neg.append(idx)
        else:
            idx_post.append(idx)
#%%
# 연속인 인덱스끼리 묶기 => idx_neg2, idx_post2
def group(numbers):
    result = []
    current_group = [numbers[0]]
    
    for i in range(1, len(numbers)):
        diff = numbers[i] - numbers[i-1]
        if diff <= 1:
            current_group.append(numbers[i])
        else:
            result.append(current_group)
            current_group = [numbers[i]]

    result.append(current_group)
    return result
idx_neg2 = group(idx_neg)
idx_post2 = group(idx_post)
#%%
# 앞 뒤로 1분씩 추가하기 => idx_neg3, idx_post3
idx_neg3 = []
idx_post3 = []

max_size = sample1_1.shape[0]

for group in idx_neg2:
    min_value = min(group)
    max_value = max(group)
    if max_value + 1 >= max_size:
        continue
    new_group = [min_value - 1] + group + [max_value + 1] + [max_value + 2]
    idx_neg3.append(new_group)

for group in idx_post2:
    min_value = min(group)
    max_value = max(group)
    if max_value + 1 >= max_size:
        continue
    new_group = [min_value - 1] + group + [max_value + 1] + [max_value + 2]
    idx_post3.append(new_group)

# 냉각수 온도차 기울기가 양수/음수일 때 셋포인트 기울기의 절대값이 0, 0.01인 경우 찾기
#%%
# DTI<-1 일 때 SPI가 -0.1보다 작은 경우가 없을 때 이상거동으로 분류 => idx_strange_neg
idx_strange_neg = []

for idx in idx_neg3:
    print(idx)
    SPI = sample1_1.iloc[idx, 3]
    if (SPI < -0.1).sum() == 0:
        idx_strange_neg.append(idx)
        #%%
# DTI>1 일 때 SPI가 0.1보다 큰 경우가 없을 때 이상거동으로 분류 => idx_strange_post
idx_strange_post = []

for idx in idx_post3:
    SPI = sample1_1.iloc[idx, 3]
    if (SPI > 0.1).sum() == 0:
        idx_strange_post.append(idx)
    #%%
# 이상거동일 때 date, time, SPI, DTI 모두 정리하기 => result_neg, result_post
result_neg = pd.DataFrame()
result_post = pd.DataFrame()

for idx in range(len(idx_strange_neg)):
    index = idx_strange_neg[idx]
    #print(result_neg, sample1_1.iloc[index], end = '///')
    result_neg = pd.concat([result_neg, sample1_1.iloc[index]], axis=0)
result_neg = result_neg.drop_duplicates()

for idx in range(len(idx_strange_post)):
    index = idx_strange_post[idx]
    result_post = pd.concat([result_post, sample1_1.iloc[index]], axis=0)
result_post = result_post.drop_duplicates()


# 5-4) 유형1 - 나타난 시간대를 표시한 데이터프레임 만들기; merged_df

# DTI 음수일 때 이상거동인 경우 정리하기
count = 0
organ_neg = []

for i in range(0, result_neg.shape[0]-1):
    date = result_neg.iloc[i, 0]
    hour = result_neg.iloc[i, 1].split(':')[0]
    
    next_date = result_neg.iloc[i+1, 0]
    next_hour = result_neg.iloc[i+1, 1].split(':')[0]
    
    count += 1
    if (date != next_date) or (hour != next_hour):
        organ_neg.append([date, hour, count])
        count = 0

df_neg = pd.DataFrame(organ_neg, columns=["Date", "Hour", "Duration"])

# DTI 양수일 때 이상거동인 경우 정리하기
count = 0
organ_post = []

for i in range(0, result_post.shape[0]-1):
    date = result_post.iloc[i, 0]
    hour = result_post.iloc[i, 1].split(':')[0]
    
    next_date = result_post.iloc[i+1, 0]
    next_hour = result_post.iloc[i+1, 1].split(':')[0]
    
    count += 1
    if (date != next_date) or (hour != next_hour):
        organ_post.append([date, hour, count])
        count = 0

df_post = pd.DataFrame(organ_neg, columns=["Date", "Hour", "Duration"])
#%%
# "Date"와 "Hour"을 기준으로 두 데이터프레임을 합치기
merged_df = pd.merge(df_neg, df_post, on=['Date', 'Hour'], suffixes=('_df1', '_df2'))

# "Duration" 열의 요소를 합치기
merged_df['Duration'] = merged_df['Duration_df1'] + merged_df['Duration_df2']

# 불필요한 열 삭제
merged_df.drop(columns=['Duration_df1', 'Duration_df2'], inplace=True)

# 5-5) 유형1 - 히트맵 그리기 ; hour_map_1
# <<히트맵 그리는 과정 총 3단계>>

# [히트맵 그리기 1단계]: 1시간 단위의 히트맵 전용 데이터프레임 df_heatmap 만들기
hour_map_1 = pd.DataFrame(index=map_dates, columns=map_hour)  # 히트맵

# [히트맵 그리기 2단계]: 기존 데이터프레임 df에 데이터가 있으면, df_heatmap에 무조건 0으로 표시, 없으면 -10 표시
hour_map_1 = data_exist_makr_0(hour_map_1, sample1)
hour_map_1 = data_nonexist_mark_minus10(hour_map_1)

# [히트맵 그리기 3단계]: 이상 거동 히트맵 그리기
# (각자 dt_heatmap에 유효한 값 채우기 ; 1시간당 이상거동 유형 *번이 *분 발생하는지)
for i in range(0, df_neg.shape[0]):
    date = merged_df.iloc[i, 0]
    time = merged_df.iloc[i, 1]
    value = merged_df.iloc[i, 2]
    hour_map_1.loc[date, time] = value

# [히트맵 그리기 4단계]: 이상 거동 히트맵 그리기
abnormal_heat_map(hour_map_1, f"loop{hp_loop} #type 1 (Incidence in 1h >= {hp_heatmap1_time}m) <heat-map>")

# 5-6) 유형1 - 비율 조사
# 전체 데이터 ; 44615분
sum_of_time = int(sample1.shape[0])
sum_of_abnormal_1 = int(merged_df.iloc[:, 2].sum())
print(f"loop{hp_loop} #type 1 (Incidence in 1h >= {hp_heatmap1_time}m) <Incidence Rate>: " + str(round((sum_of_abnormal_1/sum_of_time)*100, 2)) + "%")


#%%




#%% (6): 이상거동 유형2~4 진단 공통 과정 
# 6-1) 설정값과의 오차 수준 판별하기
sample4 = data_4_adapt.copy()

# 열 이름 변경
col = sample4.iloc[0, :]
sample4.columns = col
sample4 = sample1.iloc[1:, :]
sample4.reset_index(drop=True, inplace=True)

# 차압이 셋 포인트에 맞춰지지 않는 정도 & 맞춰지지 않는 시간대 찾아내기 위해서,
# 분단위 시계열을 열로 가지고 날짜를 인덱스로 하는 times_4라는 새로운 데이터프레임 만들기
times_4 = minute_abnormal(df_4, sample4)

# 냉각수 펌프 차압 설정값과 차압의 차이에 대한 열 새로 추가하기
new_col5 = []
for i in range(0, sample4.shape[0]):
    val = float(sample4.iloc[i, 9]) - float(sample4.iloc[i, 8])
    new_col5.append(val)
sample4.insert(9, "오차", new_col5)


#%%
# times_4에 1분 단위로 이상거동 나타났는지, 정도는 어떠한지 표시
i = 1
while i < sample4.shape[0]-1:
    date1 = sample4.iloc[i-1, 0]
    date2 = sample4.iloc[i,0]
    time = sample4.iloc[i-1,1]
    condition1 = sample4.iloc[i-1, 9] > hp_Res
    condition3 = sample4.iloc[i-1, 9] < hp_Res * (-1)
    
    code = 404 # 오류, 안 잡힌 경우
    if date1 != date2: code = 0 # 날짜가 바뀌면 마지막 시간대는 0으로 표시

    elif condition3: code = -1

    elif condition1: code = 1
    else: code = 0
    times_4.loc[date1, time] = code
    i+=1

# Check if any element is equal to 404 in the DataFrame
has_negative_one = any(times_4.values.flatten() == 404)
if has_negative_one:
    print("DataFrame contains elements equal to 404")
else:
    print("DataFrame does not contain elements equal to 404")

# 개수 및 비율 출력

# 값이 0인 것의 개수
count_0 = (times_4.values == 0).sum()
# 값이 1인 것의 개수
count_1 = (times_4.values == 1).sum()
# 값이 2인 것의 개수
count_2 = (times_4.values == 2).sum()
# 값이 -1인 것의 개수
count_minus_1 = (times_4.values == -1).sum()
# 값이 -2인 것의 개수
count_minus_2 = (times_4.values == -2).sum()

# 전체 데이터 개수
total_count = count_0 + count_1 + count_2 + count_minus_1 + count_minus_2

# 각 값의 비율 계산
ratio_0 = count_0 / total_count * 100
ratio_1 = count_1 / total_count * 100
ratio_2 = count_2 / total_count * 100
ratio_minus_1 = count_minus_1 / total_count * 100
ratio_minus_2 = count_minus_2 / total_count * 100

# 결과 출력
print("Ratio of 0: " + str(round(ratio_0, 2)) + "%")
print("Ratio of 1: " + str(round(ratio_1, 2)) + "%")
print("Ratio of 2: " + str(round(ratio_2, 2)) + "%")
print("Ratio of -1: " + str(round(ratio_minus_1, 2)) + "%")
print("Ratio of -2: " + str(round(ratio_minus_2, 2)) + "%")

#%%
# 6-2) 매일 매분마다 주파수 수준 판별 => frequency

# times_sorted_str을 열로 가지고 Dates_cleaned를 인덱스로 하는 frequency라는 새로운 데이터프레임 만들기
frequency = minute_abnormal(df_4, sample4)


i = 1
while i < sample4.shape[0]-1:
    date = sample4.iloc[i-1, 0]
    time = sample4.iloc[i-1,1]
    condition4 = float(sample4.iloc[i-1, -3]) > hp_frequency_max
    condition3 = float(sample4.iloc[i-1, -3]) <= hp_frequency_max and float(sample4.iloc[i-1, -3]) > hp_frequency_efficiency
    condition2 = float(sample4.iloc[i-1, -3]) <= hp_frequency_efficiency and float(sample4.iloc[i-1, -3]) > hp_frequency_off 
    condition1 = float(sample4.iloc[i-1, -3]) <= hp_frequency_off 

    code = 404 # 오류, 안 잡힌 경우
    if condition4: code = 4 # 풀 가동 & 비효율에 해당
    elif condition3: code = 3 # 덜 가동 & 비효율에 해당
    elif condition2: code = 2 # 덜 가동 & 비효율에 해당 x
    elif condition1: code = 1 # 장치 off
    frequency.loc[date, time] = code
    
    if code == 404: print(date, time)
    i+=1


# Check if any element is equal to 404 in the DataFrame
has_404_one = any(frequency.values.flatten() == 404)

if has_404_one:
    print("DataFrame contains elements equal to 404")
else:
    print("DataFrame does not contain elements equal to 404")
    
# 개수 및 비율 출력
# 값이 1인 것의 개수
count_1 = (frequency.values == 1).sum()
# 값이 2인 것의 개수
count_2 = (frequency.values == 2).sum()
# 값이 -1인 것의 개수
count_3 = (frequency.values == 3).sum()
# 값이 -2인 것의 개수
count_4 = (frequency.values == 4).sum()

# 전체 데이터 개수
total_count = count_1 + count_2 + count_3 + count_4

# 각 값의 비율 계산
ratio_1 = count_1 / total_count * 100
ratio_2 = count_2 / total_count * 100
ratio_3 = count_3 / total_count * 100
ratio_4 = count_4 / total_count * 100

# 결과 출력
print("Ratio of 1: " + str(round(ratio_1, 2)) + "%")
print("Ratio of 2: " + str(round(ratio_2, 2)) + "%")
print("Ratio of 3: " + str(round(ratio_3, 2)) + "%")
print("Ratio of 4: " + str(round(ratio_4, 2)) + "%")    
#%%
# 6-3) 이상거동 유형화

# miss_code = 0 : 제어변수가 셋포인트에 잘 맞춰짐, 문제 없음 => 0
# miss_code = -1, -2 & freq_code = 2: 제어변수가 셋포인트보다 높게 유지되지만, 냉각 타워의 비효율적인 제어로 인해 발생한 문제는 아님 => 유형 6
# miss_code = -1, -2 & freq_code = 3, 4 : 비효율적인 이상거동 => 유형 2
# miss_code = 1, 2 & freq_code = 4 : 장치 용량 부족으로 제어가 안 되는 이상거동 => 유형 3 
# miss_code = 1, 2 & freq_code = 2, 3 : Controller의 문제로 제어가 안 되는 이상거동 => 유형 4
# freq_code = 1 : #1 냉각수 펌프 인버터 장치가 꺼져서 제어가 안 되는 이상거동 => 유형 5


#problems = pd.DataFrame(index=map_dates, columns=times_sorted_str)
problems = minute_abnormal(df_4, sample4)
i=1
while i < sample4.shape[0]-1:
    date = sample4.iloc[i-1, 0]
    time = sample4.iloc[i-1,1]
    miss_code = times_4.loc[date, time] # miss_code가 1 또는 2면 셋 포인트에 맞춰지고 있지 않는 현상
    freq_code = frequency.loc[date, time] # freq_code가 2면 냉각수 펌프 인버터 풀 가동, 1이면 덜 가동
    code = 404
    print(miss_code, freq_code)
    
    if miss_code == 0: # 정상 
        code = 0
    elif (miss_code == -1 or miss_code == -2) and freq_code == 2: # 과한 제어이나 비효율로 분류되지는 않음
        code = 6
    elif (miss_code == -1 or miss_code == -2) and (freq_code == 3 or freq_code == 4): # 과한 제어, 비효율 제어
        code = 2
    else:
        if freq_code == 4:
            code = 3 # 불충분한 제어, 장치 용량 부족
        elif freq_code == 2 or freq_code == 3:
            code = 4 # 불충분한 제어, controller/actuator의 문제
        elif freq_code == 1:
            code = 5 # 불충분한 제어, 장치 꺼짐
    problems.loc[date, time] = code
    i += 1
    
# Check if any element is equal to 404 in the DataFrame
has_404_one = any(problems.values.flatten() == 404)

if has_404_one:
    print("DataFrame contains elements equal to 404")
else:
    print("DataFrame does not contain elements equal to 404")
    
    

#%% (8): 이상거동 유형 2

# 8-1) 유형2 - 이상거동이 나타난 시간대 & 연달아 나타난 지속 시간 표시하는 데프 만들기 ; df_results_2
columns = ["Date", "Start_time", "End_time", "Duration", "Code"]
df_results_2 = pd.DataFrame(abnormal_df(2), columns=columns)
df_results_2.set_index("Date", inplace=True)

# 8-2) 유형2 - 이상거동이 나타난 시간대를 "날짜 & 시간" 단위로 나타내기 ; df_2_grouped
df_2_grouped = min_to_hour(df_results_2)

# 맨 첫 번째 행만 남기고 나머지 행 삭제
df_2_grouped.drop_duplicates(subset=['Date', 'Hours'], keep='first', inplace=True)

# 히트맵에 10 이상인 경우만 나타내기
df_2_grouped.loc[df_2_grouped['Duration'] < hp_heatmap2_time, 'Duration'] = 0

# 8-3) 유형2 - 히트맵 그리기 ; hour_map_2

# <<히트맵 그리는 과정 총 4단계>>

# [히트맵 그리기 1단계]: 1시간 단위의 히트맵 전용 데이터프레임 df_heatmap 만들기
hour_map_2 = pd.DataFrame()
hour_map_2 = pd.DataFrame(index=map_dates, columns=map_hour)  # 히트맵

# [히트맵 그리기 2단계]: 기존 데이터프레임 df에 데이터가 있으면, df_heatmap에 무조건 0으로 표시, 없으면 -10 표시
hour_map_2 = data_exist_makr_0(hour_map_2, sample1)
hour_map_2 = data_nonexist_mark_minus10(hour_map_2)

# [히트맵 그리기 3단계]: 이상 거동 히트맵 그리기
# (각자 dt_heatmap에 유효한 값 채우기 ; 1시간당 이상거동 유형 *번이 *분 발생하는지)
for i in range(0, df_2_grouped.shape[0]):
    date = str(df_2_grouped.iloc[i, 0])
    time = "{:02d}".format(df_2_grouped.iloc[i, 1])
    value = df_2_grouped.iloc[i, 2]
    hour_map_2.loc[date, time] = value

# [히트맵 그리기 4단계]: 이상 거동 히트맵 그리기
abnormal_heat_map(hour_map_2, f"loop{hp_loop} #type 2 (Incidence in 1h >= {hp_heatmap2_time}m) <heat-map>")


# 8-4) 유형2 - 비율 조사
# 전체 데이터 ; 44615분
sum_of_time = int(sample1.shape[0])
sum_of_abnormal_2 = int(df_2_grouped.iloc[:, 2].sum())
print(f"loop{hp_loop} #type 2 (Incidence in 1h >= {hp_heatmap2_time}m) <Incidence Rate>: " + str(round((sum_of_abnormal_2/sum_of_time)*100, 2)) + "%")

#%% 전수조사용 히트맵 type2
columns = ["Date", "Start_time", "End_time", "Duration", "Code"]
df_results_2 = pd.DataFrame(abnormal_df(2), columns=columns)
df_results_2.set_index("Date", inplace=True)

# 8-2) 유형2 - 이상거동이 나타난 시간대를 "날짜 & 시간" 단위로 나타내기 ; df_2_grouped
df_2_grouped = min_to_hour(df_results_2)

# 맨 첫 번째 행만 남기고 나머지 행 삭제
df_2_grouped.drop_duplicates(subset=['Date', 'Hours'], keep='first', inplace=True)

df_2_grouped.loc[df_2_grouped['Duration'] < hp_heatmap2_timeall, 'Duration'] = 0

# <<히트맵 그리는 과정 총 4단계>>

# [히트맵 그리기 1단계]: 1시간 단위의 히트맵 전용 데이터프레임 df_heatmap 만들기
hour_map_2 = pd.DataFrame()
hour_map_2 = pd.DataFrame(index=map_dates, columns=map_hour)  # 히트맵

# [히트맵 그리기 2단계]: 기존 데이터프레임 df에 데이터가 있으면, df_heatmap에 무조건 0으로 표시, 없으면 -10 표시
hour_map_2 = data_exist_makr_0(hour_map_2, sample1)
hour_map_2 = data_nonexist_mark_minus10(hour_map_2)

# [히트맵 그리기 3단계]: 이상 거동 히트맵 그리기
# (각자 dt_heatmap에 유효한 값 채우기 ; 1시간당 이상거동 유형 *번이 *분 발생하는지)
for i in range(0, df_2_grouped.shape[0]):
    date = str(df_2_grouped.iloc[i, 0])
    time = "{:02d}".format(df_2_grouped.iloc[i, 1])
    value = df_2_grouped.iloc[i, 2]
    hour_map_2.loc[date, time] = value

# [히트맵 그리기 4단계]: 이상 거동 히트맵 그리기
abnormal_heat_map(hour_map_2, f"loop{hp_loop} #type 2 (Incidence in 1h >= {hp_heatmap2_timeall}m) <heat-map>")


# 8-4) 유형2 - 비율 조사
# 전체 데이터 ; 44615분
sum_of_time = int(sample1.shape[0])
sum_of_abnormal_2 = int(df_2_grouped.iloc[:, 2].sum())
print(f"loop{hp_loop} #type 2 (Incidence in 1h >= {hp_heatmap2_timeall}m) <Incidence Rate>: " + str(round((sum_of_abnormal_2/sum_of_time)*100, 2)) + "%")


#%% (9): 이상거동 유형 3

# 9-1) 유형3 - 이상거동이 나타난 시간대 & 연달아 나타난 지속 시간 표시하는 데프 만들기 ; df_results_3
df_results_3 = pd.DataFrame(abnormal_df(3), columns=columns)
df_results_3.set_index("Date", inplace=True)


# 9-2) 유형3 - 이상거동이 나타난 시간대를 "날짜 & 시간" 단위로 나타내기 ; df_3_grouped
df_3_grouped = min_to_hour(df_results_3)

# 맨 첫 번째 행만 남기고 나머지 행 삭제
df_3_grouped.drop_duplicates(subset=['Date', 'Hours'], keep='first', inplace=True)

# 히트맵에 10 이상인 경우만 나타내기
df_3_grouped.loc[df_3_grouped['Duration'] < 0, 'Duration'] = 0


# 9-3) 유형3 - 히트맵 그리기 ; hour_map_3

# <<히트맵 그리는 과정 총 3단계>>

# [히트맵 그리기 1단계]: 1시간 단위의 히트맵 전용 데이터프레임 df_heatmap 만들기
hour_map_3 = pd.DataFrame()
hour_map_3 = pd.DataFrame(index=map_dates, columns=map_hour)  # 히트맵

# [히트맵 그리기 2단계]: 기존 데이터프레임 df에 데이터가 있으면, df_heatmap에 무조건 0으로 표시, 없으면 -10 표시
hour_map_3 = data_exist_makr_0(hour_map_3, sample1)
hour_map_3 = data_nonexist_mark_minus10(hour_map_3)

# [히트맵 그리기 3단계]: 이상 거동 히트맵 그리기
# (각자 dt_heatmap에 유효한 값 채우기 ; 1시간당 이상거동 유형 *번이 *분 발생하는지)
for i in range(0, df_3_grouped.shape[0]):
    date = str(df_3_grouped.iloc[i, 0])
    time = "{:02d}".format(df_3_grouped.iloc[i, 1])
    value = df_3_grouped.iloc[i, 2]
    hour_map_3.loc[date, time] = value

# [히트맵 그리기 4단계]: 이상 거동 히트맵 그리기
abnormal_heat_map(hour_map_2, f"loop{hp_loop} #type 3 (Incidence in 1h >= {hp_heatmap4_time}m) <heat-map>")



# 8-4) 유형3 - 비율 조사
# 전체 데이터 ; 44615분
sum_of_time = int(sample1.shape[0])
sum_of_abnormal_3 = int(df_3_grouped.iloc[:, 2].sum())
print(f"loop{hp_loop} #type 3 (Incidence in 1h >= {hp_heatmap3_time}m) <Incidence Rate>: " + str(round((sum_of_abnormal_3/sum_of_time)*100, 2)) + "%")




#%%#%% (10): 이상거동 유형 4

# 10-1) 유형4 - 이상거동이 나타난 시간대 & 연달아 나타난 지속 시간 표시하는 데프 만들기 ; df_results_4
df_results_4 = pd.DataFrame(abnormal_df(4), columns=columns)
df_results_4.set_index("Date", inplace=True)


# 10-2) 유형4 - 이상거동이 나타난 시간대를 "날짜 & 시간" 단위로 나타내기 ; df_4_grouped
df_4_grouped = min_to_hour(df_results_4)

# 맨 첫 번째 행만 남기고 나머지 행 삭제
df_4_grouped.drop_duplicates(subset=['Date', 'Hours'], keep='first', inplace=True)

# 히트맵에 10 이상인 경우만 나타내기
df_4_grouped.loc[df_4_grouped['Duration'] < hp_heatmap4_time, 'Duration'] = 0


# 10-3) 유형4 - 히트맵 그리기 ; hour_map_4

# <<히트맵 그리는 과정 총 3단계>>

# [히트맵 그리기 1단계]: 1시간 단위의 히트맵 전용 데이터프레임 df_heatmap 만들기
hour_map_4 = pd.DataFrame()
hour_map_4 = pd.DataFrame(index=map_dates, columns=map_hour)  # 히트맵


# [히트맵 그리기 2단계]: 기존 데이터프레임 df에 데이터가 있으면, df_heatmap에 무조건 0으로 표시, 없으면 -10 표시
hour_map_4 = data_exist_makr_0(hour_map_4, sample1)
hour_map_4 = data_nonexist_mark_minus10(hour_map_4)

# [히트맵 그리기 3단계]: 이상 거동 히트맵 그리기
# (각자 dt_heatmap에 유효한 값 채우기 ; 1시간당 이상거동 유형 *번이 *분 발생하는지)
for i in range(0, df_4_grouped.shape[0]):
    date = str(df_4_grouped.iloc[i, 0])
    time = "{:02d}".format(df_4_grouped.iloc[i, 1])
    value = df_4_grouped.iloc[i, 2]
    hour_map_4.loc[date, time] = value

# [히트맵 그리기 4단계]: 이상 거동 히트맵 그리기
abnormal_heat_map(hour_map_2, f"loop{hp_loop} #type 4 (Incidence in 1h >= {hp_heatmap4_time}m) <heat-map>")


# 10-4) 유형4 - 비율 조사
# 전체 데이터 ; 44615분
sum_of_time = int(sample1.shape[0])
sum_of_abnormal_4 = int(df_4_grouped.iloc[:, 2].sum())
print(f"loop{hp_loop} #type 4 (Incidence in 1h >= {hp_heatmap4_time}m) <Incidence Rate>: "+ str(round((sum_of_abnormal_4/sum_of_time)*100, 2)) + "%")




#%% 전수조사용 type4 히트맵
# 10-1) 유형4 - 이상거동이 나타난 시간대 & 연달아 나타난 지속 시간 표시하는 데프 만들기 ; df_results_4
df_results_4 = pd.DataFrame(abnormal_df(4), columns=columns)
df_results_4.set_index("Date", inplace=True)


# 10-2) 유형4 - 이상거동이 나타난 시간대를 "날짜 & 시간" 단위로 나타내기 ; df_4_grouped
df_4_grouped = min_to_hour(df_results_4)

# 맨 첫 번째 행만 남기고 나머지 행 삭제
df_4_grouped.drop_duplicates(subset=['Date', 'Hours'], keep='first', inplace=True)

df_4_grouped.loc[df_4_grouped['Duration'] < hp_heatmap4_timeall, 'Duration'] = 0


# 10-3) 유형4 - 히트맵 그리기 ; hour_map_4

# <<히트맵 그리는 과정 총 3단계>>

# [히트맵 그리기 1단계]: 1시간 단위의 히트맵 전용 데이터프레임 df_heatmap 만들기
hour_map_4 = pd.DataFrame()
hour_map_4 = pd.DataFrame(index=map_dates, columns=map_hour)  # 히트맵


# [히트맵 그리기 2단계]: 기존 데이터프레임 df에 데이터가 있으면, df_heatmap에 무조건 0으로 표시, 없으면 -10 표시
hour_map_4 = data_exist_makr_0(hour_map_4, sample1)
hour_map_4 = data_nonexist_mark_minus10(hour_map_4)

# [히트맵 그리기 3단계]: 이상 거동 히트맵 그리기
# (각자 dt_heatmap에 유효한 값 채우기 ; 1시간당 이상거동 유형 *번이 *분 발생하는지)
for i in range(0, df_4_grouped.shape[0]):
    date = str(df_4_grouped.iloc[i, 0])
    time = "{:02d}".format(df_4_grouped.iloc[i, 1])
    value = df_4_grouped.iloc[i, 2]
    hour_map_4.loc[date, time] = value

# [히트맵 그리기 4단계]: 이상 거동 히트맵 그리기
abnormal_heat_map(hour_map_2, f"loop{hp_loop} #type 4 (Incidence in 1h >= {hp_heatmap4_timeall}m) <heat-map>")


# 10-4) 유형4 - 비율 조사
# 전체 데이터 ; 44615분
sum_of_time = int(sample1.shape[0])
sum_of_abnormal_4 = int(df_4_grouped.iloc[:, 2].sum())
print(f"loop{hp_loop} #type 4 (Incidence in 1h >= {hp_heatmap4_timeall}m) <Incidence Rate>: "+ str(round((sum_of_abnormal_4/sum_of_time)*100, 2)) + "%")
