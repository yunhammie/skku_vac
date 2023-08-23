#%% (1): Basic Setting

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
from matplotlib.font_manager import FontProperties
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap


# 글꼴 경로 설정
font_path = "C:/Users/LAB_1/Desktop/글씨/나눔스퀘어/NanumFontSetup_OTF_SQUARE/NanumSquareB.otf"
font_prop = FontProperties(fname=font_path)

#%% (2): 데이터 불러오기                                                        df: 정윤코드에서 제거한 데이터 다시 가져오기
df = pd.read_csv("C:/Users/LAB_1/Desktop/HVAC 관련 코딩/코딩 후 데이터/시스템OFF 제거 + 외기냉방 제거 + 냉동기OFF 제거.csv", index_col=0, encoding='CP949')

#%% (3): 필요한 데이터 가공 및 선택                                              Selected_df: 선택된 데이터

df["Room1_공급/설정_dT"] = df.iloc[:,62] - df.iloc[:,61] 
df["Room2_공급/설정_dT"] = df.iloc[:,71] - df.iloc[:,70]
df["Room3_공급/설정_dT"] = df.iloc[:,80] - df.iloc[:,79]

Selected_df = df.iloc[:,[0,1,3,4,5,35,
                         61,62,99,65,66,
                         70,71,100,74,75,
                         79,80,101,83,84
                         ]]

#%% (4): 이상치 제거 및 확률밀도파악 함수  
def analyze_column(df, column_name, limit=3):
    # 데이터 전처리 - 이상치 제거 (Outlier removal)
    def z_score_Outlier_rm(column_data, limit):
        z_scores = stats.zscore(column_data)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < limit)
        return column_data[filtered_entries]

    # 선택한 열에 대해서 이상치 제거
    column_data = df[column_name].values
    column_data_cleaned = z_score_Outlier_rm(column_data, limit)

    # 평균과 표준편차 계산
    mean = np.mean(column_data_cleaned)
    std = np.std(column_data_cleaned)

    # 그래프 그리기
    plt.figure(figsize=(8, 6))
    x = np.linspace(np.min(column_data_cleaned), np.max(column_data_cleaned), 100)
    pdf = stats.norm.pdf(x, mean, std)
    plt.plot(x, pdf, color='blue', label='PDF')
    plt.hist(column_data_cleaned, bins=30, density=True, alpha=0.7, label='Histogram')
    plt.xlabel(column_name, fontproperties=fm.FontProperties(fname=font_path, size=12))  # Set x-axis label to column_name
    plt.ylabel('Density', fontproperties=fm.FontProperties(fname=font_path, size=12))
    plt.title('정규화 분포', fontproperties=fm.FontProperties(fname=font_path, size=16))
    plt.grid(True)
    plt.legend()
    

    # 그래프 출력
    plt.show()

    # 결과 출력
    print('----------------------------------------------------------------------------')
    print('Data of ',column_name)
    print('')
    print('Mean:', round(mean,3))
    print('Standard Deviation:', round(std,3))
    
    # PDF가 가장 높을 때의 x값 출력
    max_pdf_index = np.argmax(pdf)
    max_x = x[max_pdf_index]
    print('Max x:', max_x)

    # 시그마, 2시그마, 3시그마 값과 그때의 x값 출력
    sigma = std
    two_sigma = 2 * std
    three_sigma = 3 * std
    sigma_x = mean + sigma
    two_sigma_x = mean + two_sigma
    three_sigma_x = mean + three_sigma
    print('Sigma:', round(sigma, 3), ', Sigma x:', round(sigma_x, 3))
    print('2 Sigma:', round(two_sigma, 3), ', 2 Sigma x:', round(two_sigma_x, 3))
    print('3 Sigma:', round(three_sigma, 3), ', 3 Sigma x:', round(three_sigma_x, 3))
    print('----------------------------------------------------------------------------')

dt = pd.DataFrame(Selected_df)


analyze_column(dt, '냉수_공급/설정_dT', limit=3)
#Data_Frame이랑 Column_name만 지정해주면 됨


#%% (5): 분단위 거동분석                                                        Analyzed_Data : 거동분류  
### 본인들이 세운 기준을 여기다가 적용하면 됩니다.


Analyzed_Data = Selected_df.copy()

# 열 추가
Analyzed_Data['거동유형'] = None
Analyzed_Data['센서작동'] = None

# print("--------------------------------------------------------------")
# print("분단위로 데이터를 분석하는 구간입니다.")
# while True:
#     try:
#         Normal_Range = float(input("정상제어 범위를 입력해주세요(0~5°C):"))
#         if 0 <=  Normal_Range <= 5:
#             print(f'정상제어 범위는 (설정값) ± {Normal_Range}°C 입니다.')
#             break  
#         else:
#             print("입력값이 허용 범위 밖입니다. 다시 입력해주세요.")
#     except ValueError:
#         print("입력값이 숫자가 아닙니다. 다시 입력해주세요.")
# print("분석 중입니다. 기다려주세요.")


hp_1 = 3
hp_2 = 1.5

# 거동분류를 위한 조건 함수 정의
def classify_behavior(row):
    if row["냉수_공급/설정_dT"] > hp_1:
        return '냉동기 ON + 냉수 Setpoint x'
    elif row["냉수_공급/설정_dT"] < -hp_2:
        return '비효율적 냉동기가동'
    else:
        return '정상 제어'

def classify_sensor_status(row):
    if row["냉동기_1분간_전력사용량(kWh)"] == 0:
        return '오류 의심'
    else:
        return '정상'

# 거동분류 열 추가
Analyzed_Data["거동유형"] = Analyzed_Data.apply(classify_behavior, axis=1)

# 센서작동 열 추가
Analyzed_Data["센서작동"] = Analyzed_Data.apply(classify_sensor_status, axis=1)

#%% (6): 거동유형 별 수치 

def analyze_movement_type(dataframe, column_name): 

    # 해당 열의 값들의 개수를 계산
    value_counts = dataframe[column_name].value_counts()

    # 전체 행의 개수를 계산
    total_count = dataframe.shape[0]

    # 비중(백분율) 계산
    proportions = (value_counts / total_count) * 100

    # 결과를 데이터프레임으로 만들어 반환
    result_df = pd.DataFrame({
        '개수': value_counts,
        '비중(%)': proportions.round(2)
    }).sort_values(by='개수', ascending=False)

    # 결과를 표 형태로 정리하여 출력
    print("--------------------------------------------------------------")
    print(result_df.to_string(index=True))
    print("--------------------------------------------------------------")

# '거동유형' 열에 대해 각 내용들의 개수와 비중 계산 및 내림차순 정렬 후 출력

analyze_movement_type(Analyzed_Data, '거동유형')
analyze_movement_type(Analyzed_Data, '센서작동')

#%% (7): 헌팅판단                                                               Analyzed_Data : 거동분류 + 헌팅판단
Analyzed_Data['냉수_제어정도'] = None

def classify_cooling_control(row):
    dT_value = row["냉수_공급/설정_dT"]
    
    if dT_value < -1.5:
        return 'Ineffective'
    elif dT_value <= 3:  # 이 부분을 수정
        return 'Normal'
    elif dT_value > 3 and dT_value <= 6:
        return 'light' 
    elif dT_value > 6 and dT_value <= 9:
        return 'Moderate'
    elif dT_value > 9:
       return 'Severe'
        
Analyzed_Data["냉수_제어정도"] = Analyzed_Data.apply(classify_cooling_control, axis=1)



#%% (8): 시간단위 묶기                                                          Hourly_list: 시간별 그룹화
Analyzed_Data['hour'] = pd.to_datetime(Analyzed_Data['date'] + ' ' + Analyzed_Data['Time'], format='%Y-%m-%d %H:%M').dt.hour
Analyzed_Data['hour'] = Analyzed_Data['hour'].astype(str).str.zfill(2)
Hourly_list = []

for date_hour, group in Analyzed_Data.groupby(['date', 'hour']):
    date = group['date'].unique()[0]
    hour = group['hour'].astype(str).unique()[0]
    Hourly_list.append(group)
    
    
#%% (9): 시간단위 거동분석                                                      Hourly_Analyzed : 시간단위 분석 내용
Hourly_Analyzed = pd.DataFrame(columns=['date', 'hour',
                                        '냉수 Setpoint x 판단', '효율성 판단','Total',
                                        '냉동기 ON + 냉수 Setpoint x','비효율적 냉동기가동','정상 제어',
                                        '진단결과', '거동코드',
                                        '센서 오류', '센서 오류 판단'])


for group in Hourly_list:
    date = group['date'].unique()[0]
    hour = group['hour'].unique()[0]
    activities = group['거동유형'].tolist()
    activities2 = group['센서작동'].tolist()
    Abnormal_1 = 0
    Abnormal_2 = 0
    Abnormal_3 = 0

    Analyzed_Result = None
    count = 0  
    Activity_1 = 0
    Activity_2 = 0
    Activity_3 = 0
    Activity_4 = 0


    for activity in activities:
        if activity == '냉동기 ON + 냉수 Setpoint x':
            count += 1
            Activity_1 +=1
            if count >= 10:
                Abnormal_1 = 1
                Analyzed_Result = '냉동기 ON + 냉수 Setpoint x'
        else:
            count = 0
    for activity in activities:
        if activity == '비효율적 냉동기가동':
            count += 1
            Activity_2 +=1
            if count >= 5:
                Abnormal_2 = 1
                Analyzed_Result = '비효율적 냉동기가동'
        else:
            count = 0
    for activity in activities:
        if activity == '정상 제어':
            Activity_3 +=1            

    total = Abnormal_1 + Abnormal_2 
            

    for activity2 in activities2:
        if activity2 == '오류 의심':
            count += 1
            Activity_4 +=1
            if count >= 10:
                Abnormal_3 = '센서 오류'
        else:
            count = 0
            

    
    if total == 0:
        max_count = max(activities.count(activity) for activity in activities)
        Analyzed_Result = [activity for activity in set(activities) if activities.count(activity) == max_count][0]
    elif total == 2 :
        max_count = max(activities.count(activity) for activity in activities)
        Analyzed_Result = [activity for activity in set(activities) if activities.count(activity) == max_count][0]    
        
        
    Hourly_Analyzed = pd.concat([Hourly_Analyzed, pd.DataFrame({
        'date': [date],
        'hour': [hour],
        '냉수 Setpoint x 판단': [Abnormal_1], '효율성 판단': [Abnormal_2],'Total': [total],
        '냉동기 ON + 냉수 Setpoint x' : [Activity_1],'비효율적 냉동기가동' : [Activity_2],'정상 제어' : [Activity_3],
        '진단결과': [Analyzed_Result],
        '센서 오류' : [Activity_4],'센서 오류 판단': [Abnormal_3]
    })], ignore_index=True)


# 거동_code 매핑
activity_code_map = {
    '냉동기 ON + 냉수 Setpoint x': 1,
    '비효율적 냉동기가동': 2,
    '정상 제어': 3,
}
Hourly_Analyzed['거동코드'] = Hourly_Analyzed['진단결과'].map(activity_code_map)
numeric_columns = [ '냉수 Setpoint x 판단', '효율성 판단', 'Total',
                    '냉동기 ON + 냉수 Setpoint x', '비효율적 냉동기가동','정상 제어',
                    '센서 오류']
Hourly_Analyzed[numeric_columns] = Hourly_Analyzed[numeric_columns].apply(pd.to_numeric, errors='coerce')


#%% (10): 히트맵 양식을 위하여                                                   DF

df2 = pd.read_csv("C:/Users/LAB_1/Desktop/all-정윤.csv", index_col=0, encoding='CP949')

DF = pd.DataFrame(df2)

# 열 이름 바꾸기
new_column_names = DF.iloc[1]
DF = DF[2:]
DF.columns = new_column_names

# 데이터들 float 형태로 바꾸기
DF.iloc[:] = DF.iloc[:].astype(float)


DF['Times'] = DF.index
DF[['dates', 'times']] = DF['Times'].str.split(' ', expand=True)
DF.insert(0, 'Date', DF['dates'])
DF.insert(1, 'Time', DF['times'])
DF = DF.drop('Times', axis=1)
DF = DF.drop('dates', axis=1)
DF = DF.drop('times', axis=1)



def change_date_format(df, date_column_name):
    Date = df.loc[:, date_column_name]
    Date = Date[1:]
    dates = pd.to_datetime(Date, format='%d.%m.%Y').dt.strftime('%Y-%m-%d')
    dates = pd.concat([pd.Series(['dates']), dates], ignore_index=True)
    dates_list = list(dates)
    df.insert(0, 'date', dates_list)
    df.drop(date_column_name, axis = 1, inplace=True)
change_date_format(DF, 'Date')

# 행의 index를 정수로 초기화
DF.reset_index(drop=True, inplace=True)
# DF 안의 요소들 자료형을 숫자형으로 바꾸기


DF.at[0, 'date'] = '2022-06-01'

DF['hour'] = pd.to_datetime(DF['date'] + ' ' + DF['Time'], format='%Y-%m-%d %H:%M').dt.hour 
DF['hour'] = DF['hour'].astype(str).str.zfill(2)
DF['데이터 유무'] = 1

#%% (11): 유형별 데이터프레임 + 히트맵                                           Pivot_Data_1~7 : 유형별 데이터

## 전체 데이터 유무 반영 후 거동 피벗데이터 반영함수
def create_pivot_data(Raw_Data, Analyzed_df, value_column, Data_x_value, Data_o_value):
    all_dates = sorted(list(set(Raw_Data['date'])))
    all_hours = sorted(list(set(Raw_Data['hour'])))

    # 빈 피벗 데이터프레임 생성
    pivot_data = pd.DataFrame(index=all_dates, columns=all_hours, data = Data_x_value)

    # 데이터프레임의 데이터 유무에 따라 값 반영
    for date in all_dates:
        for hour in all_hours:
            condition = (Raw_Data['date'] == date) & (Raw_Data['hour'] == hour) & (Raw_Data['데이터 유무'] == 1)
            if condition.any():
                pivot_data.at[date, hour] = Data_o_value

    # Hourly_Analyzed 데이터프레임의 값 반영
    for index, row in Analyzed_df.iterrows():
        date = row['date']
        hour = row['hour']
        if date in all_dates and hour in all_hours:
            value = row[value_column]
            pivot_data.at[date, hour] = value

    return pivot_data


Pivot_DF_1 = create_pivot_data(DF, Hourly_Analyzed, '냉동기 ON + 냉수 Setpoint x', -10, 0)
Pivot_DF_2 = create_pivot_data(DF, Hourly_Analyzed, '비효율적 냉동기가동', -10, 0)
Pivot_DF_3 = create_pivot_data(DF, Hourly_Analyzed, '정상 제어', -10, 0)
Pivot_DF_4 = create_pivot_data(DF, Hourly_Analyzed, '센서 오류', -10, 0)


### 히트맵 고르기
def plot_heatmap(pivot_data, title=None, cmap='Blues'):
    
    plt.figure(figsize=(8, 10))
    max_value = pivot_data.values.max()

    # vmax에 최댓값에 10을 더한 값을 설정합니다.
    vmax_value = max_value + 10
    
    sns.heatmap(pivot_data, cmap=cmap, linewidths=0.5, linecolor="lightgray", annot=False, vmin=-10, vmax=vmax_value)
    
    # 컬러바의 눈금 간격을 5로 설정합니다.
    plt.title(title, fontproperties=font_prop)
    plt.xlabel("Hour", fontsize=10, fontproperties=font_prop)
    plt.ylabel("Date", fontsize=10, fontproperties=font_prop)
    plt.show()



plot_heatmap(Pivot_DF_1, title="냉동기 ON + 냉수 Setpoint x", cmap='YlOrRd')
plot_heatmap(Pivot_DF_2, title="비효율적 냉동기가동", cmap='Greens')
plot_heatmap(Pivot_DF_3, title="정상 제어", cmap='Blues')
plot_heatmap(Pivot_DF_4, title="센서 오류", cmap='YlOrRd')




#%% (12): 전체 데이터프레임 + 히트맵                                             Pivot_DF_All: 전체 데이터프레임
#####  요거만 하면 끝남

Pivot_DF_All = create_pivot_data(DF, Hourly_Analyzed, '거동코드', 0, 4)

# 0: 데이터 공란
# 1: 냉동기 ON + 냉수 Setpoint x
# 2: 비효율적 냉동기가동
# 3: 정상 제어
# 4: 전체시스템 OFF

color_map = ListedColormap(['white',
                           '#F7C7A7',
                           '#FFFF9F',
                           '#D7E7F5',
                           '#C7C7C7'])

plt.figure(figsize=(8, 10))
sns.heatmap(Pivot_DF_All, cmap=color_map, linewidths=0.5, linecolor="lightgray", annot=False, cbar=True)

plt.title("Calendar Heatmap - 전체 거동 유형", fontproperties=font_prop)
plt.xlabel("Hour", fontsize=10, fontproperties=font_prop)
plt.ylabel("Date", fontsize=10, fontproperties=font_prop)
plt.show()


#%% (13): 헌팅 그래프                                                           Hunting_Dataset : 헌팅파악 위한 데이터셋

# 이상치 제거 함수
def remove_outliers_sigma(df, column, sigma=5):
    mean = df[column].mean()
    std = df[column].std()
    lower_bound = mean - sigma * std
    upper_bound = mean + sigma * std
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    df.reset_index(drop=True, inplace=True)
    return df


# 이상치제거를 수행할 열 선택 후 적용
column_name = "냉동기_1분간_전력사용량(kWh)"
Hunting_Dataset = remove_outliers_sigma(Analyzed_Data, column_name)



# '냉수_제어정도' 기준으로 묶은 후 그래프 그리는 함수'
def plot_grouped_data(df, x_column, y_column):
    
    
    # 냉수_제어정도로 데이터 그룹화
    grouped = df.groupby('냉수_제어정도')

    # 그래프 생성
    plt.figure(figsize=(8, 6))

    # 각 그룹별로 그래프 그리기
    for name, group in grouped:
        x_data = group[x_column]
        y_data = group[y_column]
        plt.scatter(x_data, y_data, label=name, marker='o',s=15)

    
    plt.xlim(-0.4,0.6)
    plt.ylim(-20,30)
    # y=2와 같은 수평선 그리기
    plt.axhline(y=3, color='red', linestyle='dashed')
    plt.axhline(y=-1.5, color='red', linestyle='dashed')
    plt.xlabel(x_column, fontproperties=font_prop)
    plt.ylabel(y_column, fontproperties=font_prop)
    plt.title(f'헌팅지수', fontproperties=font_prop)
    plt.grid(True)
    plt.legend(prop=font_prop)
    plt.show()

# 데이터프레임, x축, y축 선택해서 그래프 출력
plot_grouped_data(Hunting_Dataset, '냉동기_1분간_전력사용량(kWh)', '냉수_공급/설정_dT')
analyze_movement_type(Hunting_Dataset, '냉수_제어정도')


#%% (14): 효율/비효율 수치

dt2 = pd.DataFrame(Analyzed_Data)
In_effective = dt2[dt2['거동유형'] == '비효율적 냉동기가동']
Effective = dt2[dt2['거동유형'] != '비효율적 냉동기가동']

effective_total = Effective['냉동기_1분간_전력사용량(kWh)'].sum()
ineffective_total = In_effective['냉동기_1분간_전력사용량(kWh)'].sum()
Total = effective_total + ineffective_total
effective_percentage = effective_total/Total*100
ineffective_percentage = ineffective_total/Total*100

print(f"Effective 데이터프레임 냉동기_1분간_전력사용량 총 합: {round(effective_total, 2)} kWh   {round(effective_percentage, 2)}%")
print(f"In-effective 데이터프레임 냉동기_1분간_전력사용량 총 합: {round(ineffective_total, 2)} kWh   {round(ineffective_percentage, 2)}%")

#%% 또 뭐하지......







