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
df = pd.read_csv("C:/Users/LAB_1/Desktop/HVAC 관련 코딩/코딩 후 데이터/외기냉방만 포함.csv", index_col=0, encoding='CP949')

#%% (3): 필요한 데이터 가공 및 선택                                              Selected_df: 선택된 데이터

df["Room1_공급/설정_dT"] = df.iloc[:,62] - df.iloc[:,61] 
df["Room2_공급/설정_dT"] = df.iloc[:,71] - df.iloc[:,70]
df["Room3_공급/설정_dT"] = df.iloc[:,80] - df.iloc[:,79]

Selected_df = df.iloc[:,[0,1,
                         91,92,93,97,98,44,
                         61,62,99,65,66,
                         70,71,100,74,75,
                         79,80,101,83,84
                         ]]


#%% 이상치 제거 및 확률밀도파악 함수   

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

## #Data_Frame이랑 Column_name만 지정해주면 됨
dt = pd.DataFrame(Selected_df)

analyze_column(dt, 'Room1_공급/설정_dT', limit=3)
analyze_column(dt, '1# VAV 댐퍼 개도율(%)', limit=3)
analyze_column(dt, '1# VAV 풍량(M3/h)', limit=3)


analyze_column(dt, 'Room2_공급/설정_dT', limit=3)
analyze_column(dt, '3# VAV 댐퍼 개도율(%)', limit=3)
analyze_column(dt, '3# VAV 풍량(M3/h)', limit=3)


analyze_column(dt, 'Room3_공급/설정_dT', limit=3)
analyze_column(dt, '5# VAV 댐퍼 개도율(%)', limit=3)
analyze_column(dt, '5# VAV 풍량(M3/h)', limit=3)


#%% (4): 분단위 거동분석                                                        Analyzed_Data : 거동분류  
### 본인들이 세운 기준을 여기다가 적용하면 됩니다.
Analyzed_Data = Selected_df.copy()


Analyzed_Data.reset_index(drop=True, inplace=True)

# 열 추가
Analyzed_Data['Room1 거동유형'] = None
Analyzed_Data['Room2 거동유형'] = None
Analyzed_Data['Room3 거동유형'] = None


# print("--------------------------------------------------------------")
# print("분단위로 데이터를 분석하는 구간입니다.")
# while True:
#     try:
#         # Ask the user for the acceptable range for normal control.
#         Normal_Range = float(input("정상제어 범위를 입력해주세요(0~5°C):"))
        
#         # Check if the entered value is within the range of 0 to 5.
#         if 0 <=  Normal_Range <= 5:
#             print(f'정상제어 범위는 (설정값) ± {Normal_Range}°C 입니다.')
#             break  # Exit the loop if the input is valid
#         else:
#             print("입력값이 허용 범위 밖입니다. 다시 입력해주세요.")
#     except ValueError:
#         print("입력값이 숫자가 아닙니다. 다시 입력해주세요.")
# print("분석 중입니다. 기다려주세요.")

hp_1 = 1

for i in range(len(Analyzed_Data)):
    if float(Analyzed_Data.loc[i, "Room1_공급/설정_dT"]) > hp_1 and float(Analyzed_Data.loc[i, "1# VAV 댐퍼 개도율(%)"]) < 100 :
        Analyzed_Data.loc[i, 'Room1 거동유형'] = '댐퍼개도율 부족'
    elif float(Analyzed_Data.loc[i, "Room1_공급/설정_dT"]) > hp_1 and float(Analyzed_Data.loc[i, "1# VAV 댐퍼 개도율(%)"]) >= 100 :
        Analyzed_Data.loc[i, 'Room1 거동유형'] = 'VAV풍량 부족'
    elif (float(Analyzed_Data.loc[i, "Room1_공급/설정_dT"]) >= -hp_1 and float(Analyzed_Data.loc[i, "Room1_공급/설정_dT"]) <= hp_1) or (float(Analyzed_Data.loc[i, "Room1_공급/설정_dT"]) < -hp_1 and float(Analyzed_Data.loc[i, "1# VAV 댐퍼 개도율(%)"]) < 50) :
        Analyzed_Data.loc[i, 'Room1 거동유형'] = '정상 제어'
    elif float(Analyzed_Data.loc[i, "Room1_공급/설정_dT"]) < -hp_1 and float(Analyzed_Data.loc[i, "1# VAV 댐퍼 개도율(%)"]) >= 50:
        Analyzed_Data.loc[i, 'Room1 거동유형'] = '댐퍼개도율 과다'          

    
for i in range(len(Analyzed_Data)):
    if float(Analyzed_Data.loc[i, "Room2_공급/설정_dT"]) > hp_1 and float(Analyzed_Data.loc[i, "3# VAV 댐퍼 개도율(%)"]) < 100 :
        Analyzed_Data.loc[i, 'Room2 거동유형'] = '댐퍼개도율 부족'
    elif float(Analyzed_Data.loc[i, "Room2_공급/설정_dT"]) > hp_1 and float(Analyzed_Data.loc[i, "3# VAV 댐퍼 개도율(%)"]) >= 100 :
        Analyzed_Data.loc[i, 'Room2 거동유형'] = 'VAV풍량 부족'
    elif (float(Analyzed_Data.loc[i, "Room2_공급/설정_dT"]) >= -hp_1 and float(Analyzed_Data.loc[i, "Room2_공급/설정_dT"]) <= hp_1) or (float(Analyzed_Data.loc[i, "Room2_공급/설정_dT"]) < -hp_1 and float(Analyzed_Data.loc[i, "3# VAV 댐퍼 개도율(%)"]) < 50) :
        Analyzed_Data.loc[i, 'Room2 거동유형'] = '정상 제어'
    elif float(Analyzed_Data.loc[i, "Room2_공급/설정_dT"]) < -hp_1 and float(Analyzed_Data.loc[i, "3# VAV 댐퍼 개도율(%)"]) >= 50:
        Analyzed_Data.loc[i, 'Room2 거동유형'] = '댐퍼개도율 과다'          


for i in range(len(Analyzed_Data)):
    if float(Analyzed_Data.loc[i, "Room3_공급/설정_dT"]) > hp_1 and float(Analyzed_Data.loc[i, "5# VAV 댐퍼 개도율(%)"]) < 100 :
        Analyzed_Data.loc[i, 'Room3 거동유형'] = '댐퍼개도율 부족'
    elif float(Analyzed_Data.loc[i, "Room3_공급/설정_dT"]) > hp_1 and float(Analyzed_Data.loc[i, "5# VAV 댐퍼 개도율(%)"]) >= 100 :
        Analyzed_Data.loc[i, 'Room3 거동유형'] = 'VAV풍량 부족'
    elif (float(Analyzed_Data.loc[i, "Room3_공급/설정_dT"]) >= -hp_1 and float(Analyzed_Data.loc[i, "Room3_공급/설정_dT"]) <= hp_1) or (float(Analyzed_Data.loc[i, "Room3_공급/설정_dT"]) < -hp_1 and float(Analyzed_Data.loc[i, "5# VAV 댐퍼 개도율(%)"]) < 50) :
        Analyzed_Data.loc[i, 'Room3 거동유형'] = '정상 제어'
    elif float(Analyzed_Data.loc[i, "Room3_공급/설정_dT"]) < -hp_1 and float(Analyzed_Data.loc[i, "5# VAV 댐퍼 개도율(%)"]) >= 50:
        Analyzed_Data.loc[i, 'Room3 거동유형'] = '댐퍼개도율 과다'        



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
    print("Result of ",column_name)
    print(result_df.to_string(index=True))
    print("--------------------------------------------------------------")


# 함수를 사용해서 거동유형별 수치 계산함
# 데이터프레임과, 열만 선택하면 됌
analyze_movement_type(Analyzed_Data, 'Room1 거동유형')
analyze_movement_type(Analyzed_Data, 'Room2 거동유형')
analyze_movement_type(Analyzed_Data, 'Room3 거동유형')

#%% (5): 헌팅판단                                                               Analyzed_Data : 거동분류 + 헌팅판단
Analyzed_Data['Room1_온도제어'] = 0
Analyzed_Data['Room2_온도제어'] = 0
Analyzed_Data['Room3_온도제어'] = 0


for i in range(len(Analyzed_Data)):
    if float(Analyzed_Data.loc[i, "Room1_공급/설정_dT"]) >= -1 and float(Analyzed_Data.loc[i, "Room1_공급/설정_dT"]) <= 1 :
        Analyzed_Data.loc[i, 'Room1_온도제어'] = 'Good'
    elif float(Analyzed_Data.loc[i, "Room1_공급/설정_dT"]) > 1 and float(Analyzed_Data.loc[i, "Room1_공급/설정_dT"]) <= 2.5:
        Analyzed_Data.loc[i, 'Room1_온도제어'] = 'Light'        
    elif float(Analyzed_Data.loc[i, "Room1_공급/설정_dT"]) > 2.5 and float(Analyzed_Data.loc[i, "Room1_공급/설정_dT"]) <= 4:
        Analyzed_Data.loc[i, 'Room1_온도제어'] = 'Moderate'  
    elif float(Analyzed_Data.loc[i, "Room1_공급/설정_dT"]) > 4:
        Analyzed_Data.loc[i, 'Room1_온도제어'] = 'Severe' 
    else:
        Analyzed_Data.loc[i, 'Room1_온도제어'] = 'Cold' 


for i in range(len(Analyzed_Data)):
    if float(Analyzed_Data.loc[i, "Room2_공급/설정_dT"]) >= -1 and float(Analyzed_Data.loc[i, "Room2_공급/설정_dT"]) <= 1 :
        Analyzed_Data.loc[i, 'Room2_온도제어'] = 'Good'
    elif float(Analyzed_Data.loc[i, "Room2_공급/설정_dT"]) > 1 and float(Analyzed_Data.loc[i, "Room2_공급/설정_dT"]) <= 2.5:
        Analyzed_Data.loc[i, 'Room2_온도제어'] = 'Light'        
    elif float(Analyzed_Data.loc[i, "Room2_공급/설정_dT"]) > 2.5 and float(Analyzed_Data.loc[i, "Room2_공급/설정_dT"]) <= 4:
        Analyzed_Data.loc[i, 'Room2_온도제어'] = 'Moderate'  
    elif float(Analyzed_Data.loc[i, "Room2_공급/설정_dT"]) > 4:
        Analyzed_Data.loc[i, 'Room2_온도제어'] = 'Severe' 
    else:
        Analyzed_Data.loc[i, 'Room2_온도제어'] = 'Cold' 
        
        
for i in range(len(Analyzed_Data)):
    if float(Analyzed_Data.loc[i, "Room3_공급/설정_dT"]) >= -1 and float(Analyzed_Data.loc[i, "Room3_공급/설정_dT"]) <= 1 :
        Analyzed_Data.loc[i, 'Room3_온도제어'] = 'Good'
    elif float(Analyzed_Data.loc[i, "Room3_공급/설정_dT"]) > 1 and float(Analyzed_Data.loc[i, "Room3_공급/설정_dT"]) <= 2.5:
        Analyzed_Data.loc[i, 'Room3_온도제어'] = 'Light'        
    elif float(Analyzed_Data.loc[i, "Room3_공급/설정_dT"]) > 2.5 and float(Analyzed_Data.loc[i, "Room3_공급/설정_dT"]) <= 4:
        Analyzed_Data.loc[i, 'Room3_온도제어'] = 'Moderate'  
    elif float(Analyzed_Data.loc[i, "Room3_공급/설정_dT"]) > 4:
        Analyzed_Data.loc[i, 'Room3_온도제어'] = 'Severe' 
    else:
        Analyzed_Data.loc[i, 'Room3_온도제어'] = 'Cold'     

#%% (6): 방별로 자료 분류하기                                                    Room1,2,3_Analyzed_Data: 각 방 별로 분류
Room1_Analyzed_Data = Analyzed_Data.iloc[:,[0,1,
                         11,12,23,26]] 
Room1_new_column_names = {'date': 'date',
                    'Time': 'Time',
                    '1# VAV 댐퍼 개도율(%)': 'VAV 댐퍼개도율',
                    '1# VAV 풍량(M3/h)': 'VAV 풍량',
                    'Room1 거동유형': '거동유형',
                    'Room1_온도제어': '온도제어',}
Room1_Analyzed_Data.rename(columns=Room1_new_column_names, inplace=True)



Room2_Analyzed_Data = Analyzed_Data.iloc[:,[0,1,
                         16,17,24,27]] 
Room2_new_column_names = {'date': 'date',
                    'Time': 'Time',
                    '3# VAV 댐퍼 개도율(%)': 'VAV 댐퍼개도율',
                    '3# VAV 풍량(M3/h)': 'VAV 풍량',
                    'Room2 거동유형': '거동유형',
                    'Room2_온도제어': '온도제어',}
Room2_Analyzed_Data.rename(columns=Room2_new_column_names, inplace=True)



Room3_Analyzed_Data = Analyzed_Data.iloc[:,[0,1,
                         21,22,25,28]] 
Room3_new_column_names = {'date': 'date',
                    'Time': 'Time',
                    '5# VAV 댐퍼 개도율(%)': 'VAV 댐퍼개도율',
                    '5# VAV 풍량(M3/h)': 'VAV 풍량',
                    'Room3 거동유형': '거동유형',
                    'Room3_온도제어': '온도제어',}
Room3_Analyzed_Data.rename(columns=Room3_new_column_names, inplace=True)



#%% (7): Room 개별 시간단위 묶기                                                Room1,2,3_Hourly_list: 시간별 그룹화
def create_hourly_list(df):
    df['hour'] = pd.to_datetime(df['date'] + ' ' + df['Time'], format='%Y-%m-%d %H:%M').dt.hour
    df['hour'] = df['hour'].astype(str).str.zfill(2)
    Hourly_list = []

    for date_hour, group in df.groupby(['date', 'hour']):
        date = group['date'].unique()[0]
        hour = group['hour'].astype(str).unique()[0]
        Hourly_list.append(group.copy())
    return Hourly_list

Room1_Hourly_list = create_hourly_list(Room1_Analyzed_Data)
Room2_Hourly_list = create_hourly_list(Room2_Analyzed_Data)
Room3_Hourly_list = create_hourly_list(Room3_Analyzed_Data)


#%% (8): Room 개별 시간단위 거동분석                                             Room1,2,3_Hourly_Analyzed : 시간단위 분석 내용
def analyze_room_hourly_activities(Room_Hourly_list):
    Room_Hourly_Analyzed = pd.DataFrame(columns=['date', 'hour',
                                                 '개도율부족 여부', '풍량문제 여부', '개도율과다 여부', 'Total',
                                                 '댐퍼개도율 부족', 'VAV풍량 부족', '댐퍼개도율 과다', '정상 제어',
                                                 '진단결과','거동코드'])
    activity_code_map = {
        '댐퍼개도율 부족': 1,
        'VAV풍량 부족': 2,
        '댐퍼개도율 과다': 3,
        '정상 제어': 4,
    }
    
    
    for group in Room_Hourly_list:
        date = group['date'].unique()[0]
        hour = group['hour'].unique()[0]
        Room_activities = group['거동유형'].tolist()
        
        Room_Analyzed_Result = None
        Room_Abnormal_1 = 0
        Room_Abnormal_2 = 0
        Room_Abnormal_3 = 0
        Room_count = 0  
        Room_Activity_1 = 0
        Room_Activity_2 = 0
        Room_Activity_3 = 0
        Room_Activity_4 = 0
        
        
        #각 거동유형별로 데이터 처리
        for Room_activity in Room_activities:
            if Room_activity == '댐퍼개도율 부족':
                Room_count += 1
                Room_Activity_1 += 1
                if Room_count >= 10:
                    Room_Abnormal_1 = 1
                    Room_Analyzed_Result = '댐퍼개도율 부족'
                    
            else:
                Room_count = 0
                
        for Room_activity in Room_activities:
            if Room_activity == 'VAV풍량 부족':
                Room_count += 1
                Room_Activity_2 += 1
                if Room_count >= 10:
                    Room_Abnormal_2 = 1
                    Room_Analyzed_Result = 'VAV풍량 부족'
                    
            else:
                Room_count = 0
        
        for Room_activity in Room_activities:
            if Room_activity == '댐퍼개도율 과다':
                Room_count += 1
                Room_Activity_3 += 1
                if Room_count >= 10:
                    Room_Abnormal_3 = 1
                    Room_Analyzed_Result = '댐퍼개도율 과다'
                    
            else:
                Room_count = 0
        
        for Room_activity in Room_activities:
            if Room_activity == '정상 제어':
                Room_Activity_4 += 1


   
        Room_Total = Room_Abnormal_1 + Room_Abnormal_2 + Room_Abnormal_3 
        
        if Room_Total == 0:
            Room_max_count = max(Room_activities.count(Room_activity) for Room_activity in Room_activities)
            Room_Analyzed_Result = [Room_activity for Room_activity in set(Room_activities) if Room_activities.count(Room_activity) == Room_max_count][0]
        elif Room_Total >= 2:
            Room_max_count = max(Room_activities.count(Room_activity) for Room_activity in Room_activities)
            Room_Analyzed_Result = [Room_activity for Room_activity in set(Room_activities) if Room_activities.count(Room_activity) == Room_max_count][0]
        
        
        
        
        Room_Hourly_Analyzed = pd.concat([Room_Hourly_Analyzed, pd.DataFrame({
            'date': [date],
            'hour': [hour],
            '개도율부족 여부': [Room_Abnormal_1], 
            '풍량문제 여부': [Room_Abnormal_2],
            '개도율과다 여부': [Room_Abnormal_3],
            'Total': [Room_Total],
            '댐퍼개도율 부족' : [Room_Activity_1],
            'VAV풍량 부족' : [Room_Activity_2], 
            '댐퍼개도율 과다' : [Room_Activity_3],
            '정상 제어' : [Room_Activity_4],
            '진단결과': [Room_Analyzed_Result]
        })], ignore_index=True)
    
    Room_Hourly_Analyzed['거동코드'] = Room_Hourly_Analyzed['진단결과'].map(activity_code_map)
    Room_numeric_columns = ['개도율부족 여부', '풍량문제 여부', '개도율과다 여부', 'Total',
                             '댐퍼개도율 부족', 'VAV풍량 부족', '댐퍼개도율 과다', '정상 제어']
    Room_Hourly_Analyzed[Room_numeric_columns] = Room_Hourly_Analyzed[Room_numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    return Room_Hourly_Analyzed



Room1_Hourly_Analyzed = analyze_room_hourly_activities(Room1_Hourly_list)
Room2_Hourly_Analyzed = analyze_room_hourly_activities(Room2_Hourly_list)
Room3_Hourly_Analyzed = analyze_room_hourly_activities(Room3_Hourly_list)


#%% (9): 히트맵 양식을 위하여                                                    DF

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

#%% (10): Room 개별 데이터프레임 + 히트맵                                        Room1,2,3_Pivot_Data_1~7 : 유형별 데이터

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

Room1_Pivot_DF_1 = create_pivot_data(DF, Room1_Hourly_Analyzed, '댐퍼개도율 부족', -10, 0)
Room1_Pivot_DF_2 = create_pivot_data(DF, Room1_Hourly_Analyzed, 'VAV풍량 부족', -10, 0)
Room1_Pivot_DF_3 = create_pivot_data(DF, Room1_Hourly_Analyzed, '댐퍼개도율 과다', -10, 0)
Room1_Pivot_DF_4 = create_pivot_data(DF, Room1_Hourly_Analyzed, '정상 제어', -10, 0)


Room2_Pivot_DF_1 = create_pivot_data(DF, Room2_Hourly_Analyzed, '댐퍼개도율 부족', -10, 0)
Room2_Pivot_DF_2 = create_pivot_data(DF, Room2_Hourly_Analyzed, 'VAV풍량 부족', -10, 0)
Room2_Pivot_DF_3 = create_pivot_data(DF, Room2_Hourly_Analyzed, '댐퍼개도율 과다', -10, 0)
Room2_Pivot_DF_4 = create_pivot_data(DF, Room2_Hourly_Analyzed, '정상 제어', -10, 0)


Room3_Pivot_DF_1 = create_pivot_data(DF, Room3_Hourly_Analyzed, '댐퍼개도율 부족', -10, 0)
Room3_Pivot_DF_2 = create_pivot_data(DF, Room3_Hourly_Analyzed, 'VAV풍량 부족', -10, 0)
Room3_Pivot_DF_3 = create_pivot_data(DF, Room3_Hourly_Analyzed, '댐퍼개도율 과다', -10, 0)
Room3_Pivot_DF_4 = create_pivot_data(DF, Room3_Hourly_Analyzed, '정상 제어', -10, 0)


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


plot_heatmap(Room1_Pivot_DF_1, title="Room1 댐퍼개도율 부족", cmap='YlOrRd')
plot_heatmap(Room1_Pivot_DF_2, title="Room1 VAV풍량 부족", cmap='YlOrRd')
plot_heatmap(Room1_Pivot_DF_3, title="Room1 댐퍼개도율 과다", cmap='YlOrRd')
plot_heatmap(Room1_Pivot_DF_4, title="Room1 정상 제어", cmap='Blues')


plot_heatmap(Room2_Pivot_DF_1, title="Room2 댐퍼개도율 부족", cmap='YlOrRd')
plot_heatmap(Room2_Pivot_DF_2, title="Room2 VAV풍량 부족", cmap='YlOrRd')
plot_heatmap(Room2_Pivot_DF_3, title="Room2 댐퍼개도율 과다", cmap='YlOrRd')
plot_heatmap(Room2_Pivot_DF_4, title="Room2 정상 제어", cmap='Blues')


plot_heatmap(Room3_Pivot_DF_1, title="Room3 댐퍼개도율 부족", cmap='YlOrRd')
plot_heatmap(Room3_Pivot_DF_2, title="Room3 VAV풍량 부족", cmap='YlOrRd')
plot_heatmap(Room3_Pivot_DF_3, title="Room3 댐퍼개도율 과다", cmap='YlOrRd')
plot_heatmap(Room3_Pivot_DF_4, title="Room3 정상 제어", cmap='Blues')



#%% (11): Room 개별 데이터프레임 + 히트맵                                        Room1,2,3,_Pivot_DF_All: 전체 데이터프레임
#####  요거만 하면 끝남

Room1_Pivot_DF_All = create_pivot_data(DF, Room1_Hourly_Analyzed, '거동코드', 0, 5)
Room2_Pivot_DF_All = create_pivot_data(DF, Room2_Hourly_Analyzed, '거동코드', 0, 5)
Room3_Pivot_DF_All = create_pivot_data(DF, Room3_Hourly_Analyzed, '거동코드', 0, 5)

color_map = ListedColormap(['white', '#B381D9', '#FF8989','#FFFF9F' ,'#D7E7F5', '#ADADAD'])
# 0: 데이터 X
# 1: 댐퍼개도율 부족
# 2: VAV풍량 부족
# 3: 댐퍼개도율 과다
# 4: 정상 제어
# 5: 전체시스템 OFF
def plot_calendar_heatmap(dataframe, color_map, title):
    plt.figure(figsize=(8, 10))
    sns.heatmap(dataframe, cmap=color_map, linewidths=0.5, linecolor="lightgray", annot=False, cbar=True)

    plt.title(title, fontproperties=font_prop)
    plt.xlabel("Hour", fontsize=10, fontproperties=font_prop)
    plt.ylabel("Date", fontsize=10, fontproperties=font_prop)
    plt.show()

plot_calendar_heatmap(Room1_Pivot_DF_All, color_map, 'Room1 전체 거동 유형')
plot_calendar_heatmap(Room2_Pivot_DF_All, color_map, 'Room2 전체 거동 유형')
plot_calendar_heatmap(Room3_Pivot_DF_All, color_map, 'Room3 전체 거동 유형')
#%% (12): Room 종합 분단위 거동분석                                             Totally_Analyzed_Data: 방을 모두 고려

Totally_Analyzed_Data = Analyzed_Data.copy()

Totally_Analyzed_Data['전체 거동유형'] = None

for i in range(len(Totally_Analyzed_Data)):
    
    Room3_vav_damper_condition_1 = float(Analyzed_Data.loc[i, "5# VAV 댐퍼 개도율(%)"]) < 100
    Room3_vav_damper_condition_2 = float(Analyzed_Data.loc[i, "5# VAV 댐퍼 개도율(%)"]) >= 100
    
    Room1_condition_1 = Totally_Analyzed_Data.loc[i, "Room1 거동유형"] == '정상 제어'
    Room2_condition_1 = Totally_Analyzed_Data.loc[i, "Room2 거동유형"] == '정상 제어'
    Room3_condition_1 = Totally_Analyzed_Data.loc[i, "Room3 거동유형"] == '정상 제어'

    Room1_condition_2 = Totally_Analyzed_Data.loc[i, "Room1 거동유형"] == 'VAV풍량 부족'
    Room2_condition_2 = Totally_Analyzed_Data.loc[i, "Room2 거동유형"] == 'VAV풍량 부족'
    Room3_condition_2 = Totally_Analyzed_Data.loc[i, "Room3 거동유형"] == 'VAV풍량 부족'
    
    Room1_condition_3 = Totally_Analyzed_Data.loc[i, "Room1 거동유형"] == '댐퍼개도율 부족'
    Room2_condition_3 = Totally_Analyzed_Data.loc[i, "Room2 거동유형"] == '댐퍼개도율 부족'
    Room3_condition_3 = Totally_Analyzed_Data.loc[i, "Room3 거동유형"] == '댐퍼개도율 부족'
    
    Room1_condition_4 = Totally_Analyzed_Data.loc[i, "Room1 거동유형"] == '댐퍼개도율 과다'
    Room2_condition_4 = Totally_Analyzed_Data.loc[i, "Room2 거동유형"] == '댐퍼개도율 과다'
    Room3_condition_4 = Totally_Analyzed_Data.loc[i, "Room3 거동유형"] == '댐퍼개도율 과다'
    

    if (Room1_condition_1 and Room2_condition_1 and Room3_condition_1):
        Totally_Analyzed_Data.loc[i, '전체 거동유형'] = '방 전체 공조상태 우수'
    elif (Room3_condition_1 and Room3_vav_damper_condition_1 and (Room1_condition_2 or Room2_condition_2)):
        Totally_Analyzed_Data.loc[i, '전체 거동유형'] = 'Room3댐퍼에 의한 풍량부족'
    elif ((Room1_condition_1 or Room1_condition_2) and (Room2_condition_1 or Room2_condition_2) and (Room3_condition_1 or Room3_condition_2) and Room3_vav_damper_condition_2):
        Totally_Analyzed_Data.loc[i, '전체 거동유형'] = '풍량 부족'
    elif ((Room1_condition_1 or Room1_condition_3) and (Room2_condition_1 or Room2_condition_3) and (Room3_condition_1 or Room3_condition_3)):
        Totally_Analyzed_Data.loc[i, '전체 거동유형'] = '댐퍼개도율 부족'
    elif ((Room1_condition_1 or Room1_condition_4) and (Room2_condition_1 or Room2_condition_4) and (Room3_condition_1 or Room3_condition_4)):
        Totally_Analyzed_Data.loc[i, '전체 거동유형'] = '댐퍼개도율 과다'
    else:
        Totally_Analyzed_Data.loc[i, '전체 거동유형'] = '기타'
        

analyze_movement_type(Totally_Analyzed_Data, '전체 거동유형')
        
#%% (13): Room 종합 시간단위 묶기                                               Totally_Hourly_list: 시간단위 묶기

Totally_Hourly_list = create_hourly_list(Totally_Analyzed_Data)
#%% (14): Room 종합 시간단위 거동분석                                            Totally_Hourly_Analyzed: 방 전체 시간단위 분석 내용
def Analyze_Total_hourly_activities(Totall_Hourly_list):
    Totally_Hourly_Analyzed = pd.DataFrame(columns=['date', 'hour',
                                                 '풍량부족 여부', '개도율과다 여부', '개도율부족 여부', 'Room3 영향 여부','Total',
                                                 '풍량 부족', '댐퍼개도율 과다', '댐퍼개도율 부족','Room3댐퍼에 의한 풍량부족','방 전체 공조상태 우수','기타',
                                                 '진단결과','거동코드'])
    activity_code_map = {
        '풍량 부족': 1,
        '댐퍼개도율 과다': 2,
        '댐퍼개도율 부족': 3,
        'Room3댐퍼에 의한 풍량부족' : 4,
        '방 전체 공조상태 우수': 5,
        '기타': 6
    }
    
    for group in Totall_Hourly_list:
        date = group['date'].unique()[0]
        hour = group['hour'].unique()[0]
        Totall_activities = group['전체 거동유형'].tolist()
        
        Totall_Analyzed_Result = None
        
        Totall_Abnormal_1 = 0
        Totall_Abnormal_2 = 0
        Totall_Abnormal_3 = 0
        Totall_Abnormal_4 = 0
        
        Totall_count = 0  
        
        Totall_Activity_1 = 0
        Totall_Activity_2 = 0
        Totall_Activity_3 = 0
        Totall_Activity_4 = 0
        Totall_Activity_5 = 0
        Totall_Activity_6 = 0
        
        #각 거동유형별로 데이터 처리
        for Totall_activity in Totall_activities:
            if Totall_activity == '풍량 부족':
                Totall_count += 1
                Totall_Activity_1 += 1
                if Totall_count >= 10:
                    Totall_Abnormal_1 = 1
                    Totall_Analyzed_Result = '풍량 부족'
                    
            else:
                Totall_count = 0
     
                
        for Totall_activity in Totall_activities:
            if Totall_activity == '댐퍼개도율 과다':
                Totall_count += 1
                Totall_Activity_2 += 1
                if Totall_count >= 10:
                    Totall_Abnormal_2 = 1
                    Totall_Analyzed_Result = '댐퍼개도율 과다'
                    
            else:
                Totall_count = 0

        for Totall_activity in Totall_activities:
            if Totall_activity == '댐퍼개도율 부족':
                Totall_count += 1
                Totall_Activity_3 += 1
                if Totall_count >= 10:
                    Totall_Abnormal_3 = 1
                    Totall_Analyzed_Result = '댐퍼개도율 부족'
                    
            else:
                Totall_count = 0
                
        for Totall_activity in Totall_activities:
            if Totall_activity == 'Room3댐퍼에 의한 풍량부족':
                Totall_count += 1
                Totall_Activity_4 += 1
                if Totall_count >= 10:
                    Totall_Abnormal_4 = 1
                    Totall_Analyzed_Result = 'Room3댐퍼에 의한 풍량부족'
                    
            else:
                Totall_count = 0
                
        for Totall_activity in Totall_activities:
            if Totall_activity == '방 전체 공조상태 우수':
                Totall_Activity_5 += 1

        for Totall_activity in Totall_activities:
            if Totall_activity == '기타':
                Totall_Activity_6 += 1
   
        Totall_Sum = Totall_Abnormal_1 + Totall_Abnormal_2 + Totall_Abnormal_3 + Totall_Abnormal_4 
        
        if Totall_Sum == 0:
            Totall_max_count = max(Totall_activities.count(Totall_activity) for Totall_activity in Totall_activities)
            Totall_Analyzed_Result = [Totall_activity for Totall_activity in set(Totall_activities) if Totall_activities.count(Totall_activity) == Totall_max_count][0]
        elif Totall_Sum >= 2:
            Totall_max_count = max(Totall_activities.count(Totall_activity) for Totall_activity in Totall_activities)
            Totall_Analyzed_Result = [Totall_activity for Totall_activity in set(Totall_activities) if Totall_activities.count(Totall_activity) == Totall_max_count][0]
        

        Totally_Hourly_Analyzed = pd.concat([Totally_Hourly_Analyzed, pd.DataFrame({
            'date': [date],'hour': [hour],
            '풍량부족 여부': [Totall_Abnormal_1], 
            '개도율과다 여부': [Totall_Abnormal_2],
            '개도율부족 여부': [Totall_Abnormal_3], 
            'Room3 영향 여부': [Totall_Abnormal_4],
            'Total': [Totall_Sum],
            '풍량 부족' : [Totall_Activity_1],
            '댐퍼개도율 과다' : [Totall_Activity_2], 
            '댐퍼개도율 부족' : [Totall_Activity_3],
            'Room3댐퍼에 의한 풍량부족' : [Totall_Activity_4],
            '방 전체 공조상태 우수' : [Totall_Activity_5],
            '기타' : [Totall_Activity_6],
            '진단결과': [Totall_Analyzed_Result]
        })], ignore_index=True)
    
    Totally_Hourly_Analyzed['거동코드'] = Totally_Hourly_Analyzed['진단결과'].map(activity_code_map)
    Totall_numeric_columns = ['풍량부족 여부', '개도율과다 여부', '개도율부족 여부', 'Room3 영향 여부','Total',
                             '풍량 부족', '댐퍼개도율 과다', '댐퍼개도율 부족', 'Room3댐퍼에 의한 풍량부족','방 전체 공조상태 우수', '기타']
    Totally_Hourly_Analyzed[Totall_numeric_columns] = Totally_Hourly_Analyzed[Totall_numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    return Totally_Hourly_Analyzed      
        
        
Totally_Hourly_Analyzed = Analyze_Total_hourly_activities(Totally_Hourly_list)

#%% (15): Room 종합 유형 데이터프레임 + 히트맵                                   Totall_Pivot_DF: Room종합 개별 거동유형

Totall_Pivot_DF_1 = create_pivot_data(DF, Totally_Hourly_Analyzed, '풍량 부족', -10, 0)
Totall_Pivot_DF_2 = create_pivot_data(DF, Totally_Hourly_Analyzed, '댐퍼개도율 과다', -10, 0)
Totall_Pivot_DF_3 = create_pivot_data(DF, Totally_Hourly_Analyzed, '댐퍼개도율 부족', -10, 0)
Totall_Pivot_DF_4 = create_pivot_data(DF, Totally_Hourly_Analyzed, 'Room3댐퍼에 의한 풍량부족', -10, 0)
Totall_Pivot_DF_5 = create_pivot_data(DF, Totally_Hourly_Analyzed, '방 전체 공조상태 우수', -10, 0)
Totall_Pivot_DF_6 = create_pivot_data(DF, Totally_Hourly_Analyzed, '기타', -10, 0)



plot_heatmap(Totall_Pivot_DF_1, title='Room 종합 : 풍량 부족', cmap='YlOrRd')
plot_heatmap(Totall_Pivot_DF_2, title='Room 종합 : 댐퍼개도율 과다', cmap='YlOrRd')
plot_heatmap(Totall_Pivot_DF_3, title='Room 종합 : 댐퍼개도율 부족', cmap='YlOrRd')
plot_heatmap(Totall_Pivot_DF_4, title='Room 종합 : Room3댐퍼에 의한 풍량부족', cmap='YlOrRd')
plot_heatmap(Totall_Pivot_DF_5, title='Room 종합 : 방 전체 공조상태 우수', cmap='Blues')
plot_heatmap(Totall_Pivot_DF_6, title='Room 종합 : 기타', cmap='YlOrRd')

#%% (16): Room 종합 전체 데이터프레임 + 히트맵                                   Totall_Pivot_DF_All: Room종합 거동유형 통합

Totall_Pivot_DF_All = create_pivot_data(DF, Totally_Hourly_Analyzed, '거동코드', 0, 6)

color_map2 = ListedColormap(['white', '#FF949A', '#FFFFCC','#A9D18E' ,'#CC71FF', '#D7E7F5','#BABABA'])
# 0 : 데이터 x
# 1 : 풍량 부족
# 2 : 댐퍼개도율 과다
# 3 : 댐퍼개도율 부족
# 4 : Room3댐퍼에 의한 풍량부족
# 5 : 방 전체 공조상태 우수
# 6 : 데이터가 있었던 부분

plot_calendar_heatmap(Totall_Pivot_DF_All, color_map2, 'Room종합 거동유형')




