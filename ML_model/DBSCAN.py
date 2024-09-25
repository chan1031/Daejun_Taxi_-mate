import pandas as pd
from sklearn.cluster import DBSCAN
from datetime import datetime, timedelta
import numpy as np

# CSV 파일 로드
df = pd.read_csv('taxi_data.csv')

# 차량번호 데이터 제거
df = df.drop(columns=['차량번호'])

# 승차시간을 datetime으로 변환
df['승차시간'] = pd.to_datetime(df['승차시간'])

# 15분 단위로 시간 그룹화
df['시간그룹'] = df['승차시간'].dt.floor('15T')

# DBSCAN 클러스터링을 위한 데이터 준비
# 클러스터링에는 X좌표, Y좌표가 필요합니다
coords = df[['승차X좌표', '승차Y좌표']].values

# DBSCAN 적용
db = DBSCAN(eps=0.01, min_samples=5).fit(coords)  # eps와 min_samples는 데이터에 따라 조정 가능
df['클러스터'] = db.labels_

# 클러스터 중심점 계산
centroids = df.groupby(['클러스터'])[['승차X좌표', '승차Y좌표']].mean()

# 클러스터 중심점과 시간그룹, 요일을 연결
df = df.merge(centroids, how='left', on='클러스터', suffixes=('', '_중심'))

# 요일과 시간에 따른 추천 함수 정의
def recommend_location(day_of_week, time):
    # 입력된 시간과 요일에 맞는 데이터를 필터링
    time_group = pd.to_datetime(time).floor('15T')
    filtered_df = df[(df['요일'] == day_of_week) & (df['시간그룹'] == time_group)]
    
    if not filtered_df.empty:
        # 가장 빈도가 높은 클러스터의 중심 좌표를 반환
        recommended_cluster = filtered_df['클러스터'].mode()[0]
        location = centroids.loc[recommended_cluster]
        return location['승차X좌표'], location['승차Y좌표']
    else:
        return None, None

# 사용 예시
day = 'Saturday'
time = '2023-04-01 00:30:00'
x, y = recommend_location(day, time)
print(f"추천 위치: X = {x}, Y = {y}")
