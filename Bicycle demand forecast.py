import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import urllib.parse
import requests
import datetime
from bs4 import BeautifulSoup

def get_holiday(year: int) -> pd.DataFrame:
    url = "http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo"
    api_key_utf8 = "Your API Key"  # 개인 API 키 입력
    api_key_decode = urllib.parse.unquote(api_key_utf8)

    temp = ["월", "화", "수", "목", "금", "토", "일"]
    item_list = []

    for month in range(1, 7):
        params = {"ServiceKey": api_key_decode, "solYear": year, "solMonth": str(month).zfill(2), "numOfRows": 100}
        response = requests.get(url, params=params)

        if response.status_code != 200:
            print(f"{year}년 {month}월 API 요청 실패: {response.status_code}")
            continue

        xml = BeautifulSoup(response.text, "html.parser")
        items = xml.find_all("item")

        for item in items:
            dt = datetime.datetime.strptime(item.find("locdate").text.strip(), "%Y%m%d")
            item_dict = {"공휴일명": item.find("datename").text.strip(), "날짜": dt.date(), "요일": temp[dt.weekday()]}
            item_list.append(item_dict)

    return pd.DataFrame(item_list)

holidays_2024 = get_holiday(2024)
holidays_2024 = set(holidays_2024["날짜"])

file_path_2401 = '서울특별시 공공자전거 대여이력 정보_2401.csv'
file_path_2402 = '서울특별시 공공자전거 대여이력 정보_2402.csv'
file_path_2403 = '서울특별시 공공자전거 대여이력 정보_2403.csv'
file_path_2404 = '서울특별시 공공자전거 대여이력 정보_2404.csv'
file_path_2405 = '서울특별시 공공자전거 대여이력 정보_2405.csv'
file_path_2406 = '서울특별시 공공자전거 대여이력 정보_2406.csv'
weather_path = '서울특별시 2024 01_06 날씨정보.csv'

data_2401 = pd.read_csv(file_path_2401, encoding='cp949')
data_2402 = pd.read_csv(file_path_2402, encoding='cp949')
data_2403 = pd.read_csv(file_path_2403, encoding='cp949')
data_2404 = pd.read_csv(file_path_2404, encoding='cp949')
data_2405 = pd.read_csv(file_path_2405, encoding='cp949')
data_2406 = pd.read_csv(file_path_2406, encoding='cp949')

data = pd.concat([data_2401, data_2402, data_2403, data_2404, data_2405, data_2406], ignore_index=True)
data.rename(columns={'대여 대여소명': 'RENT_NM', '대여일시': 'RENT_DT'}, inplace=True)

yeouinaru_data = data[data['RENT_NM'].str.contains('여의나루역 1번출구 앞', na=False)]
yeouinaru_data['RENT_DT'] = pd.to_datetime(yeouinaru_data['RENT_DT'], errors='coerce')
yeouinaru_data['hour'] = yeouinaru_data['RENT_DT'].dt.hour
yeouinaru_data['date'] = yeouinaru_data['RENT_DT'].dt.date
hourly_demand = yeouinaru_data.groupby(['date', 'hour']).size().reset_index(name='demand')

weather_data = pd.read_csv(weather_path, encoding='cp949')
weather_data['일시'] = pd.to_datetime(weather_data['일시'])
weather_data['hour'] = weather_data['일시'].dt.hour
weather_data['date'] = weather_data['일시'].dt.date
weather_data = weather_data.drop(columns=['지점', '지점명'])

weather_data['강수량(mm)'] = weather_data['강수량(mm)'].fillna(0)
weather_data['적설(cm)'] = weather_data['적설(cm)'].fillna(0)

combined_data = pd.merge(hourly_demand, weather_data, on=['date', 'hour'], how='left')
combined_data = combined_data.fillna(0)
combined_data['weekday'] = pd.to_datetime(combined_data['date']).dt.weekday

combined_data['is_holiday'] = combined_data['date'].isin(holidays_2024).astype(int)
combined_data['before_holiday'] = (combined_data['date'] - pd.Timedelta(days=1)).isin(holidays_2024).astype(int)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(combined_data[['기온(°C)', '강수량(mm)', '적설(cm)', '풍속(m/s)', 'weekday', 'hour', 'is_holiday', 'before_holiday']])
y_scaled = scaler_y.fit_transform(combined_data['demand'].values.reshape(-1, 1))

window_size = 6

def create_sequences(data, target, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(target[i + window_size])
    return np.array(X), np.array(y)

X, y = create_sequences(X_scaled, y_scaled, window_size)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
lstm_out = LSTM(128, return_sequences=True)(input_layer)
dropout = Dropout(0.3)(lstm_out)
attention = Attention()([dropout, dropout])
concat = Concatenate()([dropout, attention])
dense_output = Dense(1)(concat)

model = Model(inputs=input_layer, outputs=dense_output)
model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=30, batch_size=4, validation_data=(X_test, y_test),
                    callbacks=[early_stopping], verbose=1)

y_pred = model.predict(X_test)
y_test_rescaled = scaler_y.inverse_transform(y_test)
y_pred_rescaled = np.maximum(scaler_y.inverse_transform(y_pred), 0)

print(f"RMSE: {np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))}, MAE: {mean_absolute_error(y_test_rescaled, y_pred_rescaled)}")
