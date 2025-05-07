# 🚴 Bicycle Demand Forecast

<img src="https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/> <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white"/> <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/> <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/> <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white"/>

**Bicycle Demand Forecast**은 여의나루역 1번출구 앞 공공자전거 대여 이력과 기상, 공휴일 데이터를 결합하여 LSTM+Attention 모델로 시간별 대여 수요를 예측하는 딥러닝 프로젝트입니다.  

---

## 🛠 주요 기능

* **공휴일 & 날씨 통합**  
  공공데이터 API로부터 공휴일을 조회하고, 기상청 CSV 데이터와 병합  
* **시퀀스 데이터 생성**  
  과거 'window_size' 시간만큼의 입력 시퀀스와 다음 시점 수요 레이블 생성  
* **LSTM + Attention 모델**  
  시계열 패턴을 학습하는 LSTM 뒤에 Self-Attention 레이어 결합  
* **모델 학습 및 평가**  
  RMSE, MAE 지표 계산 및 학습 중 과적합 방지를 위한 EarlyStopping  
* **결과 시각화**  
  예측값 vs 실제값 비교 그래프 출력

---

## ⚙️ 디렉토리 구조

```

BicycleDemandForecast\_Project/
├─ data/
│   ├─ Seoul\_Bike\_2401.csv       # 2024년 1월 대여 이력
│   ├─ Seoul\_Bike\_2402.csv       # 2024년 2월 대여 이력
│   ├─ …
│   └─ weather\_2024\_01-06.csv    # 2024년 상반기 날씨 정보
├─ Bicycle demand forecast.py    # 데이터 로드·전처리·모델 정의·학습·평가·시각화 통합 스크립트
└─ README.md                     # 프로젝트 소개 및 실행 가이드

```

---

## 🚀 빠른 시작

1. **리포지터리 클론**

   ```bash
   git clone https://github.com/aeeun-git/BicycleDemandForecast_Project.git
   cd BicycleDemandForecast_Project
    ```

2. **가상환경 설정**

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
     ```

3. **데이터 준비**
   'data/' 폴더에 공공자전거 대여 이력 CSV 파일(2401\~2406)과 날씨 정보 CSV 파일을 위치시킵니다.

4. **전체 파이프라인 실행**

   ```bash
   python "Bicycle demand forecast.py"
     ```

   * 스크립트 실행 시 데이터 로드 → 전처리 → 모델 학습 → 평가 → 시각화가 한 번에 수행됩니다.
   * 학습 완료 후 출력되는 RMSE, MAE와 그래프를 확인하세요.

---

## 🛠 기술 스택

* **언어**: Python 3.x
* **라이브러리**: Pandas, NumPy, scikit-learn, TensorFlow(Keras), Matplotlib, BeautifulSoup4, Requests
* **데이터**: 서울시 공공자전거 대여 이력, 기상청 CSV, 공휴일 API

---

## 📝 라이선스

MIT © [aeeun-git](https://github.com/aeeun-git)
