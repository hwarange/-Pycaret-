# -Pycaret를 활용한 Tripadvisor Hotel 데이터 분석-
2023년도 2학기 비즈니스 머신러닝 과제입니다.

### 개요
____
(데이터는 머신러닝 과제로써 받은 데이터이고 문제시 삭제하겠습니다!)


목적: Tripadvisor Hotel 데이터를 이용해서 별점을 예측하기! (Target은 bubble_rating)


평가지표: RMSE를 사용했습니다! 


![image](https://github.com/hwarange/-Pycaret-/blob/main/image.png)


개발환경 및 Pycaret 버전: 구글 코랩, Pycaret 3.2.0

____
Pycaret 3.2.0에서 지원하는 Scikit-learn을 활용하여 Regression을 진행을 하였다. 코드는 다음과 같다.

    !pip install pycaret
    !pip install --upgrade pycaret
    from pycaret.regression import *

이후 데이터 전처리는 간단하게 target인 “bubble_rating”과 연관성이 없어 보이는 호텔이름, 링크와 순서 컬럼을 drop을 사용해서 버려주고 class 컬럼에 no_stars를 전부 0으로 대체하였다.
    
    from google.colab import drive
    drive.mount('/content/drive')
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    result = pd.read_csv("/content/drive/MyDrive/비즈니스 머신러닝/final_project_hotelReview.csv", encoding="CP949")
    result = result.drop(columns = "Unnamed: 0")
    result = result.drop(columns = "hotel_url")
    result = result.drop(columns = "name")
    result.loc[result['class'] == "no stars", 'class'] = 0
    result = result.astype({"class":int})
    result["class"]


그리고 결측값은 dropna()를 사용하여 제거하였다.

    result = result.dropna()

    
그 후에는 Pycaret의 set_up함수로 모델을 생성해 주었다.


target은 “bubble_rating”으로 훈련데이터와 테스트데이터 비율은 8:2로 맞추고 random_state는 123고정하고 빠른 계산을 위해 gpu를 사용하였다.

    model = setup(data = result, target = "bubble_rating",session_id=123, train_size=0.8,use_gpu=True)

compare_model을 사용해서 여러 회귀모델들의 평가지표를 비교하였고 그 결과 다음과 같다.

    compare_models()

!
