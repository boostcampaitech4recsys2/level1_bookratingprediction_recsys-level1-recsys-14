# Book Rating Prediction

![image](https://user-images.githubusercontent.com/80964004/202848295-b8b91cb6-6fce-4b7f-8bdc-f683ca6f6028.png)

## 팀원 소개
|이름|깃허브|역할|
|----|---|---|
|김다은|[GitHub](https://github.com/daeni-dang)|NCF 모델 연구 및 실험, FFM 모델에 k-fold 적용|
|김동영|[GitHub](https://github.com/ktasha45)|deepconn 모델 고도화. deepconn+FFM 모델 개발. 하이퍼파라미터 튜닝 자동화|
|김보성|[GitHub](https://github.com/bo-oseng)|Hybird Model(NCF + FFM) 실험, Custom Ensemble(Warm + Cold) 실험, lightFM 라이브러리 모델 연구 및 실험|
|류지수|[GitHub](https://github.com/JisooRyu99)|FM, FFM 모델 연구 및 실험, EDA(결측치 보완), 하이퍼파라미터 튜닝 자동화|
|홍재형|[GitHub](https://github.com/secrett2633)|CNN_FM 연구 및 실험, EDA(결측치 보완)|


## 프로젝트 목표
 대한출판문화협회에 따르면 2021년 대한민국 신간 발행 책은 64,657권입니다. 책을 읽기 위해 소요되는 시간과 책을 구매하는 비용 측면에서 구매할 책을 선택하는 것은 중요한 문제입니다. 하지만 소비자는 제목, 저자, 표지, 카테고리 등 그 책의 정보와 타 구매자들의 리뷰와 평점만으로 책을 선택해야합니다.

 본 대회는 소비자들의 책 구매 결정에 도움을 주기 위해 소비자가 책에 대해 내릴 평점을 예측하는 것을 목표로 합니다.
 
 ## 프로젝트 구조
```bash
|📦 code
|    |📂 data
|        |📜 images
|        |📜 train_ratings.csv
|        |📜 test_ratings.csv
|        |📜 books.csv
|        |📜 sample_submission.csv
|        |📜 users.csv
|    |📂 models
|        |📜 FFM.pt
|    |📂 src
|        |📂 data
|            |📜 __init__.py
|            |📜 dl_data.py
|            |📜 context_data.py
|            |📜 image_data.py
|            |📜 text_data.py
|        |📂 ensembles
|            |📜 ensembles.py
|        |📂 models
|            |📜 _models.py
|            |📜 dl_models.py
|            |📜 context_models.py
|            |📜 image_models.py
|            |📜 text_models.py
|        |📜 __init__.py
|        |📜 utils.py
|    |📂 submit
|    |📜 main.py
|    |📜 ensemble.py
```

## 데이터셋 구조
주어진 데이터는 `.csv` 형식의 파일로, 사용자(user)과 책(item)의 정보 그리고 사용자(user)가 책(item)에 매긴 평점 데이터입니다.
- `users.csv` : 68,092명의 사용자(user)에 대한 정보를 담고 있는 메타데이터
- `books.csv` : 149,570개의 책(item)에 대한 정보를 담고 있는 메타데이터
- `train_ratings.csv` : 59,803명의 사용자(user)가 129,777개의 책(item)에 대해 남긴 306,795건의 평점(rating) 데이터

## 출력 데이터
 최종 결과물은 주어진 책에 대해 사용자가 매길 것이라고 예상하는 평점을 채워 넣은 .csv 형태의 파일로, 아래 `sample_submission.csv` 파일과 같이 출력해야 합니다.
| user_id | isbn | rating |
| --- | --- | --- |
| 11676 | 0002005018 | 7.4152054786 |
| 116866 | 0002005018 | 8.3852987289 |
| 152827 | 0060973129 | 8.0158720016 |

## 모델 개선
- baseline에는 없는 earlystopping 기능 추가, 보다 정확한 실험을 위해 rmse 도출 코드를 수정
- shell script와 tensorboard를 이용해서 최적의 hyper-parameter을 찾고 학습
- 10-fold validation을 목표로 모델을 학습 했으나 시간이 부족하여 4-fold vaildation의 결과를 제출

## 최종 결과
**Model** : `FFM`

**Hyper-Parameter** : `batch_size 256` `learning_rate 0.01` `weight_decay 1e-4` `epochs 10`

10-fold (4번 fold에서 멈춤)

- CV score (valid) : RMSE 2.1758
- LB score (Public) : RMSE 2.1414
- LB score (Private) : RMSE 2.1418
