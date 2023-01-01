 ![image](https://user-images.githubusercontent.com/28619804/199662224-3b4a84a3-2c5d-40fc-9ccd-513f63219573.png)
 
 # Book Rating Prediction-Recsys 06 TEAM
  
</div>
<br /> 

# 팀원 소개

| <img src="https://user-images.githubusercontent.com/79916736/207600031-b46e76d2-cba3-4c94-9fc3-d9f29cd3bef8.png" width=200> | <img src="https://user-images.githubusercontent.com/94108712/207613142-1c44bb87-f685-4b25-be16-efd06f8793ba.jpg" width=200> | <img src="https://user-images.githubusercontent.com/79916736/210170572-5323396d-2c7b-4160-8aca-9d274bfdc833.jpg" width=200> | <img src="https://user-images.githubusercontent.com/22442453/208286371-7e029cce-af2b-4cc2-89c5-851db98b7d19.jpg" width=200> | <img src="https://avatars.githubusercontent.com/u/28619804?v=4" width="200px"> |
| :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: |
|                                           [김성연](https://github.com/KSY1526)                                            |                                           [김찬호](https://github.com/KChanho)                                            |                                            [박문순](https://github.com/hxcva1)                                            |                                         [배성수](https://github.com/sungsubae)                                          |                                            [이지훈](https://github.com/JJI-Hoon)                                            |

# 🏆️ 프로젝트 목표
<!-- <p align="center"><img src="https://user-images.githubusercontent.com/65529313/168472960-0eac76e2-4fe3-4ebc-b093-f9c0aab59859.png" /></p> -->
- 사용자의 책 평점 데이터를 바탕으로 사용자가 어떤 책을 더 선호할지 예측하는 태스크

<br /> 
<br /> 

# 💻 활용 장비
- GPU Tesla V100-PCIE-32GB

<br /> 
<br /> 

# 🙋🏻‍♂️🏻‍♀️ 프로젝트 팀 구성 및 역할
- **김성연**: 팀의 방향성 설정 및 아이디어 제공, 데이터 EDA 진행, DeepCoNN 모델 실험해보기
- **김찬호**: 노션을 활용해 팀원의 진행과정 기록하기, TabNet 모델 실험해보기
- **박문순**: 깃허브 전반적인 버전 관리, FM, FFM 모델 실험해보기 
- **배성수**: FFM+DCN 모델 구현 및 고도화 하기, Cold-Start 관련 분석해보기
- **이지훈**: 부스팅 기반 모델 실험해보기, K-fold 및 Wandb와 Optuna 등을 사용해 모델 고도화 하기 


<br /> 
<br /> 

# 🏗️ Model Architecture
<!-- <p align="center"><img src="https://user-images.githubusercontent.com/65529313/168473170-938e1ce0-395f-40be-9118-ea127668b11d.png" /></p> -->

- 범주형 데이터 처리에 좋은 성능을 내는 Catboost Model 이용

<br /> 
<br /> 

# 🛠 Project Template
<!-- <p align="center"><img src="https://user-images.githubusercontent.com/65529313/168473184-7a7a5c9b-f7da-4d92-81d8-965ecd1f934f.png" /></p> -->

- 학습 환경의 경우 [pytorch-template](https://github.com/victoresque/pytorch-template)을 이용하여 이번 대회의 학습환경에 맞춰 리팩토링을 진행함

<br /> 
<br /> 

# 💯 프로젝트 수행 결과 - 최종 Private 1등

|리더보드| auroc  |     순위     |
|:--------:|:------:|:----------:|
|public| 2.1095 |  **1위**   |
|private| 2.1045 | **최종 1위** |

![image](https://user-images.githubusercontent.com/28619804/199737010-460bcab6-a537-4b5b-b2b8-cb65100e91da.png)


# 모델 실행 순서
1) FFDCN.ipynb 파일 실행 => submit/FFDCN.csv 파일 생성
2) CatBoost.ipynb 파일 실행 => submit/CatBost.csv 파일 생성
3) Ensemble.ipynb 파일 실행 => submit/Ensemble.csv 파일 생성

상세한 프로젝트 내용은 발표자료 및 레포트를 참고해주세요!