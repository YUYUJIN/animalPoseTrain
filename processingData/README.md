# Processing Data
> 프로젝트의 목적에 맞는 Fast-RCNN 모델과 LSTM 모델의 학습데이터에 맞게 데이터를 전처리하는 작업

## Data Validation
최종적으로 AIHub에서 다운 받을 수 있는 데이터는 약 7백만건이다. 이 중 Keypoint나 Bounding Box가 누락된 데이터는 제외하였다.  
<img src=https://github.com/YUYUJIN/animalPoseTrain/blob/main/images/keypointsnull.png></img>  
그 후 데이터 중에 Bounding Box가 문제가 되는 파일 또한 삭제하려 했으나 데이터량이 많아 프로젝트 인원이 모두 검수하지 못한 경우도 있었다.  
<img src=https://github.com/YUYUJIN/animalPoseTrain/blob/main/images/bbox.jpg></img>  

최종적인 데이터는 고양이는 약 180만개, 강아지는 약 95만개의 Keypoint, Bounding Box 데이터를 확보할 수 있었다.  
<img src=https://github.com/YUYUJIN/animalPoseTrain/blob/main/images/catdog.png></img>  

## How to processing
데이터가 폴더 별로(카테고리, 행동, 영상 종류) json파일로 라벨링 되어있다.  
이를 이용하여 키포인트는 각 카테고리 별(고양이, 강아지)만 구별하여 합쳐 진행하였다.  
LSTM 데이터 같은 경우에는 아래 후술.

## Fast-RCNN Data
Keypoint와 Bounding Box, 객체 분류를 위한 Fast-RCNN 모델의 학습데이터는 위에 공통 전처리된 데이터 위주로 구성하였다.  
하지만 고양이와 강아지의 데이터가 불균형하였다. 이미 데이터는 학습에 충분한 데이터량을 확보하였다고 판단하여 상대적으로 적은 강아지 데이터의 개수로 맞추었다.  
고양이 데이터에서 랜덤하게 약 85만개를 삭제하여 고양이와 강아지 데이터를 약 95만개로 통일하였다.  

## LSTM Data
행동 예측하기 위한 데이터를 준비하였다.  
연속된 프레임의 키포인트 정보를 확보하기 위해 영상 종류 별로 데이터를 전처리하였다.  
<img src=https://github.com/YUYUJIN/animalPoseTrain/blob/main/images/lstmdata.png></img>  
라벨은 원본 데이터와 상이하다.(Trouble Shooting 참조)  최종적으로 데이터량은 학습에 충분하다고 판단하여 각 라벨(고양이, 강아지) 별로 가장 적은 행동 데이터량으로 맞추어 데이터 불균형을 해소하였다.  
고양이의 경우에는 행동별로 약 12만개의 데이터를 활용하였고, 강아지에 경우에는 행동별로 약 13만개의 데이터를 활용하였다.  
LSTM 모델은 입력 시퀀스 크기를 5로 하고 입력 차원은 키포인트의 개수인 15가 각각 x,y 좌표를 가지고 있어 30차원을 사용한다.  
영상 데이터가 약 5-6프레임마다 기록되어있어 이 데이터를 5개씩 연속되게 묶어 하나의 데이터로 사용하였다. 최종 결과물에서 초당 30 프레임을 사용할 예정이기에 데이터는 1초 간격의 행동 데이터로 완성하였다.  
<img src=https://github.com/YUYUJIN/animalPoseTrain/blob/main/images/dataqueue.png></img>  

## Trouble Shooting
<details>
<summary>라벨 조정</summary>

데이터 학습 결과 성능이 예상보다 낮게 나왔다. 이를 보강하기 위해 라벨을 조정하는 작업을 진행하였다. 라벨을 아예 제외하거나 의미를 합칠 수 있는 단위로 병합하였다.  
고양이 행동은 총 12개였으나 7개로 축소하였다.
허리를 아치로 세움은 데이터가 상대적으로 많이 적어 삭제하였고, 앞발로 꾹꾹 누름과 머리를 들이댐, 좌우로 둥굴음은 놀고 있음으로 병합하였다. 또한 옆으로 누워있음, 납작 엎드림, 배를 깜은 쉬고 있음으로 병합하였다.  
강아지 행동 중 머리를 들이댐과 빙글빙글 돈다는 다른 행동과 중복되는 데이터가 많아 삭제하였다. 이후 두발을 듬과 한발을 듬을 발을 듬으로, 배와 목을 보여주며 누움과 엎드리기를 쉬고 있음으로 병합하였다.  
최종 결과는 위 LSTM Data 분포와 같다.
</details>
<details>
<summary>키포인트 정규화</summary>

Fast-RCNN 모델을 이용한 Keypoint 검출에서는 Keypoint를 정규화하지 않아도 충분히 학습이 일어나 높은 정확도를 보였지만, LSTM에서는 충분히 좋은 결과가 나타나지 않았다.  
이를 확인하기 위해 테스트를 진행한 결과 특정 위치에 대해 결과가 의존하게 되는 것을 확인하였고, 이를 해결하기 위해 위치와 무관한 분포를 데이터로 활용하기로 하였다. 이를 위해 min-max 방식을 이용해 각각의 하나의 프레임 영상에서 검출된 keypoint에 대해 x,y 좌표를 정규화하였다.  
<img src=https://github.com/YUYUJIN/animalPoseTrain/blob/main/images/normalize.png></img>  
최종적으로 결과가 위치에 종속되어 특정 위치에서 특정 결과가 강해지는 경우는 축소되었다.  
</details>
<details>
<summary>Human Check</summary>

Keypoint, Bounding Box는 팀원들이 일부 확인해본 결과 부정확한 데이터가 많이 포함되있지 않았다. 문제는 행동 분류에서 나타났는데, 데이터가 동영상 데이터이다보니 특정 행동만이 동영상에 포함된 것이 아니라서 동영상으로 데이터는 생성하는 단계에서 각 행동별로 다른 행동 데이터가 일부 섞이는 현상이 발생하였다.  
프로젝트 기간이 2주로 짧고, 팀원이 5명이기에 약 10만개의 데이터를 확인하기에는 시간이 부족해 해결하지 못했다.  
의미를 확인하는 작업인 만큼 사람이 확인하는 작업이 들어야했고, 시간 상 작업하지 못해 후에 LSTM 정확도에도 영향을 미쳐 약 60%의 정확도만 달성할 수 있었다.
</details>