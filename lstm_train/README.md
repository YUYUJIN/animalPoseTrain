# LSTM Train
> 준비한 학습 데이터를 활용하여 keypoint를 이용해 행동을 분류하는 LSTM 모델 학습을 진행하였다.

## Why LSTM
행동을 분류하기 위해서는 영상의 단편적인 정보만으로는 판단이 불가능하다.  
<img src=https://github.com/YUYUJIN/animalPoseTrain/blob/main/images/wlstm1.png></img>  
위와 같이 하나의 프레임으로는 판단이 불가하므로 연속된 프레임을 모두 확인하여야한다.  
<img src=https://github.com/YUYUJIN/animalPoseTrain/blob/main/images/wlstm2.png></img>  
이를 구현하기 위해서 keypoint를 입력으로 사용하는 rnn 구조를 고려하였고, 그 중에서 여러 시퀀스에 입력이 있을 때 이전 값의 전파가 기억되어 좋은 성능을 보여주는 LSTM 모델을 사용하기로 하였다. 앞 부분에 나온 시퀀스가 해당 행동의 큰 특징이 될 수도 있기에 이전 시퀀스를 잘 반영할 수 있는 LSTM이 최적이라 판단하였다.

## Structure
<img src=https://github.com/YUYUJIN/animalPoseTrain/blob/main/images/lstm_s.PNG></img>  
최종적으로 위와 같은 구조를 가지는 모델을 설계하였다.
각 셀에 입력되는 데이터는 x,y 좌표를 가지는 15개의 keypoint이므로 30차원의 데이터를 사용하였다. 데이터가 각각 5-6 프레임마다의 keypoint이고, 구동 환경에서 초당 30프레임의 데이터를 사용하기로 하였으므로 1초마다 행동을 판별하기 위해서는 5개의 keypoint 데이터를 입력으로 사용한다.  
최종적으로 5의 시퀀스를 입력을 받고, 각각의 시퀀스는 30차원으로 이루어진 깊이가 3차원인 모델을 구성하였다. 깊이의 경우에는 차원을 높여도 성능이 좋아지지 않아 경험적 테스트 상 최적의 값이 3을 기준으로 설정하였다. 추가로 최종 레이어의 입력 차원을 확대하여모든 시퀀스에 대한 h 값을 입력 값으로 사용하여 모델을 구성하였다.

## Train
학습 코드를 작성 시에는 아래 참고 자료를 활용하였다.  
blog: https://eunhye-zz.tistory.com/entry/Pytorch%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-Timeseries-%EC%98%88%EC%B8%A1-%EB%AA%A8%EB%8D%B81-LSTM  
link: https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/  

강아지와 고양이 행동이 다른 라벨이 존재하므로 두 모델을 각각 생성하여 학습을 진행하였다.  
모델의 구조가 상이한 부분만 수정하여 사용하였고, 코드 자체가 참고 코드들이 객체 형식이 아니라서 추후 프로젝트에서 이식 시에 문제가 될 것 같아 객체 형식으로 바꾸어 사용하였다.  
각 시퀀스마다 이후 시퀀스에 영향을 주지 않도록 셀 사이에 전파되는 h 값을 초기화하는 부분을 제외하면 기존 rnn에서 셀만 바꾸는 것과 크게 다르지 않았다.  
최종적으로 프로젝트에 맞게 코드를 수정하여 학습을 진행하였다.  

## Trouble Shooting
<details>
<summary>학습률 조정</summary>

최적의 학습률을 찾을 수가 없었다. 따라서 학습률를 학습 진행 시에 변화하도록하는 learning scheduler를 사용하기로 하였다.  
gitblog: https://gaussian37.github.io/dl-pytorch-lr_scheduler/  
<img src=https://github.com/YUYUJIN/animalPoseTrain/blob/main/images/lstm_t.png></img>  
최종적으로 step 함수를 이용하여 높은 학습률부터 시작하여 특정 구간마다 학습률을 떨어뜨리는 방식으로 학습을 진행하였고, 정확도를 향상할 수 있었다.
</details>
<details>
<summary>정확도 낮음</summary>

정확도가 60% 이상 올라가지 않아 여러 확인을 거쳐 새로 학습도 진행하였지만, 최종적으로 데이터 검수 자체가 잘못되었다는 것을 확인하였다. 데이터량이 많아 사람이 하나씩 검수하지 못해 동영상 데이터 내 특정 행동 외에도 다른 행동이 일부 존재하고 그 양이 프레임 단위로 데이터를 생성할 때 아예 다른 행동이 목표 행동이 되는 경우가 존재하였다. 많은 양에 데이터이고 대여 받은 가상머신이 만료됨에 따라 데이터 수정이 불가능하여 검증하지 못하였다.
</details>