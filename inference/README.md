# Inference
> 본 프로젝트에서 구성하고 학습시킨 Fast-RCNN 모델과 LSTM 모델을 사용하여 입력 동영상에 대해 테스트를 진행하고 최종 프로젝트에 이식하기 위한 준비 코드이다.  
> 실행코드는 각각의 train 코드에서 모델을 사용하는 부분만을 분리하여 동영상 입력에 대해 작동하도록 수정하였다.

## Fast-RCNN Keypoint Model
<img src=https://github.com/YUYUJIN/animalPoseTrain/blob/main/images/fast_rcnn_result.png></img>  
학습된 Fast-RCNN 모델의 결과이다.

## LSTM Action Classify Model
<img src=https://github.com/YUYUJIN/animalPoseTrain/blob/main/images/lstm_result.png></img>  
학습된 LSTM 모델의 결과이다.

## Trouble Shooting
<details>
<summary>동영상 적용</summary>

기존의 학습 데이터는 프레임 당 데이터여서 묶어서 처리하면 최종 데이터를 구성할 수 있었다. 하지만 실제 데이터는 프레임 당 계속해서 발생하므로 이를 평균내어 동작 저장하고 Queue 구조를 이용하여 새로운 데이터가 들어오면 기존의 데이터를 밀어내고 Queue 내의 데이터 전체가 모델의 입력으로 사용되는 방식으로 구성하여 해결하였다.
<(lstm_queue)이미지>
</details>