# Fast-RCNN Train
> 준비한 학습 데이터를 활용하여 keypoint, bounging box, label 검출을 위한 Fast-RCNN 모델 학습을 진행하였다.

## Yolov7-pose vs Fast-RCNN
대부분의 선행 연구로 keypoint와 bounding box를 동시에 검출하는 모델은 사람을 대상으로 하는 연구였다. 따라서 keypoint의 개수가 달라 문제가 발생하였다.  
Yolov7-pose에 경우에도 마찬가지였지만, 구조는 바꾸는 것에는 성공하였으나 loss를 계산하는 부분에서 Yolov7는 단일 라벨만을 사용하여 차원의 차이가 발생하였다. 이 부분을 해결할 수 없어 Pytorch 내에 기본적인 Fast-RCNN 모델을 사용하여 새로 구성하였다.  

## Train
학습 코드를 작성 시에는 두 참고 자료를 활용하였다.  
github: https://github.com/thomasreynolds4881/Animal-Keypoint-Estimation  
github: https://github.com/alexppppp/keypoint_rcnn_training_pytorch  

단일 라벨인 것을 다중 라벨로 바꾸어 구성하여 사용하였다. 나머지는 전처리 완료한 데이터를 사용할 수 있도록 구성과 코드만 수정하여 사용하였다.  

## Trouble Shooting
<(keypoint_train)이미지>  
<details>
<summary>학습 시간 소요</summary>

학습 과정에서 많이 시간이 소요되었고, 개발 기간 상 다양한 형태로 학습을 계속 돌려 중간마다 성능이 높은 모델로 바꾸어 프로젝트를 진행하였다.  
위 이미지처럼 전체 이미지를 돌리는 것에는 너무 많은 시간이 소요되므로 epoch 기준으로 확인할 때는 데이터량을 줄여 4만개의 전체 데이터를 많이 참조하도록하고, 전체 데이터를 사용할 때는 iteration 기준으로 확인하여 많은 종류의 데이터를 학습 데이터로 활용하도록 하였다.
</details>
<details>
<summary>파라미터</summary>

파라미터 선정에서도 많은 시간이 소요되었다. 최종적으로 여러 파라미터를 가지고 동시에 여러 가상머신에서 학습을 진행하였고, 다른 파라미터들은 학습 중간에 갑자기 검출하지 않은 것이 제일 loss 값이 작은 기울기 갇힘 현상이 발생하였다. 최종적으로 위 그림과 같은 파라미터를 선정하였다.
</details>
