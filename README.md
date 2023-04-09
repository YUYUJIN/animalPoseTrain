# animalPoseTrain
> 고양이와 강아지의 행동 데이터를 키포트인 검출을 위한 Fast-RCNN과 행동 분류를 위한 LSTM 모델의 학습 데이터로 전처리  
> Fast-RCNN 모델을 학습시켜 고양이와 강아지를 분류하고 키포인트 검출  
> 검출된 정보를 바탕으로 반려동물의 행동을 분류하는 모델 학습  

## Train Data
<img src=https://github.com/YUYUJIN/animalPoseTrain/blob/main/images/aihub.jpg></img>  
link: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=59  
데이터로는 AIHub의 반려동물 구분을 위한 동물 영상 데이터를 사용하였다.  

## Pytorch
Fast-RCNN 모델과 LSTM 모델을 사용하기 위해 pytorch를 설치한다.  
환경에 맞게 설치하되, 본 프로젝트의 개발 환경에 맞는 설치는 다음과 같다.
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```  

## Inforamtion
자세한 내용은 각 프로세스별 폴더 참조.  

## Reference
반려동물 구분을 위한 동물 영상: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=59  
keypoint_rcnn_training_pytorch: https://github.com/alexppppp/keypoint_rcnn_training_pytorch  
Animal-Keypoint-Estimation: https://github.com/thomasreynolds4881/Animal-Keypoint-Estimation  
Pytorch를 활용한 Timeseries 예측모델(1) - LSTM: https://eunhye-zz.tistory.com/entry/Pytorch%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-Timeseries-%EC%98%88%EC%B8%A1-%EB%AA%A8%EB%8D%B81-LSTM  
Long Short-Term Memory (LSTM) network with PyTorch: https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/  
Pytorch Learning Rate Scheduler (러닝 레이트 스케쥴러) 정리: https://gaussian37.github.io/dl-pytorch-lr_scheduler/  

## Produced by 푸루주투
<img src=https://github.com/YUYUJIN/animalPoseTrain/blob/main/images/logo.png style="width:100px; height:100px;"></img>  
team. 푸루주투