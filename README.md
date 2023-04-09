# animalPoseTrain
> 고양이와 강아지의 행동 데이터를 키포트인 검출을 위한 Fast-RCNN과 행동 분류를 위한 LSTM 모델의 학습 데이터로 전처리  
> Fast-RCNN 모델을 학습시켜 고양이와 강아지를 분류하고 키포인트 검출  
> 검출된 정보를 바탕으로 반려동물의 행동을 분류하는 모델 학습  

## Train Data
<(aihub)이미지>  
link: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=59  
데이터로는 AIHub의 반려동물 구분을 위한 동물 영상 데이터를 사용하였다.  

## Pytorch
Fast-RCNN 모델과 LSTM 모델을 사용하기 위해 pytorch를 설치한다.  
환경에 맞게 설치하되, 본 프로젝트의 개발 환경에 맞는 설치는 다음과 같다.
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```  

## Reference
