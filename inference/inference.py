import os
import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
from RCNN.model import get_model
from RCNN.utils import visualize
from LSTM.model import LSTMModel
from LSTM.utils import keypointToTensor,visualize_act

DEVICE=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
FRAME_PER_LSTM=6    # frame/seq_len
SEQ_LEN=5           # LSTM seq_len

cap = cv2.VideoCapture('D:/cat_cam1.mp4')
#cap = cv2.VideoCapture('D:/sjh_5.mp4')
print('RCNN Model load')
RCNN_model=get_model(num_keypoints=15,num_classes=3)
RCNN_model.load_state_dict(torch.load('D:/keypointsrcnn_weights_150000.pth'))
RCNN_model.eval()
RCNN_model.to(DEVICE)
print('LSTM Model load')
LSTM_cat=LSTMModel(input_dim=30,hidden_dim=30,seq_len=5,output_dim=7,layers=3)
LSTM_cat.load_state_dict(torch.load('D:\lstm\lstm_weights_best_cat.pth'))
LSTM_cat.eval()
LSTM_cat.to(DEVICE)
LSTM_dog=LSTMModel(input_dim=30,hidden_dim=30,seq_len=5,output_dim=8,layers=3)
LSTM_dog.load_state_dict(torch.load('D:\lstm\lstm_weights_best_dog.pth'))
LSTM_dog.eval()
LSTM_dog.to(DEVICE)

# data about frame
frame_queue=None
frame_store=None
keypointPerFrame=None
act_label=11
category_labels=[]
category_label_queue=[]
category_id=1   # 최초 unknown 아무거나 상관없음

print('Test start')
while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("카메라를 찾을 수 없습니다.")
            break
        image_src=cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            image_src=F.to_tensor(image_src).to(DEVICE).unsqueeze(0)
            rcnn_outputs=RCNN_model(image_src)
            #print(rcnn_outputs)

            if rcnn_outputs[0]['boxes'].shape[0]==0:
                 pass
            else:
                scores = rcnn_outputs[0]['scores'].detach().cpu().numpy()

                high_scores_idxs = np.where(scores > 0.75)[0].tolist() # Indexes of boxes with scores > 0.7
                if len(high_scores_idxs)==0:
                     pass
                else:
                    high_scores_idxs = np.where(max([scores[idx] for idx in high_scores_idxs]))[0].tolist()
                    post_nms_idxs = torchvision.ops.nms(rcnn_outputs[0]['boxes'][high_scores_idxs], rcnn_outputs[0]['scores'][high_scores_idxs], 0.3).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)
                    
                    labels=[]
                    for lbs in rcnn_outputs[0]['labels'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                        labels.append(lbs)

                    keypoints = []
                    for kps in rcnn_outputs[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                        keypoints.append([list(map(int, kp[:2])) for kp in kps])

                    bboxes = []
                    for bbox in rcnn_outputs[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                        bboxes.append(list(map(int, bbox.tolist())))
                    # if frame_queue is None:
                    #      frame_queue=torch.Tensor(np.array(keypoints[0]))
                    # print(frame_queue)
                    #image=visualize(img=image,labels=labels,bboxes=bboxes,keypoints=keypoints,keypoint_option=True,text_option=False)

                    category_labels.append(labels[0])
                    if len(category_labels)==FRAME_PER_LSTM:
                        category_label_queue.append(2 if sum(category_labels)>8 else 2)
                    if len(category_label_queue)==SEQ_LEN:
                        category_id=1 if sum(category_label_queue)<7 else 2
                        del category_label_queue[0]

                    keypoints=keypointToTensor(keypoints[0]).to(DEVICE).unsqueeze(0)
                    if frame_store is None:
                        frame_store=keypoints
                    else:
                        frame_store=torch.cat([frame_store,keypoints],axis=0)
                        if frame_store.shape[0]==FRAME_PER_LSTM:
                            keypointPerFrame=frame_store.mean(dim=0).unsqueeze(0)
                            frame_store=None
                    if keypointPerFrame is not None:
                        if frame_queue is None:
                            frame_queue=keypointPerFrame
                        else:
                            frame_queue=torch.cat([frame_queue,keypointPerFrame],axis=0)
                            keypointPerFrame=None
                            if frame_queue.shape[0]>SEQ_LEN:
                                frame_queue=frame_queue[frame_queue.shape[0]-5:frame_queue.shape[0],:]

                                if labels[0]==1:
                                    lstm_outputs=LSTM_cat(frame_queue.unsqueeze(0))
                                    act_label=torch.argmax(lstm_outputs).item()
                                else:
                                    lstm_outputs=LSTM_dog(frame_queue.unsqueeze(0))
                                    act_label=torch.argmax(lstm_outputs).item()

                    image=visualize_act(image,labels=[act_label],bboxes=bboxes,category=category_id)

        winname='test'
        cv2.namedWindow(winname)
        cv2.moveWindow(winname, 40, 30) 
        cv2.imshow(winname, image)
        cv2.waitKey(1)
cap.release()