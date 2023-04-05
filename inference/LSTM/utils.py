import torch
import numpy as np
import cv2

def keypointToTensor(keypoints):
    return torch.FloatTensor(normalize_keypoints(keypoints))

def normalize_keypoints(originKeypoints):
    keypoints_x=np.array(originKeypoints)[:,0]
    keypoints_y=np.array(originKeypoints)[:,1]
    x_min=keypoints_x.min()
    y_min=keypoints_y.min()
    x_max=keypoints_x.max()
    y_max=keypoints_y.max()

    keypoints=[]
    for x,y in zip(keypoints_x,keypoints_y):
        coordinate_x=round((x-x_min)/(x_max-x_min),5)
        coordinate_y=round((y-y_min)/(y_max-y_min),5)
        keypoints.append(coordinate_x)
        keypoints.append(coordinate_y)

    return keypoints

def visualize_act(img,labels,bboxes,category):
    if category==1:
        label_dict={0:'REST',
                    1:'SITDOWN',
                    2:'TAILING',
                    3:'WALKRUN',
                    4:'ARMSTRETCH',
                    5:'PLAYIMG',
                    6:'GROOMING',
                    11:'UNKNOWN'}
    else:
        label_dict={0:'REST',
                    1:'SITDOWN',
                    2:'TAILING',
                    3:'WALKRUN',
                    4:'FOOTUP',
                    5:'BODYSCRATCH',
                    6:'BODYSHAKE',
                    11:'UNKNOWN'}

    image=img
    for i,bbox in enumerate(bboxes):
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image.copy(), start_point, end_point, (0,255,0), 2)
        image = cv2.putText(image.copy(),label_dict[labels[i]],start_point,cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3, cv2.LINE_AA)
    return image