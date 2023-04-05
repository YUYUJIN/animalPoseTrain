import cv2
import numpy as np
import albumentations as A
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import functional as F

from animal_keypoint.utils import collate_fn
from customdataset import ClassDataset,train_transform

def visualize(image, labels, bboxes, keypoints, image_original=None, bboxes_original=None, keypoints_original=None, text_option=False):
    fontsize = 18
    keypoints_classes_ids2names = {0:"nose", 1:"center_of_forehead", 2:"corner_of_the_mouth", 3:"center_of_lower_lip", 4:"neck", 5:"front_right_start",
        6:"front_left_leg_start", 7:"front_right_leg_ankle", 8:"front_left_leg_ankle", 9:"right_femur", 10:"left_femur",
        11:"hind_right_leg_ankle", 12:"hind_left_leg_ankle", 13:"tail_start", 14:"tail_tip"}
    label_dict={1:'cat',2:'dog'}


    for bbox in bboxes:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image.copy(), start_point, end_point, (0,255,0), 2)
    
    for kps in keypoints:
        for idx, kp in enumerate(kps):
            image = cv2.circle(image.copy(), tuple(kp), 1, (255,0,0), 10)
            if text_option:
                image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3, cv2.LINE_AA)

    if image_original is None and keypoints_original is None:
        plt.figure(figsize=(40,40))
        plt.imshow(image)

    else:
        for i,bbox in enumerate(bboxes_original):
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            image_original = cv2.rectangle(image_original.copy(), start_point, end_point, (0,255,0), 2)
            image_original = cv2.putText(image_original.copy(),label_dict[labels[i]],start_point,cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3, cv2.LINE_AA)
        
        for kps in keypoints_original:
            for idx, kp in enumerate(kps):
                image_original = cv2.circle(image_original, tuple(kp), 1, (255,0,0), 10)
                if text_option:
                    image_original = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3, cv2.LINE_AA)

        f, ax = plt.subplots(1, 2, figsize=(40, 20))

        ax[0].imshow(image_original)
        ax[0].set_title('Original image', fontsize=fontsize)

        ax[1].imshow(image)
        ax[1].set_title('Transformed image', fontsize=fontsize)
    
    plt.show()

if __name__=='__main__':
    dataset = ClassDataset('D:/test/train', transform=train_transform(), demo=True)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    iterator = iter(data_loader)
    batch = next(iterator)


    # Show some images

    keypoints_classes_ids2names = {0:"nose", 1:"center_of_forehead", 2:"corner_of_the_mouth", 3:"center_of_lower_lip", 4:"neck", 5:"front_right_start",
        6:"front_left_leg_start", 7:"front_right_leg_ankle", 8:"front_left_leg_ankle", 9:"right_femur", 10:"left_femur",
        11:"hind_right_leg_ankle", 12:"hind_left_leg_ankle", 13:"tail_start", 14:"tail_tip"}
            
    image = (batch[0][0].permute(1,2,0).numpy() * 255).astype(np.uint8)
    bboxes = batch[1][0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()

    keypoints = []
    for kps in batch[1][0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
        keypoints.append([kp[:2] for kp in kps])

    image_original = (batch[2][0].permute(1,2,0).numpy() * 255).astype(np.uint8)
    bboxes_original = batch[3][0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()

    keypoints_original = []
    for kps in batch[3][0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
        keypoints_original.append([kp[:2] for kp in kps])

    visualize(image, bboxes, keypoints, image_original, bboxes_original, keypoints_original)