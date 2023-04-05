import os
import json
import numpy as np

act_labels_cat={'ARMSTRETCH':0,
    'FOOTPUSH':1,
    'GETDOWN':2,
    'GROOMING':3,
    'HEADING':4,
    'LAYDOWN':5,
    'LYING':6,
    'ROLL':7,
    'SITDOWN':8,
    'TAILING':9,
    'WALKRUN':10}

act_labels_dog={'BODYLOWER':0,
    'BODYSCRATCH':1,
    'BODYSHAKE':2,
    'FEETUP':3,
    'HEADING':4,
    'FOOTUP':5,
    'LYING':6,
    'SIT':7,
    'TURN':8,
    'TAILING':9,
    'WALKRUN':10}

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
        keypoints.append([coordinate_x,coordinate_y])

    return keypoints

label_num={'CAT':0,'DOG':1}

data_root='D:/animal'
main_root='D:\pet_data_act_label'

root_folders=os.listdir(data_root)
for root_foloder in root_folders:
    if root_foloder=='Training':
        continue
    print(root_foloder+' start')
    labels=os.listdir(os.path.join(data_root,root_foloder))
    for label in labels:
        print(label+' start')
        acts=os.listdir(os.path.join(data_root,root_foloder,label))
        for act in acts:
            save_path=os.path.join(main_root,label,act)
            os.makedirs(save_path,exist_ok=True)
            if len(act.split('.'))==1:
                print(act+' start')
                files=os.listdir(os.path.join(data_root,root_foloder,label,act))
                for i,file in enumerate(files):
                    labelAct=act_labels_cat.get(act)
                    if labelAct is None:
                            labelAct=act_labels_dog.get(act)
                    if labelAct is None:
                        break

                    if len(file.split('.'))==1:
                        data_queue=[]
                        json_path = os.path.join(data_root,root_foloder,label,act,file+'_m.json')
                        with open(json_path, 'r', encoding='utf-8') as f:
                            label_dict = json.load(f)
                        
                        for k,key in enumerate(label_dict.keys()):
                            framePerLabel=label_dict[key]
                            normalizedKeypoints=normalize_keypoints(framePerLabel[1:])
                            data_queue.append(normalizedKeypoints)
                            if len(data_queue)==5:
                                json_data={labelAct:data_queue}
                                with open(os.path.join(save_path,file+f'_{i}_{k}.json'),'w') as j:
                                    json.dump(json_data,j)
                                del data_queue[0]
                            
                    if i%50==0:
                        print(f'{i}/{len(files)} files done')
                print(act+' done')
        print(label+' done')
    print(root_foloder+' done')