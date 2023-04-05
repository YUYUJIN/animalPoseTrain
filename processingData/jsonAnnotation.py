import os
import json

data_root='D:/animal'

root_folders=os.listdir(data_root)
for root_foloder in root_folders:
    print(root_foloder+' start')
    labels=os.listdir(os.path.join(data_root,root_foloder))
    for label in labels:
        print(label+' start')
        acts=os.listdir(os.path.join(data_root,root_foloder,label))
        for act in acts:
            if len(act.split('.'))==1:
                print(act+' start')
                files=os.listdir(os.path.join(data_root,root_foloder,label,act))
                for i,file in enumerate(files):
                    if len(file.split('.'))==1:
                        json_path = os.path.join(data_root,root_foloder,label,act,file+'.json')
                        with open(json_path, 'r', encoding='utf-8') as f:
                            label_dict = json.load(f)
                        annotations=label_dict['annotations']

                        new_json=dict()
                        for annotation in annotations:
                            w=annotation['bounding_box']['width']
                            h=annotation['bounding_box']['height']
                            x=annotation['bounding_box']['x']
                            y=annotation['bounding_box']['y']
                            value=[[x,y,w,h]]
                            
                            keypoint=annotation['keypoints']
                            for num in keypoint.keys():
                                if keypoint[num]==None:
                                    break
                                value.append([keypoint[num]['x'],keypoint[num]['y'],1])
                            if len(value)==16:
                                new_json[annotation['frame_number']]=value
                        new_json_path=os.path.join(data_root,root_foloder,label,act,file+'_m.json')
                        with open(new_json_path,'w') as outjson:
                            json.dump(new_json,outjson,indent=4)
                    if i%50==0:
                        print(f'{i}/{len(files)} files done')
                print(act+' done')
        print(label+' done')
    print(root_foloder+' done')