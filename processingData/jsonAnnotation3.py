import os
import json

category_id={'cat':0,'dog':1}
annotations_json=dict()
images=dict()
annotations=[]

data_root='D:/test'

root_folders=os.listdir(data_root)

for root_folder in root_folders:
    labels=os.listdir(os.path.join(data_root,root_folder,'labels'))

    for i,label in enumerate(labels):
        image_id=i+1
        image_name=label.replace('.json','.jpg')
        images[str(image_id)]=image_name
        category=label.split('-')[0]

        label_json_path=os.path.join(data_root,root_folder,'labels',label)
        with open(label_json_path,'r') as j:
            label_data=json.load(j)
        annotation={'image_id':image_id,'bbox':label_data['bboxes'][0],'keypoints':label_data['keypoints'][0],'num_keypoints':15,'category_id':category_id[category]}
        annotations.append(annotation)

annotations_json['images']=images
annotations_json['annotations']=annotations

save_json=os.path.join(data_root,'annotations.json')
with open(save_json,'w') as outjson:
    json.dump(annotations_json,outjson)
    


