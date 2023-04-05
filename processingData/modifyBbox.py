import os
import json

label_dict={'cat':1,'dog':2}

data_root='D:\pet_test\\train\labels'

json_files=os.listdir(data_root)
for i,json_file in enumerate(json_files):
    json_path=os.path.join(data_root,json_file)
    with open(json_path,'r') as j:
        json_data=json.load(j)

    bboxes=json_data['bboxes'][0]
    new_bboxes=[[bboxes[0],bboxes[1],bboxes[0]+bboxes[2],bboxes[1]+bboxes[3]]]

    new_json_data=dict()
    new_json_data['category_id']=label_dict[json_file[:3]]
    new_json_data['bboxes']=new_bboxes
    new_json_data['keypoints']=json_data['keypoints']
    
    with open(json_path,'w') as j:
        json.dump(new_json_data,j)

    if i%1000==0:
        print(f'{i}/{len(json_files)} done')