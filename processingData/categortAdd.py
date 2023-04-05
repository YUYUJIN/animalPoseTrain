import os
import json

label_dict={'cat':1,'dog':2}

data_root='D:\pet_test\\train\labels'

json_files=os.listdir(data_root)
for json_file in json_files:
    json_path=os.path.join(data_root,json_file)
    with open(json_path,'r') as j:
        json_data=json.load(j)
    
    new_json_data=dict()
    new_json_data['category_id']=label_dict[json_file[:3]]
    new_json_data['bboxes']=json_data['bboxes']
    new_json_data['keypoints']=json_data['keypoints']
    
    with open(json_path,'w') as j:
        json.dump(new_json_data,j)
