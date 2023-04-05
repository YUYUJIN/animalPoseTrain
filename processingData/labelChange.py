import os
import json

# cat new
label_match={"0":"1",
             "1":"0",
             "2":"6",
             "3":"2",
             "4":"0",
             "5":"6",
             "6":"6",
             "7":"0",
             "8":"3",
             "9":"4",
             "10":"5"}

data_path='D:/act_train_labels_cat'
save_path='D:/act_train_labels_cat_7'
folders=os.listdir(data_path)
for folder in folders:
    print(folder+' start')
    os.makedirs(os.path.join(save_path,folder))
    files=os.listdir(os.path.join(data_path,folder))
    for i,file in enumerate(files):
        json_path=os.path.join(data_path,folder,file)
        with open(json_path,'r') as j:
            data=json.load(j)

        new_json=dict()
        for key in data.keys():
            new_json[label_match[key]]=data[key]
        
        with open(os.path.join(save_path,folder,file),'w') as j:
            json.dump(new_json,j)
        
        print(f'{i}/{len(files)} done')
