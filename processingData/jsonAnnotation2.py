import os
import shutil
import json

def _copyfileobj_patched(fsrc, fdst, length=16*1024*1024):
    """Patches shutil copyfileobj method to hugely improve copy speed"""
    while 1:
        buf = fsrc.read(length)
        if not buf:
            break
        fdst.write(buf)

shutil.copyfileobj = _copyfileobj_patched  # shutil 의 copyfileobj 대신 _copyfileobj_patched 이 호출됨

label_num={'CAT':0,'DOG':1}

data_root='D:/animal'
main_root='D:\pet_data_detection'

root_folders=os.listdir(data_root)
os.makedirs(main_root,exist_ok=True)
for root_foloder in root_folders:
    print(root_foloder+' start')
    labels=os.listdir(os.path.join(data_root,root_foloder))
    for label in labels:
        save_path_images=os.path.join(main_root,'images')
        save_path_labels=os.path.join(main_root,'labels')
        os.makedirs(save_path_images,exist_ok=True)
        os.makedirs(save_path_labels,exist_ok=True)

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

                        images=os.listdir(os.path.join(data_root,root_foloder,label,act,file))
                        for j,image in enumerate(images):
                            if image.split('.')[-1]=='db':
                                continue
                            frame_number=image.split('_')[1]
                            image_name=file+'_'+str(i)+'_'+str(j)
                            
                            json_labels=label_dict.get(frame_number)
                            if json_labels is not None:
                                if len(json_labels)==16:
                                    new_json=dict()
                                    new_json['bboxes']=[json_labels[0]]
                                    new_json['keypoints']=[json_labels[1:]]
                                    
                                    shutil.copy(os.path.join(data_root,root_foloder,label,act,file,image),os.path.join(save_path_images,image_name+'.jpg'))
                                    new_json_path=os.path.join(save_path_labels,image_name+'.json')
                                    with open(new_json_path,'w') as outjson:
                                        json.dump(new_json,outjson)
                    if i%50==0:
                        print(f'{i}/{len(files)} files done')
                print(act+' done')
        print(label+' done')
    print(root_foloder+' done')