import os
import shutil

total_data_path='D:\pet_data_detection_cat'
save_path='D:/pet_data_detection_dog'

folders=os.listdir(total_data_path)
for folder in folders:
    os.makedirs(os.path.join(save_path,folder),exist_ok=True)
    files=os.listdir(os.path.join(total_data_path,folder))
    for i,file in enumerate(files):
        label=file[:3]
        if label=='dog':
            shutil.move(os.path.join(total_data_path,folder,file),os.path.join(save_path,folder,file))
        
        if i%1000==0:
            print(f'{i}/{len(files)}')