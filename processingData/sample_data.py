from sklearn.model_selection import train_test_split
import glob
import os
import shutil

path = 'D:\pet_data_detection_dog'
copy_path = 'D:\pet'

img_path = glob.glob(os.path.join(path,'images','*'))
mark_path = glob.glob(os.path.join(path,'labels','*'))
tv = ['train','valid']*2
for fol in tv:
    os.makedirs(f'{copy_path}\{fol}',exist_ok=True)

train,val = train_test_split(img_path, test_size=0.2, random_state=7777)
train_json,val_json = train_test_split(mark_path, test_size=0.2, random_state=7777)
img_li = [train,val]
json_li = [train_json,val_json]
for tr_val_img,tr_val_json,tr_val_name in zip(img_li,json_li,tv):
    count = 1
    for file in tr_val_img:
        if count<5001:
            os.makedirs(f'{copy_path}\{tr_val_name}\images', exist_ok=True)
            shutil.copyfile(file, f'{copy_path}\\{tr_val_name}\\images\\' + os.path.basename(file))
            if count%50==0:
                print(f'train images {count}/{len(tr_val_img)}')
            count+=1
        else:
            break

    count = 1
    for file in tr_val_json:
        if count<5001:
            os.makedirs(f'{copy_path}\{tr_val_name}\labels', exist_ok=True)
            shutil.copyfile(file, f'{copy_path}\\{tr_val_name}\\labels\\' + os.path.basename(file))
            if count%50==0:
                print(f'train json {count}/{len(tr_val_json)}')
            count+=1
        else:
            break