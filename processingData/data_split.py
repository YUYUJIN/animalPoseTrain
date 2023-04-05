from sklearn.model_selection import train_test_split
import glob
import os
import shutil

path = 'D:\cat_act_labels_7'
copy_path = 'D:\\act_train_labels_cat'
tv = ['train','valid']
for fol in tv:
    os.makedirs(f'{copy_path}\{fol}',exist_ok=True)

folders=os.listdir(path)
for folder in folders:
    file_paths = glob.glob(os.path.join(path,folder,'*'))
    train,val = train_test_split(file_paths, test_size=0.2, random_state=7777)
    data_li = [train,val]
    for tr_val_data,tr_val_name in zip(data_li,tv):
        count = 1
        for file in tr_val_data:
            shutil.copyfile(file, f'{copy_path}\\{tr_val_name}\\' + os.path.basename(file))
            if count%100==0:
                print(f'train datas({folder}) {count}/{len(tr_val_data)}')
            count+=1