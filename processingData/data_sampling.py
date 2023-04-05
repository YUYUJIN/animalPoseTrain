import glob
import os
import shutil
from random import sample

def sampling(size):
    path = 'D:\pet_data_act_label\CAT'
    save_path = 'D:/cat_act_labels_7'
    folders=os.listdir(path)
    for folder in folders:
        print(f'{folder} start')
        os.makedirs(os.path.join(save_path,folder),exist_ok=True)
        files=os.listdir(os.path.join(path,folder))
        for i,idx in enumerate(sample(range(0,len(files)),size)):
            file=files[idx]
            shutil.copy(os.path.join(path,folder,file),os.path.join(save_path,folder,file))
            if i%100==0:
                print(f'{i}/{size} done')
    print('all done')

sampling(140000)
