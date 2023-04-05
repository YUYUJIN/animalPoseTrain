import os
import shutil

root_path='D:\pet_data_act_label\CAT\LYING'
save_path='D:\pet_data_act_label\CAT\LAYDOWN'

files=os.listdir(root_path)
for i,file in enumerate(files):
    shutil.move(os.path.join(root_path,file),os.path.join(save_path,file))
    print(f'{i}/{len(files)}')