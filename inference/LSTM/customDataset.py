import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset,DataLoader
#from model import classifyModel

class LabelsDataset(Dataset):
    def __init__(self,root,seqLen):
        self.root=root
        self.labels=sorted(os.listdir(root))
        self.seqLen=seqLen

    def __getitem__(self,idx):
        json_path=os.path.join(self.root,self.labels[idx])
        x=np.empty((0,30),float)
        y=np.zeros(11)
        with open(json_path,'r') as j:
            json_data=json.load(j)

        for key in json_data.keys():
            for points in json_data[key]:
                kp=[]
                for p in points:
                    kp=np.append(kp,np.array(p))
                x=np.append(x,np.array([kp]),axis=0)
            
            y[int(key)]=1
        
        x=torch.FloatTensor(x)
        y=torch.FloatTensor(y)
        return x,y
    
    def __len__(self):
        return len(self.labels)

# if __name__=='__main__':
#     model=classifyModel(30,10,5,11,1)
#     test=LabelsDataset('D:\cat_test\cat',5)
#     dl=DataLoader(test,batch_size=10)
#     for i in dl:
#         x,y=i
#         output=model(x)
#         print(output)
#         break