import torch
import torch.nn as nn
import torch.optim as optim

from module.model import classifyModel
from module.train import train_model,evaluate
from torch.utils.data import DataLoader
from module.customDataset import LabelsDataset
from matplotlib import pyplot as plt
import numpy as np

EPOCH=100
DEVICE=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LEARNINGRATE=0.01
MODEL_SAVE_PATH='D:/lstm'

print('dataset ready')
model=classifyModel(input_dim=30,hidden_dim=30,seq_len=5,output_dim=7,layers=3)
trainDataset=LabelsDataset('D:\\act_train_labels_cat_7\\train',5)
validDataset=LabelsDataset('D:\\act_train_labels_cat_7\\valid',5)
train_loader=DataLoader(trainDataset,shuffle=True,batch_size=100)
valid_loader=DataLoader(validDataset,shuffle=False,batch_size=100)

print('train ready')
model.to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr = LEARNINGRATE)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)

# epoch마다 loss 저장
train_loss = np.zeros(EPOCH)
train_acc = np.zeros(EPOCH)
valid_loss = np.zeros(EPOCH)
valid_acc = np.zeros(EPOCH)

best_acc=0

print('train start')
for epoch in range(0,EPOCH):
    avg_loss,acc_item=train_model(model=model,device=DEVICE,train_loader=train_loader,criterion=criterion,optimizer=optimizer,epoch=epoch,total_epoch=EPOCH)
    train_loss[epoch],train_acc[epoch]=avg_loss,(acc_item/len(trainDataset))
    lr_scheduler.step()
    print(f'Train [{epoch}/{EPOCH}] result : average loss : {avg_loss} / accuracy : {train_acc[epoch]}')
    val_loss,val_acc_item=evaluate(model=model,device=DEVICE,valid_loader=valid_loader,criterion=criterion)
    valid_loss[epoch],valid_acc[epoch]=val_loss,(val_acc_item/len(validDataset))
    print(f'Valid [{epoch}/{EPOCH}] result : average loss : {val_loss} / accuracy : {valid_acc[epoch]}')

    if epoch%10==0:
        torch.save(model.state_dict(), f'{MODEL_SAVE_PATH}/epoch/lstm_weights_{epoch}.pth')
    if valid_acc[epoch]>best_acc:
        torch.save(model.state_dict(), f'{MODEL_SAVE_PATH}/lstm_weights_best.pth')
        best_acc=valid_acc[epoch]

# fig = plt.figure(figsize=(10, 4))
plt.subplot(2,2,1)
plt.plot(train_loss, label="Training loss")
plt.legend()
plt.subplot(2,2,2)
plt.plot(valid_loss, label="Validation loss")
plt.legend()
plt.subplot(2,2,3)
plt.plot(train_acc, label="Training Accuracy")
plt.legend()
plt.subplot(2,2,4)
plt.plot(valid_acc, label="Validation Accuracy")
plt.legend()
plt.show()