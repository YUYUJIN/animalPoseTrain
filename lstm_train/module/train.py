import sys
import torch
from tqdm import tqdm

def train_model(model, train_loader, device, criterion,optimizer,epoch,total_epoch):
    model.train()
    avg_cost=0
    correct_item=0
    total_batch=len(train_loader)

    train_bar=tqdm(train_loader,file=sys.stdout,colour='red')
    for step,samples in enumerate(train_bar):
        x_train,y_train=samples
        x_train,y_train=x_train.to(device),y_train.to(device)

        model.reset_hidden_state()

        outputs=model(x_train)

        loss=criterion(outputs,y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avg_cost+=loss/total_batch
        correct_item+=(torch.argmax(outputs,dim=1)==torch.argmax(y_train,dim=1)).sum().item()
        train_bar.desc=f'Train epoch[{epoch}/{total_epoch}] step[{step}/{len(train_bar)}] loss[{loss.data:.3f}]'

    return avg_cost,correct_item

def evaluate(model,device,valid_loader,criterion):
    model.eval()
    avg_cost=0
    correct_item=0
    total_batch=len(valid_loader)

    valid_bar=tqdm(valid_loader,file=sys.stdout,colour='red')
    with torch.no_grad():
        for step,samples in enumerate(valid_bar):
            x_valid,y_valid=samples
            x_valid,y_valid=x_valid.to(device),y_valid.to(device)

            model.reset_hidden_state()

            outputs=model(x_valid)

            loss=criterion(outputs,y_valid)

            avg_cost=loss/total_batch
            correct_item+=(torch.argmax(outputs,dim=1)==torch.argmax(y_valid,dim=1)).sum().item()
            valid_bar.desc=f'Valid step[{step}/{len(valid_bar)}] loss[{loss.data:.3f}]'
    
    return avg_cost,correct_item