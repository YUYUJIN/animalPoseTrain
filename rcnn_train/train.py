import torch
from model import get_model
from torch.utils.data import DataLoader

from animal_keypoint.utils import collate_fn
from animal_keypoint.engine import train_one_epoch,evaluate
from customdataset import ClassDataset,train_transform

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

KEYPOINTS_FOLDER_TRAIN = 'D:\pet_test\\train'
KEYPOINTS_FOLDER_TEST = 'D:\pet_test\\valid'

dataset_train = ClassDataset(KEYPOINTS_FOLDER_TRAIN, transform=train_transform(), demo=False)
dataset_test = ClassDataset(KEYPOINTS_FOLDER_TEST, transform=None, demo=False)

data_loader_train = DataLoader(dataset_train, batch_size=4, shuffle=True, collate_fn=collate_fn)
data_loader_test = DataLoader(dataset_test, batch_size=4, shuffle=False, collate_fn=collate_fn)

model = get_model(num_keypoints = 15,num_classes=3)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
num_epochs = 100

checkout=True
# Run model
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=100)
    lr_scheduler.step()
    evaluate(model, data_loader_test, device)

    if epoch%5==0:
        torch.save(model.state_dict(), f'keypointsrcnn_weights_{epoch}.pth')
    
# Save model weights after training
torch.save(model.state_dict(), 'keypointsrcnn_weights_final.pth')