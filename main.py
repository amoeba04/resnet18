import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import timm
import random
import numpy as np
import argparse
import wandb
import os


def get_args_parser():
    parser = argparse.ArgumentParser('ResNet-18 training script', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--augment_type', default='0', choices=['0', '1', '2', '3'],
                        type=str, help='Automatic Augmentation type, 0:default, 1:tawide, 2:augmix, 3:randaug')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--lr_min', type=float, default=1e-4, help='Minimum learning rate')
    parser.add_argument('--lr_scale', default='0', choices=['0', '1'], help='Learning rate scaling, 0:False, 1:True')
    parser.add_argument('--lr_sched_type', default='0', choices=['0', '1'], type=str, help='Learning rate scheduler, 0:Step, 1:Cosine')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--optimizer_type', default='0', choices=['0', '1', '2'],
                        type=str, help='Optimizer type, 0:sgd, 1:adam, 2:adamw')
    
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--wandb', action='store_true', help='Log using wandb')
    parser.add_argument('--project', default='DL-23.1', type=str, help='Project name of wandb')
    parser.add_argument('--id', default='resnet18s', type=str, help='Experiment ID of wandb')
    
    return parser

parser = argparse.ArgumentParser('ResNet-18 training script', parents=[get_args_parser()])
args = parser.parse_args()
print(args)

wandb_name = 'b{}ep{}ag{}l{}ls{}lsch{}labs{}wd{}opt{}s{}'.format(args.batch_size, args.epochs, args.augment_type, args.lr, args.lr_scale, args.lr_sched_type, args.label_smoothing, args.weight_decay, args.optimizer_type, args.seed)

if args.wandb:
    run = wandb.init(project=args.project, id=wandb_name, name=wandb_name)
    wandb.config = args

# fix random seed
seed_number = args.seed
random.seed(seed_number)
np.random.seed(seed_number)
torch.manual_seed(seed_number)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: Cutmix/Mixup?
# Define the data transforms
transform_train_new = []
if args.augment_type == '1':
    print('TrivialAugment(Wide) Added.')
    transform_train_new.append(transforms.TrivialAugmentWide())
elif args.augment_type == '2':
    print('AugMix Added.')
    transform_train_new.append(transforms.AugMix())
elif args.augment_type == '3':
    print('RangAug Added.')
    transform_train_new.append(transforms.RandAugment())
else:
    print('Classic data augmentation only')

transform_train_base = [transforms.Resize(256),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip()]
transform_train_norm = [transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform_train_base.extend(transform_train_new)
transform_train_base.extend(transform_train_norm)

transform_train = transforms.Compose(transform_train_base)
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=16)

# Define the ResNet-18 model with pre-trained weights
model = timm.create_model('resnet18', pretrained=True, num_classes=10)
model = torch.compile(model)    # pytorch 2.0 feature
torch.set_float32_matmul_precision('high')
model = model.to(device)  # Move the model to the GPU

# Define the loss function
criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

if args.lr_scale == '1':
    lr = args.lr*args.batch_size/256
else:
    lr = args.lr
    
# Define the optimizer
if args.optimizer_type == '0':
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
elif args.optimizer_type == '1':
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
elif args.optimizer_type == '2':
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    
# Define the learning rate scheduler
if args.lr_sched_type == '0':
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
elif args.lr_sched_type == '1':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)

def train(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # Move the input data to the GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0   
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    
    return accuracy, loss.item()
            
def test(epoch):
    model.eval()
    
    # Test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # Move the input data to the GPU
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('[Epoch %d] Accuracy of the network on the 10000 test images: %f %%' % (epoch+1, 100 * correct / total))
    
    return accuracy, loss.item()

# Train the model
for epoch in range(args.epochs):
    train_acc, train_loss = train(epoch)
    test_acc, test_loss = test(epoch)
    scheduler.step()
    if args.wandb:
        wandb.log({'Train_acc':train_acc, 'Train_loss':train_loss, 'Test_acc':test_acc, 'Test_loss':test_loss})

print('Finished Training')

# Save the checkpoint of the last model
PATH = os.path.join('/raid/jaesin/lecture/Deep_Learning/ckpt', wandb_name+'.pth')
torch.save(model.state_dict(), PATH)