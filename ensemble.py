import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import timm
import argparse
import os

def get_args_parser():
    parser = argparse.ArgumentParser('ResNet-18 ensemble evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=3, type=int)
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
    
    parser.add_argument('--model_num', default=2, type=int)
    
    return parser

parser = argparse.ArgumentParser('ResNet-18 training script', parents=[get_args_parser()])
args = parser.parse_args()
print(args)

model_name = 'b{}ep{}ag{}l{}ls{}lsch{}labs{}wd{}opt{}'.format(args.batch_size, args.epochs, args.augment_type, args.lr, args.lr_scale, args.lr_sched_type, args.label_smoothing, args.weight_decay, args.optimizer_type)
model_path = os.path.join('/raid/jaesin/lecture/Deep_Learning/ckpt', model_name)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define the data transforms
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.TenCrop(224),
    transforms.Lambda(lambda crops: torch.stack([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(crop)) for crop in crops]))
])
# Load the CIFAR-10 test dataset
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=16)

# Define the list of models for ensemble
models = []
for i in range(args.model_num):
    # Define the ResNet-18 model with pre-trained weights
    model = timm.create_model('resnet18', num_classes=10)
    model = torch.compile(model)
    model.load_state_dict(torch.load(model_path+'s{}.pth'.format(i)))  # Load the trained weights
    model.eval()  # Set the model to evaluation mode
    model = model.to(device)  # Move the model to the GPU
    models.append(model)

# Evaluate the ensemble of models
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)  # Move the input data to the GPU
        bs, ncrops, c, h, w = images.size()       
        outputs = torch.zeros(bs, 10).to(device)  # Initialize the output tensor with zeros
        for model in models:
            model_output = model(images.view(-1, c, h, w))  # Reshape the input to (bs*10, c, h, w)
            model_output = model_output.view(bs, ncrops, -1).mean(1)  # Average the predictions of the 10 crops
            outputs += model_output
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the ensemble on the 10000 test images: %f %%' % (100 * correct / total))