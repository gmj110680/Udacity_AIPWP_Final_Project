# imports
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from train_functions import get_input_args
import sys

# load command line arguments
in_arg = get_input_args()

# define dataset variables
data_dir = in_arg.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# dataset transforms
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = valid_transforms

# Define the datasets 
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)

valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# Define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64)

validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

# Activate GPU 
if in_arg.gpu == 'gpu':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'

# Import model
if in_arg.arch.lower() == 'alexnet':
    model = models.alexnet(pretrained=True)
    input_value = 9216
elif in_arg.arch.lower() == 'resnet':
    model = models.resnet18(pretrained=True)
    input_value = 512
else:
    model = models.vgg16(pretrained=True)
    input_value = 25088

# Freeze model parameters
for param in model.parameters():
    param.requires_grad=False

# Build the classifier
new_classifier = nn.Sequential(nn.Linear(input_value, in_arg.hidden_units_1),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(in_arg.hidden_units_1, in_arg.hidden_units_2),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(in_arg.hidden_units_2, len(train_data.class_to_idx)),
                               nn.LogSoftmax(dim=1))

# Attach classifier to model
if in_arg.arch.lower() == 'alexnet' or in_arg.arch.lower() == 'vgg':
    model.classifier = new_classifier
else:
    model.fc = new_classifier

# Calculate loss
criterion = nn.NLLLoss()

# Train classifier parameters
if in_arg.arch.lower() == 'resnet':
    optimizer = optim.SGD(model.fc.parameters(), lr = in_arg.learning_rate)
else:
    optimizer = optim.SGD(model.classifier.parameters(), lr = in_arg.learning_rate)

# Feed model to device
model.to(device)

# Train the classifier
epochs = in_arg.epochs
steps = 0
running_loss = 0
print_every = 8

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        
        # move inputs labels to gpu
        inputs, labels = inputs.to(device), labels.to(device)
        
        # clear out gradient descent data
        optimizer.zero_grad()
        
        # feed forward inputs
        logps = model(inputs)
        
        # calculate loss/error
        loss = criterion(logps, labels)
        
        # backpropogation
        loss.backward()
        
        # update classifier parameters
        optimizer.step()
        
        # track/update running loss
        running_loss += loss.item()
        
        # run validation loop to track accuracy
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            
            # switch model to evaulation mode
            model.eval()
            
            # turn off gradient calculations
            with torch.no_grad():
                for inputs, labels in validloader:
                    
                    # move input labels to gpu
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # feed forward inputs
                    logps = model(inputs)
                    
                    # calculate loss/error
                    batch_loss = criterion(logps, labels)
                    
                    # track/update valid loss
                    valid_loss += batch_loss.item()
                    
                    # calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                    
            # print results
            print(f"Epoch {epoch+1}/{epochs} . .  "
                  f"Train Loss: {running_loss/print_every:.3f} . .   "
                  f"Valid Loss: {valid_loss/len(validloader):.3f} . .  "
                  f"Valid Accuracy: {accuracy/len(validloader):.3f} . .   ")
            
            # reset running loss
            running_loss = 0
            
            # set model to training mode
            model.train()
                
# Save trained classifier information into checkpoint
checkpoint = {'input_size': input_value,
              'output_size': 102,
              'hidden_layers': [in_arg.hidden_units_1, in_arg.hidden_units_2],
              'dropout': 0.2,
              'state_dict': model.state_dict(),
              'epochs': in_arg.epochs,
              'batch_size': 64,
              'learning_rate': in_arg.learning_rate,
              'optimizer': optimizer.state_dict(),
              'model': in_arg.arch.lower(),
              'class_to_idx': train_data.class_to_idx}

torch.save(checkpoint, in_arg.save_dir)
              
              
                                       