# imports
import argparse
import torch
from torch import nn
from torchvision import models
from PIL import Image
import numpy as np
import json

# function that sets input options for predict file
def get_input_args():
    
    # assign ArgumentParser method to variable
    parser = argparse.ArgumentParser()
    
    # add inputs
    
    # path to image input
    parser.add_argument('path_to_image', type = str, help = 'directory path for image to be classified')
    
    # checkpoint with trained classifier input
    parser.add_argument('checkpoint', type = str, help = 'file name where checkpoint with trained classifier is saved')
    
    # number of top clases input
    parser.add_argument('--top_k', type = int, default = 3, help = 'choose number of most probable classes to return')
    
    # category to names dictionary input
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'select file to map classes to real names')
    
    # GPU input
    parser.add_argument('--gpu', type = str, default = 'gpu', help = 'run model on gpu')
    
    return parser.parse_args()

# function that loads a trained classifier
def load_checkpoint(filepath, device):
    
    # load the saved model data
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    
    # rebuild the classifier architecture
    classifier = nn.Sequential(nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers'][0]),
                               nn.ReLU(),
                               nn.Dropout(checkpoint['dropout']),
                               nn.Linear(checkpoint['hidden_layers'][0], checkpoint['hidden_layers'][1]),
                               nn.ReLU(),
                               nn.Dropout(checkpoint['dropout']),
                               nn.Linear(checkpoint['hidden_layers'][1], checkpoint['output_size']),
                               nn.LogSoftmax(dim=1))
    
    # add class_to_idx dict to classifier
    classifier.class_to_idx = checkpoint['class_to_idx']
    
    # import the correct model
    if checkpoint['model'] == 'alexnet':
        model = models.alexnet(pretrained=True)  
    elif checkpoint['model'] == 'resnet':
        model = models.resnet18(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
    
    # attach trained classifier to model
    if checkpoint['model'] == 'resnet':
        model.fc = classifier
    else:
        model.classifier = classifier
    
    # attach state_dict to model and classifier
    model.load_state_dict(checkpoint['state_dict'])
    
    # attach model arch to model
    model.arch = checkpoint['model']
    
    return model

# function that preprocesses image to load into classifier
def process_image(image):
    
    # load PIL image
    pil_image = Image.open(image)
    
    # resize the image
    width = pil_image.width
    height = pil_image.height
    
    if width == height:
        pil_image_resized = pil_image.resize((256, 256))
    
    elif width > height:
        if height < 256:
            width_resized = int(width * (1 + ((256 - height) / height)))
            pil_image_resized = pil_image.resize((width_resized, 256))
        
        elif height > 256:
            width_resized = int(width * ((height - 256) / height))
            pil_image_resized = pil_image.resize((width_resized, 256))
        
        else:
            pil_image_resized = pil_image
            
    else:
        if width < 256:
            height_resized = int(height * (1 + ((256 - width) / width)))
            pil_image_resized = pil_image.resize((256, height_resized))
        
        elif width > 256:
            height_resized = int(height * ((width - 256) / width))
            pil_image_resized = pil_image.resize((256, height_resized))
        
        else:
            pil_image_resized = pil_image

    # crop the image
    new_width = pil_image_resized.width
    new_height = pil_image_resized.height
    
    left = int((new_width - 224) / 2)
    top = int((new_height - 224) / 2)
    right = int((new_width + 224) / 2)
    bottom = int((new_height + 224) / 2)
    
    pil_image_crop = pil_image_resized.crop((left, top, right, bottom))
    
    # convert image to np.array and normalize
    np_image = np.array(pil_image_crop)
    np_image_float = np_image.astype(float)
    np_image_converted = np_image_float / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image_normal = (np_image_converted - mean) / std
    
    # transpose dimensions of image
    image = np_image_normal.transpose((2, 0, 1))
    
    return image

# function that predicts class of image
def predict(image_path, model, topk=5):
    in_arg = get_input_args()
    # activate GPU
    if in_arg.gpu == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    
    # preprocess the image
    image = process_image(image_path)
    
    # feed image into model
    model.to(device)
    model.eval()
    with torch.no_grad():
        # if torch.cuda.is_available():
            # image = torch.from_numpy(image).type(torch.cuda.FloatTensor).unsqueeze_(0)
        # else:
            # image = torch.from_numpy(image).type(torch.FloatTensor).unsqueeze_(0)
        image = torch.from_numpy(image).type(torch.float).unsqueeze_(0).to(device)
        logps = model(image)
        
        # find the most probable classes
        ps = torch.exp(logps)
        probs, classes = ps.topk(in_arg.top_k, dim=1)
        
        # convert indicies to correct class names
        if model.arch == 'resnet':
            class_to_idx = model.fc.class_to_idx
        else:
            class_to_idx = model.classifier.class_to_idx
            
        idx_to_class = dict(zip(class_to_idx.values(), class_to_idx.keys()))
        classes_converted = []
        for i in range(0, in_arg.top_k):
            classes_converted += idx_to_class[classes[0,i].item()]
        
        # create cat_to_name dictionary
        with open (in_arg.category_names, 'r') as f:
            cat_to_name = json.load(f)
        
        # create list of flower names using classes_converted list
        classes_names = []
        for i in range(0, in_arg.top_k):
            classes_names.append(cat_to_name[classes_converted[i]])
        
        for i in range(0, in_arg.top_k):
            print('The flower has a {:.3f} probability of being a {:30}.'.format(probs[0,i], classes_names[i]))
      
        
       
            
        
    
    
    
