# imports
import torch
from torch import nn
from torchvision import models
from PIL import Image
import numpy as np
import json

from predict_functions import get_input_args, load_checkpoint, process_image, predict

# load command line inputs
in_arg = get_input_args()

# Activate GPU
if in_arg.gpu == 'gpu':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

# load command line arguments
in_arg = get_input_args()

# load trained model and classifier
model = load_checkpoint(in_arg.checkpoint, device)

# predict image class
predict(in_arg.path_to_image, model, in_arg.top_k)


