import argparse

# function that sets up input options for train file
def get_input_args():
    
    # Assign ArgumentParser method to variable
    parser = argparse.ArgumentParser()
    
    # Add inputs
    
    # Data directory input
    parser.add_argument('data_directory', type=str, help = 'directory where training, validation, test sets of images are saved')
    
    # Save directory input
    parser.add_argument('--save_dir', type = str, default = 'checkpoint_script.pth', help = 'name of file where checkpoint is saved')
    
    # Architecture input
    parser.add_argument('--arch', type = str, default = 'vgg16', help = 'select a model architecture to use between AlexNet, ResNet, and vgg')
    
    # Learning rate input
    parser.add_argument('--learning_rate', type = float, default = 0.005, help = 'select learning rate for model training')
    
    # First hidden layer input
    parser.add_argument('--hidden_units_1', type = int, default = 408, help = 'select size of first hidden layer for model classifier')
    
    # Second hidden layer input
    parser.add_argument('--hidden_units_2', type = int, default = 204, help = 'select size of second hidden layer for model classifier')
    
    # Epochs input
    parser.add_argument('--epochs', type = int, default = 5, help = 'select number of epochs for model training')
    
    # GPU input
    parser.add_argument('--gpu', type = str, default = 'gpu', help = 'run model on gpu')
    
    return parser.parse_args()