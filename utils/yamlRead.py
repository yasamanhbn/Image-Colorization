
import yaml
def read_configs():
    with open('config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

# Access hyperparameters
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    gamma = config['gamma']
    momentum = config['momentum']
    model_save_path = config['model_save_path']
    model_load_path = config['model_load_path']
    cnnBlockType = config['cnnBlockType']

# Access dataset parameters
    dataset_path = config['dataset']['path']
    num_classes = config['dataset']['num_classes']
    train_split = config['dataset']['train_split']
    
    return learning_rate, batch_size, num_epochs, dataset_path, num_classes, train_split, gamma, momentum, model_save_path, model_load_path, cnnBlockType



def get_datasetPath():
    with open('config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config['dataset']['path']



