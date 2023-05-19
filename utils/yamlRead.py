
import yaml
def read_configs():
    with open('config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

# Access hyperparameters
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    gamma = config['gamma']

# Access dataset parameters
    dataset_path = config['dataset']['path']
    train_split = config['dataset']['train_split']
    
    return learning_rate, batch_size, num_epochs, dataset_path, train_split, gamma


# def read_configs():
# # Access hyperparameters
#     learning_rate = 0.01
#     batch_size = 64
#     num_epochs = 10
#     gamma = 0.75

# # Access dataset parameters
#     dataset_path = '/content/landscapes'
#     train_split = 0.2
    
#     return learning_rate, batch_size, num_epochs, dataset_path, train_split, gamma




