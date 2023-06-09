import torch
import dataloaders
import deeplearning
import nets
import utils
from torchsummary import summary

def train():
    torch.manual_seed(42)
    learning_rate, batch_size, num_epochs, dataset_path, train_split, gamma = utils.read_configs()
    train_loader, valid_loader, test_loader, testdataset = dataloaders.get_dataloaders(dataset_path, batch_size, train_split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_model = nets.CNNBlock()
    # print(summary(my_model))
    model, optimizer, val_loss, train_loss = deeplearning.train(
        train_loader = train_loader,
        val_loader = valid_loader,
        model = my_model,
        epochs = num_epochs,
        learning_rate = learning_rate,
        gamma = gamma,
        device = device
    )
    print(val_loss)
    print(train_loss)
    utils.plot_loss(train_loss, val_loss)
    return model, optimizer, testdataset, device, test_loader

if __name__ == "__main__":
    train()