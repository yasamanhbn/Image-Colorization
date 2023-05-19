import torch
import dataloaders
import deeplearning
import nets
import utils

def train():
    torch.manual_seed(42)
    learning_rate, batch_size, num_epochs, dataset_path, train_split, gamma = utils.read_configs()
    train_loader, test_loader, testdataset = dataloaders.get_dataloaders(dataset_path, batch_size, train_split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_model = nets.CNNBlock()
    model, optimizer, val_loss, train_loss = deeplearning.train(
        train_loader = train_loader,
        val_loader = test_loader,
        model = my_model,
        epochs = num_epochs,
        learning_rate = learning_rate,
        gamma = gamma,
        device = device
    )
    print(val_loss)
    print(train_loss)
    # utils.plot_acc_loss(train_acc, train_loss, val_acc, val_loss)
    return model, optimizer, testdataset, device

if __name__ == "__main__":
    train()