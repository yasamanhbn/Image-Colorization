import torch
import dataloaders
import deeplearning
import nets
import utils

def train():
    torch.manual_seed(42)
    learning_rate, batch_size, num_epochs, dataset_path, train_split, gamma = utils.read_configs()
    train_loader = dataloaders.get_train_dataloaders(dataset_path, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_model = nets.CNNBlock()
    model, optimizer, val_loss, train_loss = deeplearning.train(
        train_loader = train_loader,
        val_loader = train_loader,
        model = my_model,
        epochs = num_epochs,
        learning_rate = learning_rate,
        gamma = gamma,
        device = device
    )
    # utils.plot_acc_loss(train_acc, train_loss, val_acc, val_loss)
    return model, optimizer

if __name__ == "__main__":
    train()