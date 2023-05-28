import train
import test

if __name__ == "__main__":
    model, optimizer, test_dataset, device, test_loader = train.train()
    test.test(model, optimizer, test_dataset, device, test_loader)