import train
import test

if __name__ == "__main__":
    model, optimizer, test_loader, device = train.train()
    test.test(model, optimizer, test_loader, device)