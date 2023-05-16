import train
import test

if __name__ == "__main__":
    model, optimizer = train.train()
    test.test(model, optimizer)