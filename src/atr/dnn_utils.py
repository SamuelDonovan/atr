# From the python standard library
import logging
import os

# From pytorch
import torch

# From matplolib
from matplotlib import pyplot as plt

# From numpy
import numpy as np

from tqdm import tqdm

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    correct = 0
    for batch, (x, y) in enumerate(tqdm(dataloader)):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # For correct predictions
        max_index = pred.max(dim = 1)[1]
        correct += (max_index == y).sum()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), (batch + 1) * len(x)
        #     logging.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    training_accuracy = 100. * correct / size
    return training_accuracy


def test(dataloader, model, loss_fn, device, no_output=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    accuracy_test = 100 * correct
    if no_output:
        return accuracy_test
    logging.info("Test Error:")
    logging.info(f"Accuracy: {accuracy_test:>0.3f}%")
    logging.info(f"Avg loss: {test_loss:>8f}")
    logging.info("")
    return accuracy_test


# def plot_accuracy(training_data, validation_data, plot_name="accuracy_plot"):
def plot_accuracy(validation_data, plot_name="accuracy_plot"):
    plt.plot(
        list(range(1, len(training_data) + 1)), training_data, label="Training Data"
    )
    plt.plot(
        list(range(1, len(validation_data) + 1)),
        validation_data,
        label="Validation Data",
    )
    plt.grid()
    plt.title("Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (percentage)")
    plt.legend()
    PLOT_NAME = f"{plot_name}.png"
    FOLDER = "plots"
    if not os.path.exists(FOLDER):
        os.mkdir(FOLDER)
    plt.savefig(os.path.join(FOLDER, PLOT_NAME))
    logging.debug(f"Created accuracy vs epochs plot {PLOT_NAME}.")


if __name__ == "__main__":
    raise Exception("This module is not an entry point!")
