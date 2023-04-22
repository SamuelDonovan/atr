# From the python standard library
import logging
import os

# From pytorch
import torch

# From matplolib
from matplotlib import pyplot as plt

# From numpy
import numpy as np


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            logging.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device, no_output=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    accuracy = 100 * correct
    if no_output:
        return accuracy
    logging.info("Test Error:")
    logging.info(f"Accuracy: {accuracy:>0.3f}%")
    logging.info(f"Avg loss: {test_loss:>8f}")
    logging.info("")
    return accuracy


def plot_accuracy(training_data, validation_data):
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
    plt.ylim(top=100)
    plt.legend()
    PLOT_NAME = "accuracy_plot.png"
    plt.savefig(PLOT_NAME)
    logging.debug(f"Created accuracy vs epochs plot {PLOT_NAME}.")

if __name__ == "__main__":
    raise Exception("This module is not an entry point!")
