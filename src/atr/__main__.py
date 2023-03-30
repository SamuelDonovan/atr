#!/usr/bin/env python3

# From the python standard library
import argparse
import logging
import os
import sys

# From pytorch
import torch
import torchvision

# From numpy
import numpy as np

# Local imports
from . import cnn_model
from . import dnn_utils
from . import libdata
from . import liblogging


def parse_inputs():
    def error_printout():
        logging.error(f"See help menu for interface below:")
        parser.print_help()
        sys.exit()

    parser = argparse.ArgumentParser(
        description="""
        The objective for this project is to gain hands-on 
        experience implementing, training, and experimenting with basic
        deep neural network (DNN) architectures for a computer vision problem.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--save", action="store_true", help="Save the model")
    parser.add_argument("--load", action="store_true", help="Load the model")
    parser.add_argument(
        "--model_info", action="store_true", help="Print information about the model"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Create training and/or class confusion plot",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="atr_model",
        help="Name the model to save/load",
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate for training"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=40, help="Number of training epochs"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbosity",
        action="count",
        default=0,
        help="""Verbosity, 
        between 0-1 occurrences with more leading 
        to moreverbose logging. INFO=0, DEBUG=1
        """,
    )

    args = parser.parse_args()

    if args.save and (not args.train):
        logging.error("Cannot save model results without training.")
        error_printout()

    if args.test and ((not args.train) and (not args.load)):
        logging.error(
            "Cannot test model results without training or loading the model."
        )
        error_printout()

    if (
        (not args.test)
        and (not args.save)
        and (not args.train)
        and (not args.load)
        and (not args.model_info)
    ):
        logging.error("Please select an option.")
        error_printout()

    return args


if __name__ == "__main__":
    args = parse_inputs()

    # To set the default logger verbosity to info an offset of 3 is used.
    liblogging.setup_logger(args.verbosity + 3)

    # Testing done using pytorch 2.0.0+cu117.
    logging.debug(f"Pytorch version is {torch.__version__}")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.debug(f"Using {DEVICE} for torch.")

    # Download training/test data.
    libdata.download_zip_data(
        [
            "https://zenodo.org/record/2654887/files/COMPASS-XP.zip",
        ]
    )

    # Chosen hyperparameters.
    BATCH_SIZE = args.batch_size
    logging.debug(f"Using batch size of {BATCH_SIZE}.")
    LEARNING_RATE = args.learning_rate
    logging.debug(f"Using learning rate of {LEARNING_RATE}.")
    TRAINING_EPOCHS = args.epochs
    logging.debug(f"Using number of training epochs of {TRAINING_EPOCHS}.")

    model = cnn_model.NeuralNetwork().to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss()
    if args.model_info:
        logging.info(f"Model Information:\n{model}\n")

    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = torchvision.datasets.ImageFolder(
        "COMPASS-XP/COMPASS-XP",
        transform=train_transform,
    )
    TRAIN_SIZE = int(0.8 * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [
            TRAIN_SIZE,
            len(dataset) - TRAIN_SIZE,
        ],
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    if args.train:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        training_accuracy = []
        # validation_accuracy = []
        for epoch in range(TRAINING_EPOCHS):
            logging.info(f"-----------------------------")
            logging.info(
                f"---------- Epoch {epoch + 1} {'----------' if epoch <= 10 else '---------'}"
            )
            logging.info(f"-----------------------------")
            dnn_utils.train(train_loader, model, loss_fn, optimizer, DEVICE)
            accuracy = dnn_utils.test(
                train_loader, model, loss_fn, DEVICE, no_output=True
            )
            training_accuracy.append(accuracy)
            # accuracy = dnn_utils.test(validation_loader, model, loss_fn, DEVICE)
            # validation_accuracy.append(accuracy)

    if args.plot and args.train:
        dnn_utils.plot_accuracy(training_accuracy, validation_accuracy)

    if args.save:
        torch.save(model.state_dict(), f"{args.model_name}.pth")

    if args.load:
        model.load_state_dict(torch.load(f"{args.model_name}.pth"))

    if args.test:
        logging.info("")
        logging.info("Using final test data:")
        dnn_utils.test(test_loader, model, loss_fn, DEVICE)
    if args.plot and args.test:
        dnn_utils.plot_confusion_matrix(model, test_loader, DEVICE)
