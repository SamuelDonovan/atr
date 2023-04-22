#!/usr/bin/env python3

# From the python standard library
import argparse
import logging
import os
import sys
import textwrap

# From pandas
import pandas as pd

# From pytorch
import torch
import torchvision

# From numpy
import numpy as np

# Local imports
# from . import cnn_model
from . import dnn_utils
from . import libdata
from . import liblogging


def parse_inputs():
    def error_printout():
        logging.error(f"See help menu for interface below:")
        parser.print_help()
        sys.exit()

    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """\
        We are solving the problem of detecting objects that may be threatening to
        the safety of air travel. Detecting threatening objects prevents them from 
        causing harm to air travel. False positives in this process delay the 
        screening process and reduce the efficiency of airports.

        The value added by our project and the way we feel it advances is the field
        in a novel way is that we will be training various neural network models on 
        a number different subsets/combinations of of the image data shown in 
        the COMPASS-XP dataset (e.g. high energy x-ray image with density image vs 
        low energy x-ray image with full color image, etc). This will allow us to 
        compare and contrast the various permutations and assert what the best model 
        and input dataset is for this specific application.

        This code is made to be highly parameterizable to enable this mixing and matching
        of different models and subsets of the data. 
        """
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: python3 -m atr --train --test --save --save_name resnet18_e10_b4_dCDGHL --batch_size 4 --epochs 10 --model resnet18 --data Colour Density Grey High Low",
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
        "--save_name",
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
        "--model",
        "-m",
        type=str,
        choices=[
            "alexnet",
            "resnet18",
            "resnet50",
            "vit_h_14",
            "vgg11",
            "efficientnet_b0",
            "densenet121",
            "densenet201",
            "maxvit_t",
            "swin_t",
            "swin_v2_t",
            "efficientnet_v2_s",
            "convnext_tiny",
        ],
        default="resnet18",
        help="The model to use.",
    )
    parser.add_argument(
        "--data",
        nargs="+",
        type=str,
        default=["Colour", "Density", "Grey", "High", "Low", "Photo"],
        help="""
        The types of data to use from the dataset.
        Any combination of Colour, Density, Grey, High, Low, Photo can be used.
        """,
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

    if "resnet18" == args.model:
        model = torchvision.models.resnet18(num_classes=2).to(DEVICE)
    elif "resnet50" == args.model:
        model = torchvision.models.resnet50(num_classes=2).to(DEVICE)
    elif "alexnet" == args.model:
        model = torchvision.models.alexnet(num_classes=2).to(DEVICE)
    elif "vit_h_14" == args.model:
        model = torchvision.models.vit_h_14(num_classes=2).to(DEVICE)
    elif "vgg11" == args.model:
        model = torchvision.models.vgg11(num_classes=2).to(DEVICE)
    elif "efficientnet_b0" == args.model:
        model = torchvision.models.efficientnet_b0(num_classes=2).to(DEVICE)
    elif "densenet121" == args.model:
        model = torchvision.models.densenet121(num_classes=2).to(DEVICE)
    elif "densenet201" == args.model:
        model = torchvision.models.densenet201(num_classes=2).to(DEVICE)
    elif "maxvit_t" == args.model:
        model = torchvision.models.maxvit_t(num_classes=2).to(DEVICE)
    elif "swin_t" == args.model:
        model = torchvision.models.swin_t(num_classes=2).to(DEVICE)
    elif "swin_v2_t" == args.model:
        model = torchvision.models.swin_v2_t(num_classes=2).to(DEVICE)
    elif "efficientnet_v2_s" == args.model:
        model = torchvision.models.efficientnet_v2_s(num_classes=2).to(DEVICE)
    elif "convnext_tiny" == args.model:
        model = torchvision.models.convnext_tiny(num_classes=2).to(DEVICE)
    else:
        raise Exception("Invalid model specified!")

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

    def preprocess_image(image):
        if image.mode != "RGB":
            image = image.convert("RGB")

        width, height = image.size
        crop_size = min(width, height, 224)
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(crop_size),
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        return transform(image)

    dataset_metadata = pd.read_csv("COMPASS-XP/COMPASS-XP/meta.txt", sep="\t")

    root_dirs = ["COMPASS-XP/COMPASS-XP/" + image_type for image_type in args.data]

    for image_type in args.data:
        dataset = libdata.CXPDataset(
            metadata=dataset_metadata,
            root_dirs=root_dirs,
            transform=preprocess_image,
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

    loss_fn = torch.nn.CrossEntropyLoss()
    if args.model_info:
        logging.info(f"Model Information:\n{model}\n")

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
        torch.save(model.state_dict(), f"{args.save_name}.pth")

    if args.load:
        model.load_state_dict(torch.load(f"{args.save_name}.pth"))

    if args.test:
        logging.info("")
        logging.info("Using final test data:")
        dnn_utils.test(test_loader, model, loss_fn, DEVICE)
    if args.plot and args.test:
        dnn_utils.plot_confusion_matrix(model, test_loader, DEVICE)
