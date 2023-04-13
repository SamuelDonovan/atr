# From the python standard library
import logging
import os
import shutil
import urllib
import zipfile

# From pandas
import pandas

# From pytroch
import torch

# From Python Imaging Library
from PIL import Image

# From numpy
import numpy as np


def download_zip_data(urls):
    for url in urls:
        file = url.rsplit("/", 1)[1]
        if "zip" != file.rsplit(".", 1)[1]:
            raise Exception(f"{url} is not a zip!")
        folder = file.rsplit(".", 1)[0]
        logging.debug(f"Looking for image data {folder}.")
        if os.path.exists(folder):
            logging.debug(f"Image data {folder} found.")
            continue
        logging.info(f"Downloading {file}.")
        urllib.request.urlretrieve(url, file)
        logging.info(f"Extracting {file}.")
        with zipfile.ZipFile(file, "r") as zip_ref:
            zip_ref.extractall(folder)
        logging.info(f"Deleting {file}.")
        os.remove(file)


class CXPDataset(torch.utils.data.Dataset):
    def __init__(self, metadata, root_dirs, transform=None):
        self.metadata = metadata
        self.root_dirs = root_dirs
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        label = 1 if self.metadata.dangerous[idx] else 0
        img_name = str(self.metadata.basename[idx]).zfill(4)
        image = None
        for root_dir in self.root_dirs:
            img_path = os.path.join(root_dir, img_name)
            if os.path.exists(img_path + ".png"):
                image = Image.open(img_path + ".png")
                break
            elif os.path.exists(img_path + ".jpg"):
                image = Image.open(img_path + ".jpg")
                break
        if image is None:
            raise ValueError(
                f"Image {img_name} not found in any of the root directories"
            )
        image = self.transform(image)
        return image, label


if __name__ == "__main__":
    raise Exception("This module is not an entry point!")
