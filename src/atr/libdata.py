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

# From scikit-image
# import skimage

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
    def __init__(self, metadata, root_dir, transform=None):
        self.metadata = metadata
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        index = self.metadata.basename[idx]
        img_name = os.path.join(self.root_dir, f"{str(index).zfill(4)}.jpg")
        print(index)
        print(img_name)
        # image = skimage.io.imread(img_name)
        # landmarks = self.metadata.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'landmarks': idx}

        # sample = [image, float(self.metadata.iloc[idx, 0])]
        # sample = image

        # if self.transform:
        #     sample = self.transform(sample)

        # return sample
        return image, index


if __name__ == "__main__":
    raise Exception("This module is not an entry point!")
