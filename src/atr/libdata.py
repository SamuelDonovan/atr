# From the python standard library
import logging
import os
import shutil
import urllib
import zipfile

# From pandas
import pandas


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


# I know there are other ways to do this such as make a custom
# dataset class but this felt like the path of least resistance
# and would allow for more common code.
def sort_test_data():
    SORTED_DIR = "GTSRB_Final_Test_Images/GTSRB/Final_Test/Sorted"
    if os.path.exists(SORTED_DIR):
        return
    logging.info("Sorting testing data.")
    df = pandas.read_csv("GTSRB_Final_Test_GT/GT-final_test.csv", sep=";")
    os.mkdir(SORTED_DIR)
    for idx in range(len(df.index)):
        classFolder = os.path.join(SORTED_DIR, str(df.ClassId[idx]).zfill(5))
        if not os.path.exists(classFolder):
            os.mkdir(classFolder)
        shutil.copy(
            os.path.join(
                "GTSRB_Final_Test_Images/GTSRB/Final_Test/Images", str(df.Filename[idx])
            ),
            classFolder,
        )


if __name__ == "__main__":
    raise Exception("This module is not an entry point!")
