# Automatic Threat Recognition

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Towards Automatic Threat Recognition: Airtport X-ray Screening

## How To

### Install Requirements 

If you chose to use a virtual environment activate it now. Then navigate cloned repo and run the following:

```sh
python3 -m pip install -r requirements.txt
```

### Running The Code

After installing the necessary requirements the code can be code can be run from the `src` directory of the cloned repo. To print the help menu run the following:

```sh
python3 -m atr --help
```

The command line interface allows for training, testing, model saving, model loading, and variable hyperparameters. For example to train and test with the default hyperparameters, run the following:

```sh
python3 -m atr --train --test
```

To train, save the model, and test with a batch size of 64 and 10 epochs, run the following:

```sh
python3 -m atr --train --test --save --batch_size 64 --epochs 10 
```

### Other

A `training_automation.sh` script has been provided as well. This can be used to wrap the python command line interface to allow for batch testing of various models/hyperparameters.