# Kaggle - Digit Recognizer

Learn computer vision fundamentals with the famous MNIST data.


## Introduction

### Overview

The overview of the Kaggle competition can be found here: 
[LINK](https://www.kaggle.com/c/digit-recognizer/overview) 

### Dataset

A detailed dataset description can be found here: 
[LINK](https://www.kaggle.com/c/digit-recognizer/data)

### Model Architecture

The model architecture is based on convolutional networks, implemented in PyTorch and 
PyTorch-Lightning. See: `./model/classifier.py` 


## Instructions

This is a step-by-step guide on how to replicate the Kaggle submission. 

### Python Environment

- Create virtual environment:
```shell script
$ pipenv --python path/to/python3.7
````
- Activate the virtual environemt:
```shell script
$ pipenv shell
```
- Install the required Python packages:
```shell script
$ pipenv sync
```

### Submission Script

Training the model on the provided training set and running predictions on the provided test set can
be done with the use of the script `./scripts/mnist.py`.

The required command-line argumets are described by running the following command:
```shell script
$ python scripts/mnist.py --help
``` 

The default model configuration can be found in `./scripts/config.json`.

If the `train.csv` and `test.csv` are not found inside the provided folder under the `-dir_in` 
argument, then it is necessary to provide a Kaggle API token:
[LINK](https://github.com/Kaggle/kaggle-api#api-credentials)
