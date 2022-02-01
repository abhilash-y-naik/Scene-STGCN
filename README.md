# Scene-STGCN

<p align="center">
<img src="scene_stgcn.png" alt="Scene-STGCN" align="middle" width="600"/>
</p>


This repository contains Python code and pretrained models for pedestrian intention estimation presented in our paper [**Abhilash Y. Naik, Ariyan Bighashdel, Pavol Jancura, and Gijs Dubbelman, "Scene Spatio-Temporal Graph Convolutional Network for Pedestrian
Intention Estimation".**]


### Table of contents
* [Dependencies](#dependencies)
* [PIE dataset](#PIE_dataset)
* [Train](#train)
* [Test](#test)

<a name="dependencies"></a>
## Dependencies
The interface is written and tested on Ubuntu 16.04 with Python 3.5, CUDA 9 and cuDNN 7. The interface also requires
the following external libraries:<br/>
* tensorflow (tested with 1.9 and 1.14)
* keras (tested with 2.1 and 2.2)
* scikit-learn
* numpy
* pillow

To install via virtual environment (recommended) follow these steps:

- Install virtual environment `sudo apt-get install virtualenv`.

- Create a virtual environment with Python3:

```
> virtualenv --system-site-packages -p python3 ./venv
> source venv/bin/activate
```
- Install dependencies:
`pip3 install -r requirements.txt`


<a name="datasets"></a>
## PIE Dataset
The code is trained and tested with [Pedestrian Intention Estimation (PIE) dataset](http://data.nvision2.eecs.yorku.ca/PIE_dataset/).

Download annotations and video clips from the [PIE webpage](http://data.nvision2.eecs.yorku.ca/PIE_dataset/) and place them in the `PIE_dataset` directory. The folder structure should look like this:

```
PIE_dataset
    annotations
        set01
        set02
        ...
    PIE_clips
        set01
        set02
        ...

```

Videos will be automatically split into individual frames for training. This will require **1.1T** of free space on the hard drive.

Create environment variables for PIE data root and add them to your `.bashrc`:

```
export PIE_PATH=/path/to/PIE/data/root
export PIE_RAW_PATH=/path/to/PIE/data/PIE_clips/
```

Download PIE data interface `pie_data.py` from [PIE github](https://github.com/aras62/PIE).


<a name="train"></a>
## Train

To train all models from scratch and evaluate them on the test data use this command:
```
python train_test.py 1
```
This will train intention, speed and trajectory models separately and evaluate them on the test data.

_Note: training intention model uses image data and requires 32GB RAM.

Due to the random initialization of the networks and minor changes to the annotations there might be slight variation in the results.

<a name="test"></a>
## Test

To reproduce the results of our best model which combines pedestrian intention and vehicle speed for pedestrian trajectory prediction run this command:

```
python train_test.py 2
```

