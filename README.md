# Handwriting-Recognition

## Overview
These are the codes for 2 experiments.

We have the same 3 network structures for each experiment, namely 
1. fully connected neural network(fc)
2. AlexNet with max pooling 
3. and AlexNet with average pooling

1st experiment:
* Train the 3 network structures with normal MNIST train dataset.
* Test the 3 well trained network models with normal MNIST test dataset.
* Test the 3 well trained network models with handwritten numbers by ourselves by running the "testing personal image process."
* code
    * Fully_connected/normal.ipynb
    * AlexNet/max_normal.py
    * AlexNet/avg_normal.py

* experiment .pth file
    * Fully_connected/normal.pt
    * AlexNet/max_normal_params.pth
    * AlexNet/avg_normal_params.pth

2nd experiment:
* Train the 3 network structures with *added noise* MNIST train dataset.
* Test the 3 well trained network models with *added noise* MNIST test dataset.
* Test the 3 well trained network models with *added noise* handwritten numbers by ourselves by running the "testing personal image process."
* code
    * Fully_connected/noise.ipynb
    * AlexNet/max_noise.py
    * AlexNet/avg_noise.py

* experiment .pth file
    * Fully_connected/noise.pt
    * AlexNet/max_noise_params.pth
    * AlexNet/avg_noise_params.pth

## Prerequisite & Usage
First run the following instruction in your command line to set up the pytorch environment.
```
pip3 install torch torchvision torchaudio
```
#### Code in ./Fully connected
1. Simply run the jupyter note book and the MNIST dataset will be automatically downloaded.
2. For the training and testing process, run the corresponding blocks.

#### Code in ./AlexNet
1. When running the first time, please set the "download" part in the "trainset" and "testset"(in the code) into "True", and the MNIST dataset will be automatically downloaded when running up the code. You can set it back to False after successfully downloaded the dataset.

2. For training process:
*  please turn off the code that load the .pth file in line 125-126 in *normal.py and in line 142-143 in *noise.py
* please turn on the train() function right below main
* run python3 XXX.py in your command line

3. For testing personal image process:
*  please turn on the code that load the .pth file in line 125-126 in *normal.py and in line 142-143 in *noise.py
* please turn off the train() function right below main
* run python3 XXX.py in your command line

*Hyperparameters have been already set up in the code.

> Kindly note 
> 1. The .pth files in the repository are the experiment record data. If you want to train your own models, we suggesst you to load to the new .pth file in order the retain the original one.
> 2. If the code can not be run up successfully, please check (1) .pth files and image files are under the right path (2) Dataset is succefully downloaded and be in the right path (3) Use the pth file with models' *trained* parameters to detect your image

## Experiment Result
* AlexNet + max pooling has the best performance overall
<img width="557" alt="image" src="https://user-images.githubusercontent.com/73454628/173620281-8b3d73a3-f0e5-4b40-879d-762e0e5d16ea.png">

* AlexNet structure is less affected by the Gaussian noise, especially AlexNet + average pooling
<img width="545" alt="image" src="https://user-images.githubusercontent.com/73454628/173621185-66cd4f4f-f7f2-4bb9-a5c2-02f1dec506de.png">

* Special 3
<img width="266" alt="image" src="https://user-images.githubusercontent.com/73454628/173620497-48317b0b-5a10-4764-b1f9-1395c98959c5.png">

## Contribution of each member
席秉萱 - 50%
1. fully connected neural network implementation
2. 22-page presentation ppt
3. Record 15 mins viedo presentation 

陳姵帆 - 50%
1. AlexNet implementation
2. 2-page written report
3. github repository set up
