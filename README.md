# FER
Facial Expression Recognition

This work is to demonstrate the below problem: 
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge

A real time face detector and emotion classifier is built using Convolution Neural Network and OpenCV.
The CNN model is tuned for fine performance even on a low end device.

## Instructions
Follow the [guided tutorial](FER_CNN.ipynb) for neural network training.

To run the model, `python FER.py`. Face detection and emotion recognition should start on webcam video stream.

## Installation
Using Python virtual environment will be advisable.
* For model prediction

    `pip install -r requirements.txt`
    
    OR
    
    `pip install opencv-python`
    
    `pip install tensorflow` (Note here we are installing tensorflow-cpu)
    
    `pip install keras`
    
* For model training,
    `pandas` `numpy` `tensorflow` `keras` `matplotlib` `scikit-learn` `seaborn`

## Contributing
* Report issues on issue tracker
* Fork this repo
* Make awesome changes
* Raise a pull request

##
#### Copyright & License
Halite Bot Competition Source Code

Copyright (C) 2018  Mayur Madnani

Licensed under MIT License

See the [LICENSE](LICENSE).