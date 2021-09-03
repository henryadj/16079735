# Introduction 
This directory contains the code to generate synthetic data for training the CNN model. 

# Execution
1. Change directory in Command Line to this directory 
2. Generate Data : 
   ````
   python generate_graphs.py (Enter file save location and the number of training and validation data)
   ````
3. Run Training : 
   ````
   python train_cnn.py (Enter file save location and number of epochs)

In train_cnn.py, it is possible to change parameters such as batch size and image size. 
