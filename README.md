# Introduction 
This repo contains the research conducted during the summer of 2021 on the implemenation of machine learning methods in extracting infromation for the organisational group structures.

Methods of the research processed are contained within this repository.


# Getting Started
In order to run this on your local machine, follow the steps below using Command Prompt (optimised for Windows): 


1.
````
cd path/to/hold-IQ-phase-3
````
2.
#### Generation of graph requires Visual C/C++ from : https://visualstudio.microsoft.com/visual-cpp-build-tools/ and pygraphivz

Pygraphviz can be installed if Visual C/C++ is installed by executing :
````
python -m pip install --global-option=build_ext  --global-option="-IC:\Program Files\Graphviz\include"  --global-option="-LC:\Program Files\Graphviz\lib" pygraphviz
````
If any problem occurs, check https://pygraphviz.github.io/documentation/stable/install.html. Contains the link to download Graphviz.
3.
````python
pip install -r requirements.txt
````
4.	Each of the files contains individual README.md explaining how to use and execute each section of code


## File Conversion
A file conversion method was implemented to convert group structures in PDF and PowerPoint format into JPG format contained in the PDF_PP2JPG.py file. This method works
only on Windows. Below is an example of how to execute the code to convert files.
````python
from PDF_PP2JPG import converter # From file in the same directory 

convert = converter('path/to/files', 'path/to/output')

convert.Convert_All()
````

## Line Detection (Hough Transform)
A method using a rule based approach to convert the lines in images into data points with Hough Transform was experimented with
but it is fragile and not robust, and was abandoned in favour of machine learning techniques.

## Graph Generation
Using Graphviz and Networkx to generate graphs that have similar design style to the organisational group structures. This enabled the creation of a large number of synthetic group structures.
Examples of graph generation for different use cases are found in the individual files.

## Convolution Neural Network
A model using Covolution Neural Network was built to test the viability to understand the features of the group structure. The experiment
that was done was to use the CNN to identify the nodes and count how many there are. It is possible to generate synthetic data and train them in
the cnn folder, but it turns out that it did not work well.

## Object Detection
A system to automatically annotated the group structure images was created to enable to generation of large amount of training data. This system is very fragile however and adding text to the nodes will cause it to fail.
An object detection model, YOLO-v5 was implemented during this project and trained with the generated group structure images. The best weights of the training is saved in the yolov-runs/runs/train/pre-trained folder. If more training is required, follow the README file in the yolo folder.

## Attention Mechanism
A model with attention mechanism was built to examine the viability of using for the extraction of the parent-child relationships of the entities, this came about from the failure of the CNN model.  Due to the lack of time, only an attention mechanism was implemented for classification
of the existence of nodes in images or the lack of nodes. This can ba paired with other neural network to tackle the problem of extracting the parent-child relationship. This enables the 
network to focus its attention on certain parts of the image for classification or regression problem.

## Optical Character Recognition
Tesseract OCR (pytesseract) was experimented for the use of reading the text from entities provided with the bounding boxes. The 
pytesseract library can be downloaded at https://pypi.org/project/pytesseract/, which also includes instructions on how to utilise it.

