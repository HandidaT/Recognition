# Recognition

*In collaboration with Aman Jaswal*

## Introduction

It is often difficult to predict which machine learning algorithm works best for a particular classification task. Almost always the only way to identify the best algorithm is to run all of them on the concerned dataset. This application uses several machine learning algorithms which can be
compared to check performance on given dataset and identify best algorithm. Primarily, decision trees and artificial neural networks are compared.
The dataset being used for this project is MNIST hand written digits dataset which is a classification type of dataset along with additional appended data for characters. The recognition of characters, digits and symbols is crucial in many real world applications. Decision trees and Artificial neural network are choosen as the algorithms to be compared for this classification task. Decision tree is a very useful decision making algorithm that helps in decision making tasks. It does it by splitting the decision space into clusters. This segregation usually done by CART or ID3 algorithm. Decision trees are simple and works well on small datasets. While Artificial neural networks are universal function approximators, which means they can approximate to any function, thus can be applied for large datasets. Artificial neural networks are flexible in decision making but decision trees can only segregate the decision space using true and false values.Thus, these aspects of both decision trees and artificial neural networks along with the crucial classification task of character recognition, makes it appealing to work on and compare.



## System requirements

Python is the language of choice for this project. Particularly python version 3. Python comes preinstalled in operating systems like Ubuntu. For other operating systems if python is not already installed, it can be downloaded from the python software foundation website www.python.org as a an executable and run the installer locally. The numpy module is a prerequisite for this project. Numpy is numerical python library in python. It is used for scientific computing. It is optimized for mathematical computation. Numpy use ndarray object at its core. Most of the matrix operations in the work done using numpy. Using conda it can be installed with command
```
conda install numpy
```
Using pip it can be installed using command
```
pip install numpy
```
Logging module of python is utilized for tracking events in the system and debugging code. It is used in this work for tracking the processing of
neural network and decision tree. It can be installed with command
```
pip install logging
```

## Converting rgb pixel to yuv pixel
The input data i.e, the pixel data also has to be in same pixel format for all inputs data points and on the same scale(0-255). The images of all standard formats gif, tiff and jpg are stored in the range 0-255. This range is choosen particularly because pixel data can be represented in 8 bits. The format used here is YUV format. Data from RGB images are converted to YUV pixel format and then used as input. YUV is a color coding scheme used for efficiently storing images. RGB is also a color encoding scheme. The rgb stands for red, green and blue. RGB is a very nice system that works really well. The rgb is the system that is used by actual physical equipment. So, in any computer or physical device an image always need to be converted to rgb. Despite all that, rgb is not always used in the system, especially when processing the image. Rather YUV aka YcbCr is used for processing. Just like rgb, yuv also has three values per pixel. Y inyuv represents luminance of the image, while U and V represent coordinates on chrominance plane.

RGB can be converted to YUV using following coefficients:
```
Y = 0.29900 R+ 0.58700G + 0.11400B + 0
U = -0.16874R – 0.33126G + 0.500B + 128
V = 0.50000R - 0.41869G – 0.08131B + 128
```

## Dataflow Diagram


