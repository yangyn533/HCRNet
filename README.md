# **HCRNet**
This tool is developed for circRNA-binding event identification from CLIP-seq data using deep temporal convolutional network

## **Requirements**
HCRNet is written in Python3 and requires the following dependencies to be installed: <br>
+ [PyTorch 1.8.1](http://pytorch.org/) <br>
+ [Keras 2.4.3](http://keras.org/)
+ [Tensorflow-gpu 2.4.0](http://tensorflow.org/)  
+ [Sklearn](https://github.com/scikit-learn/scikit-learn)
+ [Matplotlib 3.3.4](https://matplotlib.org/)
+ [Numpy 1.19.5](http://numpy.org/)
+ [Gensim 3.8.3](http://gensim.org/)

## **Installation**
We recommend you to build a python virtual environment with [Anaconda](https://docs.anaconda.com/anaconda/install/linux/). 
```
conda create -n hcrnet python=3.7.6
conda activate hcrnet
https://github.com/yangyn533/HCRNet.git
cd HCRNet
```
## **Data Availability**
The pre-trained models (including circRNA2Vec_model, linRNA2Vec_model and bert_model), the 37 circRNA fragment datasets, 31 linear RNA datasets, a full-length circRNA dataset containing 740 sequences and the eCLIP data with binding sites of 150 novel RBPs can be downloaded in this Repositories. Meanwhile, all supporting datasets and source codes for our analyses are also freely available at (https://docs.anaconda.com/anaconda/install/linux/). 

## **Usage**
```
python HCRNet-Train.py [--RBP_ID <circRNA or linearRNA or eCLIP data>]
                       [--kmer <default=3>] 
                       [--modelType <default='~./Pre-trained models/circRNA2Vec_model'>] 
                       [--storage <default='~./resultpath/result/'>]
```
