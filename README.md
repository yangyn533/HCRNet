# **HCRNet**
This tool is developed for circRNA-binding event identification from CLIP-seq data using deep temporal convolutional network

# **Requirements**
HCRNet is written in Python3 and requires the following dependencies to be installed: <br>
+ [PyTorch 1.8.1](http://pytorch.org/) <br>
+ [Keras 2.4.3](http://keras.org/)
+ [Tensorflow-gpu 2.4.0](http://tensorflow.org/)  
+ [Sklearn](https://github.com/scikit-learn/scikit-learn)
+ [Matplotlib 3.3.4](https://matplotlib.org/)
+ [Numpy 1.19.5](http://numpy.org/)
+ [Gensim 3.8.3](http://gensim.org/)

## Installation
We recommend you to build a python virtual environment with [Anaconda](https://docs.anaconda.com/anaconda/install/linux/). 
```
conda create -n hcrnet python=3.7.6
conda activate hcrnet
https://github.com/yangyn533/HCRNet.git
cd HCRNet
python3 -m pip install -r requirements.txt
```

