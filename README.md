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
### **How to train the HCRNet model**

You can train the model of 5-fold cross-validation with a very simple way by the command blow: *Python HCRNet-Train.py.*

 The script of if **name** == "**main** calls training process which trains several models of each model type for an RNA and finds the best set of hyperparameters. The main function then trains the models several times (num_final_runs) and saves the best model.

### **How to predict the probability of unknown circRNA**

The *HCRNet-Predict.py* is proposed to calculate the probability for the circRNAs of unknown types. Please also change following paths to suit your system:

1. set the sequence location. e.g.,

   ```python
   seqPath = '/home/Sequence/'
   ```

2. set the type of the RNA Embeddings. e.g., 

   ```python
   modelType = '/modelpath/circRNA2Vec_model'
   ```

3. set the type of circRNA model. e.g., 

   ```python
   modelPredictType = '/finalmodel_path/model.h5'
   ```

The prediction results will be displayed automatically. If you need to save the results, please specify the path yourself. 

## **Website of HCRNet**

We also provide a website http://39.104.118.143:5001/. HCRNet provides identification of the specific binding events for circRNA and linearRNA segments or full-length circRNA sequences. Meanwhile, HCRNet also allows users to facilitate the identification of potential circRNA-RBP binding targets with a generic strategy model.

## Contact
Thank you and enjoy the tool! If you have any suggestions or questions, please email me at [*yangyn533@nenu.edu.cn*](mailto:yangyn533@nenu.edu.cn)*.*
