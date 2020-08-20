  # Image Classifier 

  Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smartphone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.
  
  In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice, you'd train this classifier, then export it for use in your application. We'll be using this dataset of 102 flower categories.
  
  When you've completed this project, you'll have an application that can be trained on any set of labelled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset.
  
  *This is the final Project of the Udacity AI with Python Nanodegree*


## Prerequisites
The Code is written in Python 3.6.5 . If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. 

To install pip run in the command Line
```
python -m ensurepip -- default-pip
``` 
to upgrade it 
```
python -m pip install -- upgrade pip setuptools wheel
```
to upgrade Python
```
pip install python -- upgrade
```
Additional Packages that are required are: [Numpy](http://www.numpy.org/), [Pandas](https://pandas.pydata.org/), [MatplotLib](https://matplotlib.org/), [Pytorch](https://pytorch.org/), PIL and json.\
You can donwload them using [pip](https://pypi.org/project/pip/)
```
pip install numpy pandas matplotlib pil
```
or [conda](https://anaconda.org/anaconda/python)
```
conda install numpy pandas matplotlib pil
```
In order to intall Pytorch head over to the Pytorch site select your specs and follow the instructions given.


## Command Line Application
* Train a new network on a data set with ```train.py```
  * Basic Usage : ```python train.py data_directory```
  * Prints out current epoch, training loss, validation loss, and validation accuracy as the netowrk trains
  * Options:
    * Set direcotry to save checkpoints: ```python train.py data_dor --save_dir save_directory```
    * Choose arcitecture (alexnet, densenet121 or vgg16 available): ```pytnon train.py data_dir --arch "vgg16"```
    * Set hyperparameters: ```python train.py data_dir --learning_rate 0.001 --hidden_layer1 120 --epochs 20 ```
    * Use GPU for training: ```python train.py data_dir --gpu gpu```
    
* Predict flower name from an image with ```predict.py``` along with the probability of that name. That is you'll pass in a single image ```/path/to/image``` and return the flower name and class probability
  * Basic usage: ```python predict.py /path/to/image checkpoint```
  * Options:
    * Return top **K** most likely classes:``` python predict.py input checkpoint ---top_k 3```
    * Use a mapping of categories to real names: ```python predict.py input checkpoint --category_names cat_To_name.json```
    * Use GPU for inference: ```python predict.py input checkpoint --gpu```

## Json file
In order for the network to print out the name of the flower a .json file is required. If you aren't familiar with json you can find information [here](https://www.json.org/). By using a .json file the data can be sorted into folders with numbers and those numbers will correspond to specific names specified in the .json file.

## Data and the json file
The data used specifically for this assignemnt are a flower database are not provided in the repository as it's larger than what github allows. Nevertheless, feel free to create your own databases and train the model on them to use with your own projects. The structure of your data should be the following:\
The data need to comprised of 3 folders, test, train and validate. Generally the proportions should be 70% training 10% validate and 20% test.\
Inside the train, test and validate folders there should be folders bearing a specific number which corresponds to a specific category, clarified in the json file. For example if we have the image a.jpj and it is a rose it could be in a path like this /test/5/a.jpg and json file would be like this {...5:"rose",...}. Make sure to include a lot of photos of your catagories (more than 10) with different angles and different lighting conditions in order for the network to generalize better.
    

## GPU
As the network makes use of a sophisticated deep convolutional neural network  the training process is impossible to be done by a common laptop. In order to train your models to your local machine you have to use Cuda

1. **Cuda** -- If you have an NVIDIA GPU then you can install CUDA from [here](https://developer.nvidia.com/cuda-downloads). With Cuda you will be able to train your model however the process will still be time consuming

However, once a model is trained then a normal CPU can be used for the predict.py file and you will have an answer within some seconds.
