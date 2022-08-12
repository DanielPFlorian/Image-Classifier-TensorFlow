# Image-Classifier

Image-Classifier utilizes deep learning with TensorFlow to predict flower species
from an image. The neural network is trained on thousands of images of 102 species
of commonly occurring flowers in the United Kingdom. When an image of a flower is
given as input to the trained model, the highest probabilities of the species that
flower corresponds to are predicted.

### TensorFlow Model Overview

The dataset the model is trained on comes from the
[oxford_flowers102 dataset](https://www.tensorflow.org/datasets/catalog/oxford_flowers102).
To efficiently train, validate and test the TensorFlow model, a pipeline is
created to format, cache, shuffle, prefetch and batch the images to be fed to the
model. To increase the accuracy of predictions, the pretrained
[MobileNet V2](https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4)
neural network is utilized as an image feature extractor after which dense layers
of relu activation units and a final softmax layer are used. Dropout and early
stopping techniques are applied to prevent overfitting.
The [Project_Image_Classifier_Project.ipynb](https://github.com/DanielPFlorian/Image-Classifier/blob/main/Project_Image_Classifier_Project.ipynb)
jupyter notebook file contains the code and steps used for model creation and
evaluation.

### Techniques Used

- TensorFlow 2.8
- Data exploration
- Image normalization/pre-processing
- Training, Validation and Testing Pipeline creation
- Pre-Trained Neural Networks
- Overfitting prevention using Dropout and Early-Stopping
- Plotting Learning Curves using Matplotlib
- Image Recognition Inference using Neural Network trained Model
- ArgParse command-line app creation

### Packages and Tools Required:
```
Python
TensorFlow
TensorFlow Datasets
TensorFlow Hub
Numpy
Matplotlib
Jupyter Notebook
Pillow

```
### Package Installation
```
Python 3.7-3.10
pip install tensorflow==2.8.0
pip install tensorflow_datasets==4.5.2
pip install tensorflow_hub==0.12.0
pip install numpy==1.22.3
pip install Pillow==9.1.1
pip install jupyter notebook==6.4.11

```
