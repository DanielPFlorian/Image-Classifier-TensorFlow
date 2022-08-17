# Image-Classifier

Image-Classifier utilizes deep learning with [TensorFlow](https://www.tensorflow.org/)
to predict flower species from an image. The neural network is trained on thousands
of images of 102 species of commonly occurring flowers in the United Kingdom. When
an image of a flower is given as input to the trained model, the highest
probabilities of the species that flower corresponds to are predicted.

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

### Packages and Tools Required

#### For Jupyter Notebook
```
Python 3.7-3.10
pip install jupyter notebook==6.4.*
pip install matplotlib==3.5.*
pip install numpy==1.21.*
pip install Pillow==9.2.*
pip install tensorflow==2.8.*
pip install tensorflow_datasets==4.5.*
pip install tensorflow_hub==0.12.*
```
#### For Command Line App
```
Python 3.7-3.10
pip install matplotlib==3.5.*
pip install numpy==1.21.*
pip install Pillow==9.2.*
pip install tensorflow==2.8.*
pip install tensorflow_hub==0.12.*
```
### Command Line App Instructions
1. Clone [Image-Classifier repo](https://github.com/DanielPFlorian/Image-Classifier.git)
2. From command_line_app folder install requirements.txt with
```
pip install -r requirements.txt
```
3. Run commands from command_line_app folder with python:
```
usage: predict.py [-h] [--top_k TOP_K] [--category_names CATEGORY_NAMES] image_path model_path

positional arguments:
  image_path                        File path of image
  model_path                        Name of trained tensorflow .h5 model

optional arguments:
  -h, --help                        show this help message and exit
  --top_k TOP_K                     (int) Return the top K most likely classes
  --category_names CATEGORY_NAMES   .json file that maps labels to flower names
```
**Example 1:**
  ```
  python predict.py --top_k 3 --category_names label_map.json ./test_images/cautleya_spicata.jpg ./trained_model.h5
  ```
**Returns** image of flower and top k (int) probabilities. Close figure to continue:

![command_line_app_figure](https://github.com/DanielPFlorian/Image-Classifier/blob/main/assets/command_line_app_figure.jpeg)

**Example 2:**
```
python predict.py  --category_names label_map.json ./test_images/tiger_lily.jpg ./trained_model.h5
```
**Returns** command line output of probabilities for all 102 flower species:
```
Category Name: tiger lily
Label: 6
Probability: 0.999970555305481

Category Name: cape flower
Label: 37
Probability: 2.0300290998420678e-05

Category Name: fire lily
Label: 21
Probability: 8.172311027010437e-06

Category Name: gaura
Label: 57
Probability: 7.215427331175306e-07

Category Name: blackberry lily
Label: 102
Probability: 2.3401975113301887e-07
```

### License

This project was submitted by Daniel P Florian as part of the Nanodegree At Udacity.

As part of the Udacity Honor code, your submissions must be your own work, hence
submitting this project as yours will cause you to break the Udacity Honor Code
and the suspension of your account.

Me, the author of the project, allow you to check the code as a reference, but if
you submit it, it's your own responsibility if you get expelled.

Copyright (c) 2022 Daniel P Florian

Besides the above notice, the content of this repository is licensed under a
[MIT License](https://opensource.org/licenses/MIT)
