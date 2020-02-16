## Hacktiv8 - Intermediate Python for Data Science Course
Batch 2 > 2019.11 - 2020.02  

### Final Project 
Classification of images of cats vs dogs using deep neural network.

### Abstract
Image classification of cats and dogs are not as trivial for machines, unlike for humans. Here we present a classifier that is based on neural network composed of multiple convolutional layers of increasing kernel size. Our model was trained on a small dataset of 2000 images of equal annotation distribution, with image augmentations during training. Multiple rounds of trainings were conducted, and the best trained model achieved training and validation accuracy of 86.80% and 83.30%, and training and validation AUC of 0.947 and 0.901, respectively. We tested our model using several images obtained from an online search platform, and we attribute any misclassification to the slight overfitting of the model and its moderate evaluation metrics.

---

### Objective
Build a neural-network based image classifier that can distinguish a cat from a dog and vice-versa.

### Dataset
"Dogs vs. Cats" dataset from Kaggle, subsampled by Google to 2000 training images and 1000 test/validation images, with an equal distribution of cats and dogs annotation. [https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip). 

### Solution
* A sequential neural-network with multiple convolutional layers of increasing kernel size was constructed with 10% dropout post-convolutional 2D max pooling.  
* Training images were augmented to include rotational shift, zooming, x- and y-axes shifts, and horizontal- and vertical-flipping. Test/validation images were not augmented.  
* Training was conducted on Google Colab platform with GPU hardware accelerator.

### Model Architecture: Sequential
1. 1x Conv layer of 32 channel of 3x3 kernel and same padding.  
2. 1x Maxpool layer with 2x2 pool size and 2x2 stride.
3. 1x Dropout layer with fraction of 0.2.
4. 1x Conv layer of 64 channel of 3x3 kernel and same padding.  
5. Repeat Maxpool, Dropout as above.
6. 1x Conv layer of 128 channel of 3x3 kernel and same padding.
7. Repeat Maxpool, Dropout as above.
8. 1x Conv layer of 256 channel of 3x3 kernel and same padding.
9. Repeat Maxpool, Dropout as above.
10. 1x Flattening layer.
11. 1x Dense layer with 4096 units.
12. 1x Dropout layer with fraction of 0.4
13. 1x Dense sigmoid layer with 1 unit.

### Training Summary and Statistics of Best Model 
* Optimizer = Adam
  * Epsilon = 1e-3
* Epochs = 128, actual 88
* EarlyStopping = True
  * Patience = 15
  * Metric = val_loss
* Learning Rate Reduction = True
  * Metric = val_accuracy
  * Patience = 12 
  * Factor = 2/3
  * Min LR = 1e-5 

Metric  | Training  | Validation    | ∆
--- | :---: | :---: | :---: | 
Accuracy | 86.80% | 83.30% | 3.50%
AUC | 0.947 | 0.901 | 0.046
ModelName = 20200216_0212

#### Other Trained Models and Their Statistics
Multiple models were trained, listed below are the ones with validation accuracy > 0.8.
ModelName | valAccuracy  | ∆Accuracy (Train-Val) |valAUC  | 
|--- | :---: | :---: | :---: | 
20200214_1320 | 82.40% | 1.849% | 0.888
20200214_1346 | 80.59% | 2.399% | 0.895
20200214_1603 | 82.99% | 0.900% | 0.912
20200214_1738 | 83.30% | 8.799% | 0.924
20200214_1925 | 81.80% | 7.199% | 0.892
20200215_0302 | 81.99% | 2.399% | 0.901
20200215_1508 | 80.80% | 2.549% | 0.889

---
### Installation
This model was developed with the following module versions. If you would like to have a go, please refer to the versions stated below.
* ```tensorflow == 2.1.0```
* ```matplotlib == 3.1.3```
* ```numpy == 1.17.5```
* ```pandas == 0.25.3```
* ```PIL == 6.2.2```

### Predict Catto vs Doggo
To use our best model, ```20200216_0212```, open ```Predict_Catto-Doggo.ipynb``` notebook above and follow the instructions there. *Note: image shall be provided as a web link.*


---

### Bibliography
1. Google Brain Team. 2020. *TensorFlow* (v2.1.0). [Software]. [Accessed 07 Feb 2020]. 
2. Hansen, C. 2019. *Optimizers Explained - Adam, Momentum and Stochastic Gradient Descent*. [Accessed 07 Feb 2020]. Available from: [https://mlfromscratch.com/optimizers-explained/#/](https://mlfromscratch.com/optimizers-explained/#/).
3. Liu, B., Liu. Y., and Zhou, K. [no date]. *Image Classification for Dogs and Cats*. [Online]. University of Alberta. [Accessed 08 Feb 2020]. Available from: [https://sites.ualberta.ca/~bang3/files/DogCat_report.pdf](https://sites.ualberta.ca/~bang3/files/DogCat_report.pdf). 
4. Lau, S. 2017. *Learning Rate Schedules and Adaptive Learning Rate Methods for Deep Learning*. [Accessed 14 Feb 2020]. Available from: [https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1](https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1).
5. Ludwig, J. [no date]. *Image Convolution: Satellite Digital Image Analysis, 581*. [Online]. Portland State University. [Accessed 15 Feb 2020]. Available from: [http://web.pdx.edu/~jduh/courses/Archive/geog481w07/Students/Ludwig_ImageConvolution.pdf](http://web.pdx.edu/~jduh/courses/Archive/geog481w07/Students/Ludwig_ImageConvolution.pdf).
6. Agbede, S. 2018. *Convolution Operations and Image Processing*. [Online]. [Accessed 15 Feb 2020]. Available from: [https://medium.com/@shinaolaagbede728/convolution-operations-and-image-processing-55e7d8bdef3f](https://medium.com/@shinaolaagbede728/convolution-operations-and-image-processing-55e7d8bdef3f).