## Hacktiv8 - Intermediate Python for Data Science Course
Batch 2 > 2019.11 - 2020.02  

### Final Project 
Classification of images of cats vs dogs using deep neural network.

### Objective
Build a neural-network based image classifier that can distinguish a cat from a dog and vice-versa.

### Dataset
"Dogs vs. Cats" dataset from Kaggle, subsampled by Google to 2000 training images and 1000 test/validation images, with an equal distribution of cats and dogs annotation. [https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip). 

### Abstract
Image classification of cats and dogs are not as trivial for machines, unlike for humans. Here we present a classifier that is based on neural network composed of multiple convolutional layers of increasing kernel size. Our model was trained on a small dataset of 2000 images of equal annotation distribution, with image augmentations during training. Multiple rounds of trainings were conducted, and the best trained model achieved training and validation accuracy of **nn% and mm%**, and training and validation AUC of **yy and zz**, respectively. We tested our model using several images obtained from an online search platform, and we attribute any misclassification to the slight overfitting of the model.

### Solution
* A sequential neural-network with multiple convolutional layers of increasing kernel size was constructed with 10% dropout post-convolutional 2D max pooling.  
* Training images were augmented to include rotational shift, zooming, x- and y-axes shifts, and horizontal flipping. Test/validation images were not augmented.  
* Training was conducted on Google Colab platform with GPU hardware accelerator.

### Model Architecture: Sequential
1. 1x Conv layer of 32 channel of 3x3 kernel and same padding.  
2. 1x Maxpool layer with 2x2 pool size and 2x2 stride.
3. 1x Dropout layer with fraction of 0.1.
4. 1x Conv layer of 64 channel of 3x3 kernel and same padding.  
5. 1x Maxpool layer with 2x2 pool size and 2x2 stride.
6. 1x Dropout layer with fraction of 0.1.
7. 1x Conv layer of 128 channel of 3x3 kernel and same padding.
8. 1x Maxpool layer with 2x2 pool size and 2x2 stride.
9. 1x Dropout layer with fraction of 0.1.
7. 1x Conv layer of 256 channel of 3x3 kernel and same padding.
8. 1x Maxpool layer with 2x2 pool size and 2x2 stride.
9. 1x Dropout layer with fraction of 0.1.
10. 1x Conv layer of 512 channel of 3x3 kernel and same padding.
11. 1x Maxpool layer with 2x2 pool size and 2x2 stride.
12. 1x Dropout layer with fraction of 0.1.
13. 1x Flattening layer.
14. 1x Dense layer with 2048 units.
15. 1x Dense sigmoid layer with 1 unit.

### Training Summary and Statistics of Model 
* Epochs = 75, actual 55
* EarlyStopping = True
  * Patience = 10
  * Metric = val_loss
* Learning Rate Reduction = True
  * Metric = val_accuracy
  * Patience = 10
  * Patience = 4  
  * Factor = 2/3
  * Min LR = 1e-5
* Accuracy, ValidationAccuracy = 92.25%, 85.50% (∆ = 6.75%)
* AUC, ValidationAUC = 98.02%, 94.08% (∆ = 3.94)


### Bibliography
1. Google Brain Team. 2020. *TensorFlow* (v2.1.0). [Software]. [Accessed 07 Feb 2020]. 
2. Hansen, C. 2019. *Optimizers Explained - Adam, Momentum and Stochastic Gradient Descent*. [Accessed 07 Feb 2020]. Available from: [https://mlfromscratch.com/optimizers-explained/#/](https://mlfromscratch.com/optimizers-explained/#/).
3. Liu, B., Liu. Y., and Zhou, K. [no date]. *Image Classification for Dogs and Cats*. [Online]. University of Alberta. [Accessed 08 Feb 2020]. Available from: [https://sites.ualberta.ca/~bang3/files/DogCat_report.pdf](https://sites.ualberta.ca/~bang3/files/DogCat_report.pdf). 
4. Lau, S. 2017. *Learning Rate Schedules and Adaptive Learning Rate Methods for Deep Learning*. [Accessed 14 Feb 2020]. Available from: [https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1](https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1).