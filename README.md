## Hacktiv8 - Intermediate Python for Data Science Course
Batch 2 > 2019.11 - 2020.02  

Final project: Image classification of cats vs dogs sub-dataset.

Final project: Classification of images of cats vs dogs using neural network.

Task: Build a neural-network based image classifier that can distinguish a cat from a dog and vice-versa.

Dataset: "Dogs vs. Cats" dataset from Kaggle, subsampled by Google to 2000 training images and 1000 test/validation images, with an equal distribution of cats and dogs annotation. [https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip). 

Solution:  
* A neural-network with multiple convolutional layers of increasing kernel size was constructed with 10% dropout post-convolutional 2D max pooling.  
* Training images (1k images each for cats and dogs) were augmented to include rotational shift, zooming, x- and y-axes shifts, and horizontal flipping. 
* Test/validation images were not augmented.  
* Training was conducted on Google Colab platform with GPU hardware accelerator.

Model Architecture:
* Sequential
*  


Training Summary and Statistics of Model: 
* Epochs = 50
* Accuracy, ValidationAccuracy = 
* AUC, ValidationAUC = 


Bibliography: 
1. Google Brain Team. 2020. *TensorFlow* (v2.1.0). [Software]. [Accessed 07 Feb 2020]. 
2. Hansen, C. 2019. *Optimizers Explained - Adam, Momentum and Stochastic Gradient Descent*. [Accessed 07 Feb 2020]. Available from: [https://mlfromscratch.com/optimizers-explained/#/](https://mlfromscratch.com/optimizers-explained/#/).
3. Liu, B., Liu. Y., and Zhou, K. *Image Classification for Dogs and Cats*. [Online]. University of Alberta. [Accessed 08 Feb 2020]. Available from: [https://sites.ualberta.ca/~bang3/files/DogCat_report.pdf](https://sites.ualberta.ca/~bang3/files/DogCat_report.pdf). 
