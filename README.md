# Deeper Networks for Image Classification

MNIST CIFAR and RESNET implementation

Three models have been used to accomplish the task VGG16, GoogleNet and ResNet. All the models have been made according to description in the paper published. The model has been trained and tested according to the input size of the data. No pretrained weights are used on any of the network.

In this work I have evaluated very deep convolutional networks for largescale image classification. It was demonstrated that the representation depth is beneficial for the classification accuracy, and that state-of-the-art performance on the ImageNet challenge dataset can be achieved using a conventional ConvNet architecture (LeCun et al., 1989; Krizhevsky et al., 2012) with substantially increased depth. While VGG achieves a phenomenal accuracy on MNIST dataset, its deployment on even the most modest sized GPUs is a problem because of huge computational requirements, both in terms of memory and time. It becomes inefficient due to large width of convolutional layers. GoogleNet has 22 layers which increases the accuracy, but it can increase the overfitting problem. Likewise, Number of layers and ScaleRange increases the accuracy of the GoogleNet and the result above speak for themselves. GoogleNet has performed very well on both the datasets. The ResNet model with 18 layers has performed well on both the datasets as well. ResNet with the idea of ‘identity shortcut connection’ helps us to understand that increasing network depth does not work by simply stacking layers together. Deep networks are hard to train because of the notorious vanishing gradient problem — as the gradient is back-propagated to earlier layers, repeated multiplication may make the gradient infinitively small. As a result, as the network goes deeper, its performance gets saturated or even starts degrading rapidly. ResNet with 18 layers has performed significantly better than the VGG and comparable to GoogleNet. I have also shown that models generalise well to a wide range of tasks and datasets, matching or outperforming more complex recognition pipelines built around less deep image representations. The results have yet again confirmed the importance of depth in visual representations.

Training Time
The training time has been found out for all the models. The Initial Learning Rate for all the model is kept at 0.01 and it
is clear from below figure that VGG takes longer time to train the network and is quite slow when compared to GoogleNet and
ResNet. The VGG model takes around 7200 sec which is more than double the time what GoogleNet and ResNet take.

![image](https://user-images.githubusercontent.com/61591442/153454434-bc5a5eda-6468-499c-b1ce-da6e8a886d71.png)

Comparison between models
I. Below is the comparison chart for all the models with different learning rates.All the models has
performed really well with more than 99% accuracy result on test data. It is clearly evident that the
Googlenet has perfomed best on all and VGG16 has comparetivly performed least of all.

![image](https://user-images.githubusercontent.com/61591442/153454475-3e14830d-d0c7-40d0-9f6f-f3c402fde322.png)

![image](https://user-images.githubusercontent.com/61591442/153454527-530a0dce-7aa4-4d82-ab59-52978d394eec.png)
