# Dive into back propagation algorithm
![N|Solid](https://raw.githubusercontent.com/BoltzmannZhaung/Fashion-Boltzmann/master/img/logo5.png)
**Backpropagation**, aka backward propagation of errors, is an algorithm for supervised learning of artificial neural networks using gradient descent.Given an artificial neural network and an error function(loss), it calculates the gradient of the error function to update the weights of each layer.Here,this post is my attempt to explain how it works with a concrete example Iris Dataset. 

In this project, you will see:
  - How to build a shallow artificial nerual network?
  - How do forward and backward propagation work?
  - How to vectorize?
 
## 1.Introduction
### 1.1 Dataset
The data set Iris consists of 50 samples from each of three species of Iris (**Iris setosa**, **Iris virginica** and **Iris versicolor**). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.However,for simplifying task, I removed 50  'Iris virginica' samples to convert my job to a binary classification.Let's have a look at Iris data.<br>
**5.1,3.5,1.4,0.2,Iris-setosa** ---->the first sample<br>
**7.0,3.2,4.7,1.4,Iris-versicolor** ---->the second sample<br>
Given the first sample,we got **x1=5.1 x2=3.5 x3=4.0 x4=0.2** as 4 features and **y='Iris-setosa'**(after one_hot encoding **y=1.0**).<br>
For the second sample, **x1=7.0 x2=3.2 x3=4.7 x4=1.4 y=0.0** right?<br>
Finaly,100 samples should be divided up into training set(80 samples) and test set(20 samples).<br>

### 1.2 Network Architectures
For this tutorial, we’re going to use a neural network with four inputs corresponding to four features **x1 x2 x3 x4**, six hidden neurons, one output neuron. Additionally, the hidden and output neurons will include a bias.I didn't give the output layer a active funtion such as sigmod function,since I wanna simplify the derivate processing.Actually,the g(z) in hidden neurons is sigmod function.
![N|Solid](https://raw.githubusercontent.com/BoltzmannZhaung/Backpropagation/master/img/architecture.PNG)


## 2.Derivation
Given the first sampkle **5.1,3.5,1.4,0.2,Iris-setosa**,<br>
it can be converted to **x=[5.1,3.5,1.4,0.2],y=[1.0]**

##ON GOING...
