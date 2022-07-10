# Fully-Connected-NN
Implementation of Fully Connected Neural Network with two layers using Numpy and comparing results to PyTorch model on the MNIST-Digits dataset

# Porgram Description
In this project I implemented a two-layer neural network with one hidden layer.  
And trained it on the MNIST-Digits dataset than compered the result with a PyTorch model.

### Neural Network
The first layer is the input layer, which receives a flattened image of 28x28 (pixels) = 784 input cells in our case.<br/>
Notice that when we say N-layer neural network, we do not count the input layer (because it is not trainable).<br/>
The input layer is linked to a "hidden layer" with full connectivity ("fully connected layer").<br/>
The hidden layer contains hidden units (or cells), the amount of hidden units is configurable.<br/>
The last layer is the output layer.<br/>
Because we want to classify the output image into a digit between 0 and 9, the output layer will include 10 output cells.<br/>
Each output neuron corresponds to a certain class (digit). The output layer is also fully connected.<br/>
![image](https://user-images.githubusercontent.com/108329249/178142182-3061310e-8fd7-4c51-b89b-ff7da0aa46f0.png)




### Neuron Structure
Each cell in the network (called a neuron) receives a number of weighed inputs, sums
them up and passes them through a nonlinear function (such as sigmoid, hyperbolic
tangent, RELU, etc.), which is called "activation function".<br/>
In addition, one of the inputs to each cell is the bias (which is not dependent on the previous layers).<br/>
![image](https://user-images.githubusercontent.com/108329249/178142184-20be318d-c038-40ae-9bd9-a546df0a31c9.png)

<br/>
In this project I used the Sigmoid function for the activation function:<br/>
σ(𝑧) = 1 / (1+𝑒^−𝑧)
<br/>
Where z is the output of the fully connected layer:<br/>
𝑧 = 𝑤∙𝑥 + 𝑏
<br/>
Where x is the output from the previous layer (or the input of the first layer), b is the
bias.
<br/>
The Sigmoid's derivative:<br/>
σ'(𝑧) = σ(𝑧)∙(1 − σ(𝑧))

### Learning Algorithm
In the learning process, the weights and bias inputs are updated to map the input
image to the correct output cell corresponding to the class represented by the image
(the digit in the image).<br/>
The process contains three main parts that are repeated:<br/>

#### Forward Propagation
The information from the input layer passes through the network to the cells in the
hidden layer, and the output of the hidden cells continues to propagate to the output
layer (the network's output is our "hypothesis", it is a vector marked by h).<br/>
Next we compare the network's output with the Ground Truth (GT) digit represented
in the image, computing the networks error with the loss function.<br/>
We will be using Softmax classifier on our output layer, it takes a vector of arbitrary
real-valued scores (in z) and squashes it to a vector of values between zero and one
that sum to one (similar to probabilities).<br/>
For each output cell, the softmax function:<br/>
![image](https://user-images.githubusercontent.com/108329249/178142384-4e42c85c-e8f4-4942-ba36-a32cb45b9993.png)

Cross Entropy Loss: indicates the distance between what the model
believes the output distribution should be, and what the original distribution reallyis.<br/>
It is defined as:

![image](https://user-images.githubusercontent.com/108329249/178142520-a30cc40b-fb14-4794-a43f-119b427ea565.png)

Where j is the index of the correct class.<br/>
Cross entropy measure is a widely used alternative of squared error.<br/>

#### Back Propagation
For the optimization purpose, we will want to change the neural network's parameters so that we minimize the error it produces, in other words, we will want
the output class to be as close as possible to the GT class.<br/>
Back propagation allows the information to go back from the cost backward through the network in order to compute the gradients needed for the optimization.<br/>
We will use Gradient Descent (or, more precisely, Stochastic Gradient Descent because we update the network as
we enter the data – for each 'minibatch') to update the NN's weights.<br/>
Therefore, first (back propping from the end) - we will derive the error as a function of the network's output, where t is a 'one-hot' vector with t(GT digit) = 1 and
t(elsewhere) = 0.<br/>

![image](https://user-images.githubusercontent.com/108329249/178142589-b2c0f187-85db-4119-800f-57ddf435a5b8.png)


Continue to 'back-prop' by yourselves, using the ‘chain rule’ for derivatives.<br/>
Example of the chain rule:  

![image](https://user-images.githubusercontent.com/108329249/178142613-95a69f85-0124-4cb5-8bd8-ae7a1f2236c9.png)


Where 𝑓(𝑟) and 𝑟(𝑠).<br/>
The purpose of the backpropagation is to find the gradient of the cost function with
respect to the networks parameters (weights and biases):<br/>  

![image](https://user-images.githubusercontent.com/108329249/178142645-88f949ca-75fd-4f83-89b2-a3a07525c369.png)

#### Update
Now that we have calculated the derivatives, we can update the weights so that they minimize the error with the gradient descent algorithm:  

![image](https://user-images.githubusercontent.com/108329249/178142699-bd8a59e9-d638-44d2-adc4-569681625206.png)


Where μ is the learning rate.<br/>
Similarly for the bias:  

![image](https://user-images.githubusercontent.com/108329249/178142755-c01910f0-5de0-4da8-8320-69a41063c848.png)


This process is done for each 'minibatch' of the training data and the process is done
iteratively until the network converges.

### Results
#### Test Accuracy 97%
![image](https://user-images.githubusercontent.com/108329249/178142846-e0cbe8aa-065d-4601-8a2b-319922af1f7b.png)


