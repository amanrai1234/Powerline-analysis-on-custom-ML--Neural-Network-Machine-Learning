# Project-6-Multi-layer_perceptron



# PART 1:
In this task I am going to comapre the  linear regression and at least three different MLP(multi-layer 
perceptron’s) and then I am comparing the performance of the three 
models and linear regression.


# Dataset: 

The dataset contains the salary of people (data scientists) across united states.
Visualization and reading of dataset: 

I have used pandas to read and display first five elements (dataframe.head()) as 
It’s easier to visualization compared to the NumPy. Then to get the information 
of the dataset contents, I have used pandas.info(), we can use this but we should 
always check the datset manually also. But sometimes the dataset might be too 
big to open, in that case we can use these kinds of techniques to do the 
computation.

# Finding Correlations:

Here I have used the .Corr() function from the pandas that gives the correlation 
of the elements in the dataset as the output. Here we can check the correlation of 
the dataset on the single element or on the entire dataset. In the Notebook, I 
have also plotted this using seaborn library. Here to plot the normal distribution 
of the dataset I have used the histograms and plotted the distribution of all the 
features in a single block.

# Data Cleaning:

This is the most challenging part of the dataset and here I have again used 
pandas to clean the data, I have searched for special characters also so that I 
could remove them and make the data less redundant. Here I removed the 
columns(features) that are having text(sentence like) because categorical 
features with only one or two unique words can be one-hot encoded. 
Here we have to note that the values that I have chosen to do cleaning are the 
ones which had a good positive or negative correlation with the target variables.
If there is close to zero correlation between features and the target values then it 
is not a wise choice to keep the feature in the training set.

# Data Division:

I have divided the data into numerical and categorical features, I did this 
because we need to scale the numerical values and one hot encode the 
categorical features. After the division of data and doing One hot encoding 
+Scaling (non-categorical variables), I have concatenated the dataset again and 
split it in training and testing sets.

# Linear Regression:

Now, the data cleaning part is completed, this is the time where we have to do 
linear regression. I have used Sckit-learn library of linear Regression to do this 
task.

After fitting the dataset on the data and target variable I perform the below three 
tasks for evaluation of the model using the scikit learn:

1] mean square error

2] mean square absolute error 

# MLP layers:

Multilayer perceptron is the same as building dense layer of the neural network. 
According to the problem solution we have create three custom models and we 
are doing exactly that. 
# Model1:

Type: sequential() #keras

Layer1:

Activation: relu

Number of neurons:50

Input size: same as shape of the data

Layer2:

Activation: relu

Number of neurons:25

Layer3:

Activation: Linear

Number of neurons:1

Optimizer: Adam

Metrics: mean absolute square error

# Model2:

Type: sequential() #keras

Layer1:

Activation: relu

Number of neurons:100

Input size: same as shape of the data

Layer2:

Activation: relu

Number of neurons:75

Layer3:

Activation: relu

Number of neurons:25

Layer4:

Activation: linear

Number of neurons:1

Optimizer: Adam

Learning rate: 0.01

Metrics: mean absolute square error

# Model3:

Type: sequential () #keras

Layer1:

Activation: LeakyReLU

Number of neurons:50

Layer2:

Activation: LeakyReLU

Number of neurons:25

Layer3:

Activation: linear

Number of neurons: 1

Optimizer: RMSprop

Metrics: mean absolute square error

# Comparison of the models:

Here I have trained the classical linear regression and the three MLP models 
and based on the result I got I can fairly say that the MLP models 
outperformed the Linear Regression of the sklearn, I believe this happened due 
to the fact that a neural network trains on taking the number of parameters 
and the number of layers and neurons make it better for making decisions due 
to the interconnect between layers and neurons.

Model 1 error: 0.4487923800049779

Model 2 error: 0.5831369918981913

Model 3 error: 0.4221451177059507

In the above three models I have seen that the model
1 and model 3 out-performed the model2. In these mode
ls the accuracy of the last model increased by changi
ng the activation function to leaky RElU but the accu
racy of the second model did not change because here 
we are not following the rules of the how the hidden 
layers (number of neurons are chosen), usually the ru
les are defined as:

1. The number of hidden neurons should be between the size of the input layer
and the size of the output layer.

2. The number of hidden neurons should be 2/3 the size of the input layer,
plus the size of the output layer.

3. The number of hidden neurons should be less than twice the size of the
input layer.

According to the source on internet (stack overflow), 
but here we are allowed to change and tweak the thing
s so, I changed the parameters drastically and then c
alculate the error, it was around 0.67, then I used t
he dropout layer and then the error got to be around
0.58.

The best models were the first and third and the only 
difference between them was the activation function (
RELU vs Leaky RELU)

In the real world we need to improve our model more a
nd to do that we can use better data augmentation tec
hniques where we can convert the textual data (catego
rical values that can’t be converted to One hot encod
e) to numerical values using word2vec or bag of words
or other approaches encode them and apart from that w
e can increase the data if possible as it can only be
nefit the model in terms of training.


