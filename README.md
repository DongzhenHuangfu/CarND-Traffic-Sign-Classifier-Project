# **Traffic Sign Recognition** 

## README

---
### Build a Traffic Sign Recognition Project**

The goals of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/data_bar.jpg "Data distribution in bar"
[image2]: ./examples/data_bar.png "Data distribution in table"
[image3]: ./download_picture/3.jpg "pic 1"
[image4]: ./download_picture/11.jpg "pic 2"
[image5]: ./download_picture/12.jpg "pic 3"
[image6]: ./download_picture/14.jpg "pic 4"
[image7]: ./download_picture/25.jpg "pic 5"
[image8]: ./download_picture/31.jpg "pic 6"
[image9]: .

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

You're reading it! and here is a link to my [project code](https://github.com/DongzhenHuangfu/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32*32*3
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributed

![alt text][image1]

To make it clear, I list them in a table:

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Description of the preprocess

Here, I decided not to convert the images to grayscale because the color is also an important factors for the traffic sign classifier, for example the speed limits and the end of the speed limits.

To preprocess the , I normalized the image data because it's will be much more easier to process the optimization if the data have a mean zero and equal variance.


#### 2. Description of my LeNet architecture.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 5x5 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x6 				    |
| Convolution 5x5	    | 5x5 stride, valid padding, outputs 10x10x16  	|
| RELU                  |                                               |
| Max pooling	      	| 2x2 stride, outputs 5x5x16 				    |
| Flatten    	      	| outputs 400 	             			        |
| Dropout    	      	| keep_prob 0.5              				    |
| Fully connected		| output 120   									|
| RELU					|												|
| Dropout    	      	| keep_prob 0.6              				    |
| Fully connected		| output 84   									|
| RELU					|												|
| Dropout    	      	| keep_prob 0.7              				    |
| Fully connected		| output 43   									|
| Softmax				| output 43   									|
 


#### 3. Training process.

To train the model, I used AdamOptimizer with learning rate 0.01, batch size of 128 and epochs I set it as 100, cause I wanted the accuracy to be more than 0.95, when it reach 0.95, the training process will stop automatically and save the data.

#### 4. Approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

To get a solution, I tried the structure of LeNet, which is proved to be a good structur for traffic signs classifier. In this structure, the input will be operated on two convolution process, which elimates the influence of the position.
To avoid over fitting, I added three dropout progresses before each fully connected layer, and tried to adjust the keep_prob.
As the results shows, this model is working well.


### Test a Model on New Images

Here are six German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7] ![alt text][image8]

The third and the fifth image might be difficult to classify because there are much more noise in the picture, and after resize the traffic sign will deform.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			                                    |     Prediction	      | 
|:-------------------------------------------------:|:-----------------------:| 
| Speed limit (60km/h)                              | Speed limit (60km/h)    | 
| Right-of-way   			                        | Right-of-way 	     	  |
| Priority road					                    | Priority road			  |
| Stop	      		                                | Stop					  |
| Road work			                                | Road work            	  |
| Wild animals crossing                             | Wild animals crossing	  |


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%.


The code for making predictions on my final model is located in the19th cell of the Ipython notebook.

To make it more clear, I also tried to show the top 5 of the softmax matrix:

For the first image, the model is relatively sure that this is a Speed limit (60km/h) (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.92         			| Stop sign   									| 
| 0.71     				| U-turn 										|
| 0.01					| Yield											|
| 0.003	      			| Bumpy Road					 				|
| 9.3e-08				| Slippery Road      							|


For the second image ... 

### Visualizing the Neural Network
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


