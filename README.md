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
[image3]: ./download_picture/0.jpg "pic 0"
[image4]: ./download_picture/3.jpg "pic 3"
[image5]: ./download_picture/11.jpg "pic 11"
[image6]: ./download_picture/12.jpg "pic 12"
[image7]: ./download_picture/14.jpg "pic 14"
[image8]: ./download_picture/19.jpg "pic 19"
[image9]: ./download_picture/21.jpg "pic 21"
[image10]: ./download_picture/24.jpg "pic 24"
[image11]: ./download_picture/25.jpg "pic 25"
[image12]: ./download_picture/27.jpg "pic 27"
[image13]: ./download_picture/30.jpg "pic 30"
[image14]: ./download_picture/31.jpg "pic 31"
[image15]: ./examples/precision_and_recall.png "precision of the test set"
[image16]: ./download_picture/softmax_0.jpg "softmax 0"
[image17]: ./download_picture/softmax_3.jpg "softmax 3"
[image18]: ./download_picture/softmax_11.jpg "softmax 11"
[image19]: ./download_picture/softmax_12.jpg "softmax 12"
[image20]: ./download_picture/softmax_14.jpg "softmax 14"
[image21]: ./download_picture/softmax_19.jpg "softmax 19"
[image22]: ./download_picture/softmax_21.jpg "softmax 21"
[image23]: ./download_picture/softmax_24.jpg "softmax 24"
[image24]: ./download_picture/softmax_25.jpg "softmax 25"
[image25]: ./download_picture/softmax_27.jpg "softmax 27"
[image26]: ./download_picture/softmax_30.jpg "softmax 30"
[image27]: ./download_picture/softmax_31.jpg "softmax 31"
[image28]: ./visualization/0_layer1.jpg "layer1 0"
[image29]: ./download_picture/3_layer1.jpg "layer1 3"
[image30]: ./download_picture/11_layer1.jpg "layer1 11"
[image31]: ./download_picture/12_layer1.jpg "layer1 12"
[image32]: ./download_picture/14_layer1.jpg "layer1 14"
[image33]: ./download_picture/19_layer1.jpg "layer1 19"
[image34]: ./download_picture/21_layer1.jpg "layer1 21"
[image35]: ./download_picture/24_layer1.jpg "layer1 24"
[image36]: ./download_picture/25_layer1.jpg "layer1 25"
[image37]: ./download_picture/27_layer1.jpg "layer1 27"
[image38]: ./download_picture/30_layer1.jpg "layer1 30"
[image39]: ./download_picture/31_layer1.jpg "layer1 31"

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
* training set accuracy of = 0.982
* validation set accuracy  = 0.952 
* test set accuracy of     = 0.933

To get a solution, I tried the structure of LeNet, which is proved to be a good structur for traffic signs classifier. In this structure, the input will be operated on two convolution process, which elimates the influence of the position.
To avoid over fitting, I added three dropout progresses before each fully connected layer, and tried to adjust the keep_prob.
As the results shows, this model is working well.


### Test a Model on New Images

The following is the precision, recall and F values of the test set:

![alt text][image15]

From the result we can see that the F value of the following signs are relevantly low, which means the prediction of these signs are relevantly not so reliable:
1. 27: Pedestrians - 62.14%  
2. 30: Beware of ice/snow - 62.84% 
3. 24: Road narrows on the right - 67.11% 
4. 21: Double curve - 67.57%
5. 41: End of no passing - 72.22%
6.  0: Speed limit (20km/h) - 75.25%
7. 19: Dangerous curve to the left - 77.55%

To test my model, I found the following traffic signs online:

![alt text][image3] ![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7] ![alt text][image8]
![alt text][image9] ![alt text][image10] ![alt text][image11]
![alt text][image12] ![alt text][image13] ![alt text][image14]

Here are the results of the prediction:

| Image			                        | ID | Prediction	               | ID | T/F |
|:-------------------------------------:|:--:|:---------------------------:|---:|----:|
| Speed limit (20km/h)                  |  0 | Slippery road               | 23 |  F  |
| Speed limit (60km/h)                  |  3 | Speed limit (60km/h)        |  3 |  T  | 
| Right-of-way at the next intersection | 11 | Right-of-way...             | 11 |  T  |
| Priority road				  	        | 12 | Priority road			   | 12 |  T  |
| Stop	      		                    | 14 | Stop					       | 14 |  T  |
| Dangerous curve to the left           | 19 | Dangerous curve to the left | 19 |  T  |
| Double curve                          | 21 | Double curve				   | 21 |  T  |
| Road narrows on the right             | 24 | Priority road               | 12 |  F  |
| Road work                             | 25 | Road work              	   | 25 |  T  |
| Pedestrians                           | 27 | Road narrows on the right   | 24 |  F  |
| Beware of ice/snow                    | 30 | Right-of-way...       	   | 11 |  F  |
| Wild animals crossing                 | 31 | Wild animals crossing	   | 31 |  T  |


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 66.7%.

To make it more clear, I also tried to show the top 5 of the softmax matrix:


![alt text][image16] ![alt text][image17] ![alt text][image18] 
![alt text][image19] ![alt text][image20] ![alt text][image21]
![alt text][image22] ![alt text][image23] ![alt text][image24]
![alt text][image25] ![alt text][image26] ![alt text][image27]

For the 1st image, the model totally wrong, it is not sure and says that this maybe a 'Slippery road'(probability of 0.408), and the image actually contain a 'Speed limit (20km/h)'. and the top five soft max contains also not the right answer. Corresponding to the 63.33% precision rate of 'Speed limit (20km/h)' and 80.47% recall rate of 'Slippery road' in test set.

For the 2nd image, the model is quite sure that this is a 'Speed limit (60km/h)'(probability of 0.919), and the image does contain a 'Speed limit (60km/h)'. Corresponding to the 95.56% precision rate of 'Speed limit (60km/h)' in test set.

For the 3rd image, the model is very sure that this is a 'Right-of-way at the next intersection'(probability of 0.997), and the image does contain a 'Right-of-way at the next intersection'. Corresponding to the 92.62% precision rate of 'Right-of-way at the next intersection' in test set.

For the 4th image, the model is very sure that this is a 'Priority road'(probability of 0.999), and the image does contain a 'Priority road'. Corresponding to the 96.38% precision rate of 'Priority road' in test set.

For the 5th image, the model is pretty sure that this is a 'Stop'(probability of 0.950), and the image does contain a 'Stop'. Corresponding to the 99.63% precision rate of 'Stop' in test set.

For the 6th image, the model is not sure that this is a 'Dangerous curve to the left'(probability of 0.634), and the image does contain a 'Dangerous curve to the left'. Corresponding to the 63.33% precision rate of 'Dangerous curve to the left' in test set.

For the 7th image, the model is pretty sure that this is a 'Double curve'(probability of 0.961), and the image does contain a 'Double curve'. Corresponding to the 55.56% precision rate of 'Double curve' in test set.

For the 8th image, the model is quite sure that this is a 'Road narrows on the right'(probability of 0.984), and the image does contain a 'Road narrows on the right'. Corresponding to the 55.56% precision rate of 'Road narrows on the right' in test set.

For the 9th image, the model is very sure that this is a 'Road work'(probability of 1.000), and the image does contain a 'Road work'. Corresponding to the 95.42% precision rate of 'Road work' in test set.

For the 10th image, the model is not very sure that this is a 'Road narrows on the right'(probability of 0.712), and the image does contain a 'Pedestrians'. In the top five soft max it does contains 'Pedestrians' with probability of 0.194. Corresponding to the 53.33% precision rate of 'Pedestrians' and 84.75% recall rate of 'Road narrows on the right' in test set.

For the 11th image, the model totally wrong, it is not very sure that this is a 'Right-of-way at the next intersection'(probability of 0.573), and the image does not contain a 'Beware of ice/snow'. Corresponding to the 62.00% precision rate of 'Beware of ice/snow' and 88.81% recall rate of 'Right-of-way at the next intersection' in test set.

For the 12th image, the model is very sure that this is a 'Wild animals crossing'(probability of 1.000), and the image does contain a 'Wild animals crossing'. Corresponding to the 95.56% precision rate of 'Wild animals crossing' in test set.

### Visualizing the Neural Network
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
The visual output of the first layer are like the follows:

![alt text][image28] 
![alt text][image29] 
![alt text][image30] 
![alt text][image31] 
![alt text][image32] 
![alt text][image33]
![alt text][image34] 
![alt text][image35] 
![alt text][image36]
![alt text][image37] 
![alt text][image38] 
![alt text][image39]

We can find that it takes the shape of the sign and also the content into account when doing a prediction.