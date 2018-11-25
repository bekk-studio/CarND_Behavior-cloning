# Project 3: Use Deep Learning to Clone Driving Behavior

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for P3, Behavioral Cloning.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument `run1` is the directory to save the images seen by the agent to. If the directory already exists it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp when the image image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Create a video based on images found in the `run1` directory. The name of the video will be name of the directory following by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

The video will run at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).


----------------------

# **Behavioral Cloning Reporting** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report/croppedimage.png "Cropped Image"
[image2]: ./report/eximage.png "Example image"
[image3]: ./report/distribution1.png "Data distribution"
[image4]: ./report/sideimages.png "3 images, Left, Center, Right"
[image5]: ./report/flippedimage.png "Flipped image"
[image6]: ./report/augdistribution1.png "Augmented data distribution"
[image7]: ./report/MSELplot1.png "MSE Loss plot"
[image8]: ./report/predictiondist1.png "Prediction distribution"
[video1]: ./report/run1.mp4 "Video with first model"
[image9]: ./report/eximage2.png "Example image"
[image10]: ./report/distribution2.png "Data distribution"
[image11]: ./report/augdistribution2.png "Augmented data distribution"
[image12]: ./report/MSELplot2.png "MSE Loss plot"
[image13]: ./report/predictiondist2.png "Prediction distribution"
[video2]: ./report/run2.mp4 "Video with second model"
[video3]: ./report/run22.mp4 "Video with second model"
[image14]: ./report/eximage3.png "Example image"
[image15]: ./report/distribution3.png "Data distribution"
[image16]: ./report/augdistribution3.png "Augmented data distribution"
[image17]: ./report/MSELplot3.png "MSE Loss plot"
[image18]: ./report/predictiondist3.png "Prediction distribution"
[video4]: ./report/run3.mp4 "Video with third model"
[video5]: ./report/run32.mp4 "Video with third model"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.html summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 or 3x3 filter sizes and depths between 5 and 30 (model.py lines 119-126) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 118). 

#### 2. Attempts to reduce overfitting in the model

The model contains **Maxpooling layers** in order to reduce overfitting (model.py lines 120, 122, 124).

The complexity of model and depht of layers are balanced between learning efficiency and overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 137).

The loss is a mean square error because problem is a regression problem. (It could be possible to treat the problem like a classification problem.)

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, smooth curving, inverse way on the track.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to go from Nvidia model architecture and adapted to the data.

My first step was to use a convolution neural network model similar to the Nvidia solution. I thought this model might be appropriate because it was used for similar project. But image should be more complex for Nvidia, then model could be simplified. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.  I found that the model was sometimes overfitting or underfitting. 

To combat the overfitting or underfitting, I modified the model so that I add, modify, delete some layers. I try to use Dropout layer, Maxpooling Layer. I try transfer learning by using Inception V3.

Then I compare solution by evaluating the training with validation loss. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. to improve the driving behavior in these cases, I search to have a better model.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 116-131) consisted of a convolution neural network very close to the Nvidia model. I fail to have something better well. I just decrease the deepht of the convolutional layers and add Maxpooling layer.

Here is the summary of the architecture:


|Layer (type)						|Output Shape          |Param #   |                       
|:---------------------------------:|:--------------------:|:--------:|
|Input                              | (None, 80, 320, 3)   |          |
|Cropping2D   ((60,20), (0,0))      | (None, 80, 320, 3)   | 0        |         
|Lambda: x: (x / 255.0) - 0.5)      | (None, 80, 320, 3)   | 0        |               
|Convolution2D 5x5x5 "relu"         | (None, 76, 316, 5)   | 380      |                     
|MaxPooling2D                       | (None, 38, 158, 5)   | 0        |           
|Convolution2D 10x5x5 "relu"        | (None, 34, 154, 10)  | 1260     |             
|MaxPooling2D                       | (None, 17, 77, 10)   | 0        |           
|Convolution2D 20x5x5 "relu"        | (None, 13, 73, 20)   | 5020     |             
|MaxPooling2D                       | (None, 6, 36, 20)    | 0        |           
|Convolution2D 30x3x3 "relu"        | (None, 4, 34, 30)    | 5430     |             
|Convolution2D 30x3x3 "relu"        | (None, 2, 32, 30)    | 8130     |           
|Flatten                            | (None, 1920)         | 0        |           
|Dense 100 "relu"                   | (None, 100)          | 192100   |                  
|Dense 50 "relu"                    | (None, 50)           | 5050    |               
|Dense 10 "relu"                    | (None, 10)           | 510      |             
|Dense 1                            | (None, 1)            | 11       |                   


Total params: 217,891

Trainable params: 217,891

Non-trainable params: 0

Note that, inside the model, image is cropped to focus on the road view.

![cropped image][image1]

#### 3. Creation of the Training Set & Training Process

##### * First time, I use the provided data in the project ressources.  

Here is an example image of data set:

![example image][image2]

Here is the data distribution of center image:

![Data distribution 1][image3]

I have **8036 samples**.

I then augment the data with left and right image so that the vehicle would learn to back the center. Steering angle correction for coming back to center is set at 0.05 (1.25°). These images show left, center, right image :

![side image][image4]

To augment the data set, I also flipped images and angles thinking that this would generalize data and balancing example between left and right curves. For example, here is an image that has then been flipped:

![flipped image][image5]

Here is the data distribution after data augmentation:

![Augmented data distribution 1][image6]

I have now **48216 samples**.

I then preprocessed this data just by normalize it.

I finally randomly shuffled the data set and put **0.05%** of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. **The ideal number of epochs was 11** as evidenced by the validation plot minimum. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Here is the mean squarred error loss plot:

![model mean squarred error loss plot][image7]

Here is a prediction of model on the validation set:

![prediction][image8]

Here is the video result with this data set:

![Video 1][video1]

##### * Second time, I capture behavior by simulator on track one.

To capture good driving behavior, I first recorded 4 laps on track one using center lane driving and one lap in inverse turn. Here is an example image of center lane driving:

![Center image][image9]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to center back.

Here is the data distribution of center image:

![Data distribution 2][image10]

I have **9712 samples**.

Then I proceed like before.

Here is the data distribution after data augmentation:

![Augmented data distribution 2][image11]

I have now **58272 samples**.

Here is the mean squarred error loss plot:

![model mean squarred error loss plot][image12]

**The ideal number of epochs was 18** as evidenced by learning (validation loss decrease) before this point, and flat after.

Here is a prediction of model on the validation set:

![prediction][image13]

Here is the video result with this data set for track one. Lap of track is completed.

![Video 2][video2]

Here is the video result with this data set for track two. Car ND is inefficient. Data has to be generalized.

![Video 3][video3]

##### * Finally, I repeated this process on track two in order to get more data points and have experience on track two.

I recorded 3 laps on track two.

Here is an example image of center lane driving on track two:

![Center image][image14]

Here is the data distribution of center image:

![Data distribution 3][image15]

I have **15679 samples**.

Here is the data distribution after data augmentation:

![Augmented data distribution 3][image16]

I have now **94074 samples**.

Here is the mean squarred error loss plot:

![model mean squarred error loss plot][image17]

**The ideal number of epochs was 18**

Here is a prediction of model on the validation set:

![prediction][image18]

Here is the video result with this data set for track one. Lap of track is completed.

![Video 4][video4]

Here is the video result with this data set for track two. Car ND is inefficient. Data has to be generalized.

![Video 5][video5]

### Tries and Troubles during project

* I lose many times because of simulator understanding and drive file setting.
* I succeed to drive correctly the **Simulator** by using mouse, and setting the lowest parameter.
* In the drive file, speed is preponderant for success of self driving because of the limit of my computer.
* I can't find some improvement of model with preprocessing: YUV, resizing, bluring, canny.
* I can't find some improvement of model according to overfitting with dropout.
* But I think improvement is possible with better data input set. And adapted preprocessing.  
