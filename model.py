
# containing the script to create and train the model


import csv
import cv2
import numpy as np
from skimage.io import imread
from sklearn.utils import shuffle as sklearnShuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from numpy.random import shuffle
from keras.models import Model
from keras.models import load_model


# ## Functions


# generator function for training
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples) #Shuffle the samples
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0] #extract file name
                # if flipped image
                if 'flipped' in name:
                    name = name.replace('flipped_', '')
                    image = imread("./data/IMG/" + name)
                    image = np.fliplr(image)
                #for other image
                else:
                    image = imread("./data/IMG/" + name)
                angle = float(batch_sample[1]) #extract steering angle
                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearnShuffle(X_train, y_train)


# function for spliting data in train data set and validation data set    
def split_samples(samples, test_size=0.1):
    train_samples, validation_samples = train_test_split(samples, test_size=test_size, random_state=0)
    
    return train_samples, validation_samples



# ## Data preprocessing and transformation



# Data transformation: flipped image, and use left, right, center cameras

# correction angle for left and right cameras
correction = 0.05

samples = []

with open('./data/driving_log.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
             
            if row[3] != 'steering': # sometimes first line is the column name, sometimes not
                img_center = row[0].split("\\")[-1]
                img_center = img_center.split("/")[-1]
                img_left = row[1].split("\\")[-1]
                img_left = img_left.split("/")[-1]
                img_right = row[2].split("\\")[-1]
                img_right = img_right.split("/")[-1]
            
                img_center_flipped = 'flipped_' + img_center # Add flipped string to image name
                img_left_flipped = 'flipped_' + img_left
                img_right_flipped = 'flipped_' + img_right
            
                steering_center = float(row[3]) # steering correction
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                if steering_left > 1.:
                    steering_left = 1.
                if steering_right < -1.:
                    steering_right = -1.
            
                steering_center_flipped = -steering_center #flipped correction
                steering_left_flipped = -steering_left
                steering_right_flipped = -steering_right
            
                samples.append([img_center, steering_center])
                samples.append([img_left, steering_left])
                samples.append([img_right, steering_right])
                samples.append([img_center_flipped, steering_center_flipped])
                samples.append([img_left_flipped, steering_left_flipped])
                samples.append([img_right_flipped, steering_right_flipped])

samples = np.array(samples)

train_samples, validation_samples = split_samples(samples)


# ## Modeling Neural Network, Training and evaluating
ex_name = samples[0,0]
ex_image = imread("./data/IMG/" + ex_name)
input_shape = ex_image.shape # Image shape

model = Sequential()
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=input_shape))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(5,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(10,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(20,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(30,3,3, activation="relu"))
model.add(Convolution2D(30,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1))

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(train_generator, samples_per_epoch= len(train_samples),
                              validation_data=validation_generator, nb_val_samples=len(validation_samples), 
                              nb_epoch=18, verbose = 1)


model.save('model.h5')
print('model saved')

