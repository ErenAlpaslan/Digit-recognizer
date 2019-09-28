# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 22:44:50 2019

@author: eren
"""

import numpy as np
import pandas as pd


train_set = pd.read_csv('train.csv')
y = train_set.label.values
train_set.drop(['label'], axis = 1, inplace = True)
test_set = pd.read_csv('test.csv')

# Normalization
train_set /= 255
test_set /= 255


X = train_set
X = X.values.reshape(-1,28,28,1)
test_set = test_set.values.reshape(-1,28,28,1)


from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y,random_state = 42, test_size = 0.1)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# One Hot encode our labels(Y)
from keras.utils import np_utils

# now we one hot encode output
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense,Dropout

classifier = Sequential()
classifier.add(Convolution2D(64, (3,3), padding = 'Same',activation = 'relu', input_shape = (28,28,1)))
classifier.add(Convolution2D(64, (3,3), padding = 'Same',activation = 'relu', input_shape = (28,28,1)))
classifier.add(MaxPooling2D(2,2))
classifier.add(Dropout(0.25))

classifier.add(Convolution2D(128, (3,3), padding = 'Same',activation = 'relu', input_shape =(28,28,1)))
classifier.add(Convolution2D(128, (3,3), padding = 'Same',activation = 'relu', input_shape = (28,28,1)))
classifier.add(MaxPooling2D(2,2))
classifier.add(Dropout(0.25))

classifier.add(Convolution2D(256, (3,3), padding = 'Same',activation = 'relu', input_shape = (28,28,1)))
classifier.add(Convolution2D(256, (3,3), padding = 'Same',activation = 'relu', input_shape = (28,28,1)))
classifier.add(MaxPooling2D(2,2))
classifier.add(Dropout(0.25))
classifier.add(Flatten())

classifier.add(Dense(output_dim = 512, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim = 10, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)
history = classifier.fit_generator(datagen.flow(X_train,y_train, batch_size=86),
                              epochs = 25, validation_data = (X_test,y_test),
                              verbose = 1, steps_per_epoch=X_train.shape[0] // 86)



score = classifier.evaluate(X_test, y_test, verbose = 0)
print('Test loss: ',score[0])
print('Test accuracy',score[1])


test_pred = classifier.predict(test_set)
test_pred = pd.DataFrame(test_pred)

prediction = test_pred.idxmax(axis = 1)


ImageId = []
for i in range(1,len(prediction)+1):
    ImageId.append(i)
    
  
prediction = pd.DataFrame(prediction)
ImageId = pd.DataFrame(ImageId)

submission = pd.concat([ImageId,prediction],axis = 1)
submission.columns = ['ImageId','Label']
submission.to_csv(r'submission6.csv', index = False)

import matplotlib.pyplot as plt
plt.plot(submission.Label)
plt.show()

 from keras.wrappers.scikit_learn import KerasClassifier
 from sklearn.model_selection import GridSearchCV
 
 def build(optimizer):
     classifier = Sequential()
     classifier.add(Dense(output_dim = 1400, activation = 'relu', input_dim = 784))
     classifier.add(Dropout(0.25))
     classifier.add(Dense(output_dim = 700, activation = 'relu'))
     classifier.add(Dropout(0.5))
     classifier.add(Dense(output_dim = 10, activation = 'softmax'))
     classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
     return classifier
 
 classifier = KerasClassifier(build_fn = build)
 
 #dictionary for hyperparameters
 parameters = {'batch_size' : [128, 256],
               'epochs' : [25, 10],
               'optimizer' : ['adam']
               }
 
 grid_search = GridSearchCV(estimator = classifier, param_grid = parameters,  cv = 10)
 grid_search = grid_search.fit(X_train,y_train)
 
 best_parameters = grid_search.best_params_
 best_accuracy = grid_search.best_score_


# save model
classifier.save('minst_ann.h5')
print('Model Saved')
# load model
#from keras.models import load_model
#classifier = load_model('minst_ann.h5')














