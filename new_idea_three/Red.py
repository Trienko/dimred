import scipy.io
import pylab as plt
from sklearn.decomposition import PCA
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from matplotlib.patches import Ellipse
from termcolor import colored
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import pickle
import tensorflow as tf 
from tensorflow import keras     
import math
#from keras.layers import Lambda

from sklearn import mixture

class RedClass():

  def __init__(self):
      pass

  def loadDataSet(self,name="Gauteng_nochange.mat",province="Gauteng"):
      print("Loading Dataset")

      mat = scipy.io.loadmat(name)
      #print(mat.keys())

      if province == "Gauteng":
         veg = mat['veg_gauteng']
         bwt = mat['bwt_gauteng']
    
      #(time,pixels,band)
      #(time,pixels,7) - NDVI
      return veg,bwt

  def concatDataSets(self,veg,bwt):
      print("Concatting data set...")
      chan = veg.shape[2]
      timeslots = veg.shape[0]
      v_pixels = veg.shape[1]
      b_pixels = bwt.shape[1] 

      X_concat = np.zeros((v_pixels+b_pixels,timeslots,chan),dtype=float)

      for c in range(chan): 
        Xv = veg[:,:,c].T
        Xb = bwt[:,:,c].T  	

        X_concat[:,:,c] = np.vstack((Xv,Xb))

      yv = np.ones((v_pixels,1),dtype=int)
      yb = np.zeros((b_pixels,1),dtype=int)
      y = np.vstack((yv,yb))[:,0]
 
      return X_concat,y

  def exp1(self,X_train,X_valid,y_train,y_valid):

      #BUILDING THE MODEL
      model = keras.models.Sequential()
      model.add(keras.layers.Flatten(input_shape=[368,7]))
      #model.add(keras.layers.Dense(368,activation="relu"))
      model.add(keras.layers.Dense(100,activation="relu"))
      model.add(keras.layers.Dense(1,activation="sigmoid"))
      model.summary() 

      #COMPILE
      model.compile(loss="binary_crossentropy",optimizer="sgd",metrics=["accuracy"])

      #FIT
      history = model.fit(X_train,y_train,epochs=10,batch_size=1,validation_data=(X_valid,y_valid))

  def exp2(self,X_train,X_valid,y_train,y_valid):

      #Adding an extra dimension to use convolution layer
      X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
      X_valid = X_valid.reshape((X_valid.shape[0],X_valid.shape[1],X_valid.shape[2],1))

      #BUILDING THE MODEL
      model = keras.models.Sequential()
      model.add(keras.layers.Conv2D(1,(1,7),activation="tanh",input_shape=[368,7,1],padding="valid"))
      model.add(keras.layers.Flatten())
      #model.add(keras.layers.Dense(368,activation="relu"))
      model.add(keras.layers.Dense(100,activation="relu"))
      model.add(keras.layers.Dense(1,activation="sigmoid"))
      model.summary() 

      #COMPILE
      model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

      #FIT
      history = model.fit(X_train,y_train,epochs=10,batch_size=1,validation_data=(X_valid,y_valid))

  def exp3(self,X_train,X_valid,y_train,y_valid):

      #Adding an extra dimension to use convolution layer
      X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
      X_valid = X_valid.reshape((X_valid.shape[0],X_valid.shape[1],X_valid.shape[2],1))

      #Division layer
      #result = keraLambda(lambda inputs: inputs[0] / inputs[1])([input1, input2])


      #USING THE FUNCTIONAL API
      input_ = keras.layers.Input(shape=X_train.shape[1:])
      c1 = keras.layers.Conv2D(1,(1,7),activation=None,input_shape=[368,7,1],padding="valid")(input_)
      fc1 = keras.layers.Flatten()(c1)
      c2 = keras.layers.Conv2D(1,(1,7),activation="tanh",input_shape=[368,7,1],padding="valid")(input_)
      fc2 = keras.layers.Flatten()(c2)
      result = keras.layers.Lambda(lambda inv: inv[0] / inv[1])([fc1, fc2])
      idx = keras.layers.Activation("tanh")(result)
      h = keras.layers.Dense(100,activation="relu")(idx)
      output = keras.layers.Dense(1,activation="sigmoid")(h)
      model = keras.Model(input_,output)
      model.summary()

      #COMPILE
      model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

      #FIT
      history = model.fit(X_train,y_train,epochs=10,batch_size=1,validation_data=(X_valid,y_valid))

   
if __name__ == "__main__":
   np.random.seed(30)
            
   red_object = RedClass() 
     
   #LOADING DATASET
   veg,bwt = red_object.loadDataSet()
   #print(veg.shape)

   #CONCAT_DATASETS
   X,y = red_object.concatDataSets(veg,bwt)
   #print(X.shape)

   X = X[:,:,:-1]/10000.0

   plt.plot(X[0,:,0])
   plt.show()

   idx = np.random.randint(2,size=X.shape[0])

   #print(idx)

   X_train, X_valid, y_train, y_valid = X[idx==0,:,:],X[idx==1,:,:],y[idx==0],y[idx==1]

   print(X_train.shape)
   #print(X_valid.shape)
   
   print(tf.__version__)
   
   red_object.exp1(X_train,X_valid,y_train,y_valid) 
   red_object.exp2(X_train,X_valid,y_train,y_valid) 
   red_object.exp3(X_train,X_valid,y_train,y_valid) 
   
