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
      model.add(keras.layers.Dense(100,activation="relu",kernel_initializer="he_normal",kernel_regularizer=keras.regularizers.l2(1e-3)))
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
      model.add(keras.layers.Conv2D(1,(1,7),activation="tanh",input_shape=[368,7,1],padding="valid",kernel_initializer="glorot_normal",kernel_regularizer=keras.regularizers.l1_l2(1e-3,1e-3)))
      model.add(keras.layers.Flatten())
      #model.add(keras.layers.Dense(368,activation="relu"))
      model.add(keras.layers.Dense(100,activation="relu",kernel_initializer="he_normal",kernel_regularizer=keras.regularizers.l2(1e-3)))
      model.add(keras.layers.Dense(1,activation="sigmoid"))
      model.summary() 

      #COMPILE
      model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

      

      #FIT
      history = model.fit(X_train,y_train,epochs=10,batch_size=1,validation_data=(X_valid,y_valid))

      w,b = model.layers[0].get_weights()
      print(w[0,:,0,0])
      print(b)

  def exp3(self,X_train,X_valid,y_train,y_valid):

      #Adding an extra dimension to use convolution layer
      X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
      X_valid = X_valid.reshape((X_valid.shape[0],X_valid.shape[1],X_valid.shape[2],1))

      #Division layer
      #result = keraLambda(lambda inputs: inputs[0] / inputs[1])([input1, input2])


      #USING THE FUNCTIONAL API
      input_ = keras.layers.Input(shape=X_train.shape[1:])
      c1 = keras.layers.Conv2D(1,(1,7),activation=None,input_shape=[368,7,1],padding="valid",kernel_initializer="glorot_normal",kernel_regularizer=keras.regularizers.l1_l2(1e-2,1e-2))(input_)
      fc1 = keras.layers.Flatten()(c1)
      c2 = keras.layers.Conv2D(1,(1,7),activation="tanh",input_shape=[368,7,1],padding="valid",kernel_initializer="glorot_normal",kernel_regularizer=keras.regularizers.l1_l2(1e-2,1e-2))(input_)
      fc2 = keras.layers.Flatten()(c2)
      result = keras.layers.Lambda(lambda inv: inv[0] / inv[1])([fc1, fc2])
      idx = keras.layers.Activation("tanh")(result)
      h = keras.layers.Dense(100,activation="relu",kernel_initializer="he_normal",kernel_regularizer=keras.regularizers.l2(1e-3))(idx)
      output = keras.layers.Dense(1,activation="sigmoid")(h)
      model = keras.Model(input_,output)
      model.summary()

      #COMPILE
      model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

      #FIT
      history = model.fit(X_train,y_train,epochs=10,batch_size=1,validation_data=(X_valid,y_valid))
      w,b = model.layers[1].get_weights()
      print(w.shape)
      print(w[0,:,0,0])
      print(b)
      w,b = model.layers[2].get_weights()
      print(w.shape)
      print(w[0,:,0,0])
      print(b)

  def exp4(self,X_train,X_valid,y_train,y_valid):

      #Adding an extra dimension to use convolution layer
      X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
      X_valid = X_valid.reshape((X_valid.shape[0],X_valid.shape[1],X_valid.shape[2],1))

      #Division layer
      #result = keraLambda(lambda inputs: inputs[0] / inputs[1])([input1, input2])


      #USING THE FUNCTIONAL API
      #1e-5 obtained good values for these hyperparameters
      input_ = keras.layers.Input(shape=X_train.shape[1:])
      r = 0

      c1 = keras.layers.Conv2D(1,(1,7),activation=None,input_shape=[368,7,1],padding="valid",kernel_initializer="glorot_normal",kernel_regularizer=keras.regularizers.l1_l2(r,r),kernel_constraint=keras.constraints.max_norm(1))(input_)
      #c1 = keras.layers.Conv2D(1,(1,7),activation=None,input_shape=[368,7,1],padding="valid",kernel_initializer="glorot_normal")(input_)
      fc1 = keras.layers.Flatten()(c1)
      c2 = keras.layers.Conv2D(1,(1,7),activation="tanh",input_shape=[368,7,1],padding="valid",kernel_initializer="glorot_normal",kernel_regularizer=keras.regularizers.l1_l2(r,r),kernel_constraint=keras.constraints.max_norm(1))(input_)
      #c2 = keras.layers.Conv2D(1,(1,7),activation="tanh",input_shape=[368,7,1],padding="valid",kernel_initializer="glorot_normal")(input_)
      fc2 = keras.layers.Flatten()(c2)
      result = keras.layers.Lambda(lambda inv: inv[0] / inv[1])([fc1, fc2])
      idx = keras.layers.Activation("tanh")(result)
      h = keras.layers.Dense(100,activation="relu",kernel_initializer="he_normal",kernel_regularizer=keras.regularizers.l2(r))(idx)
      output = keras.layers.Dense(1,activation="sigmoid")(h)
      model = keras.Model(input_,output)
      model.summary()

      #COMPILE
      model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

      #FIT
      history = model.fit(X_train,y_train,epochs=10,batch_size=1,validation_data=(X_valid,y_valid))
      w,b = model.layers[1].get_weights()
      w1 = np.array(w[0,:,0,0])
      print(w1)
      b1 = b[0]
      #print(w.shape)
      #print(w[0,:,0,0])
      #print(b)
      w,b = model.layers[2].get_weights()
      w2 = np.array(w[0,:,0,0])
      print(w2)
      b2 = b[0]
      #print(w.shape)
      #print(w[0,:,0,0])
      #print(b)
      X_new = self.process_data2(X_train,w1,b1,w2,b2)
      
      X_veg = X_new[y_train==1,:]
      X_set = X_new[y_train==0,:]

      plt.plot(X_veg[0,:],"r")
      plt.plot(X_set[0,:],"b")
      plt.show()

      v = np.mean(X_veg,axis=0)
      s = np.mean(X_set,axis=0)

      v_std = np.std(X_veg,axis=0)
      s_std = np.std(X_set,axis=0)

      plt.plot(v,"r")
      #plt.plot(v+v_std,"r:")
      #plt.plot(v-v_std,"r:")
      plt.plot(s,"b")
      #plt.plot(s+s_std,"b:")
      #plt.plot(s-s_std,"b:")
      plt.show()

      plt.plot(np.absolute(v-s))
      plt.show()
     
      t = np.argsort(np.absolute(v-s))[-2:]
       
      

      for k in range(X_new.shape[0]):
          if k % 2 == 0:
             if y_train[k] == 0:
                plt.plot(X_new[k,t[0]],X_new[k,t[1]],"ro",alpha=0.4,ms=10.0)
             else:
                plt.plot(X_new[k,t[0]],X_new[k,t[1]],"bo",alpha=0.4,ms=10.0)

      plt.show()
         

      '''
      pca = PCA(n_components=X_new.shape[1],whiten=False)
      X_T = pca.fit(X_new).transform(X_new)

      for k in range(X_new.shape[0]):
          if k % 2 == 0:
             if y_train[k] == 0:
                plt.plot(X_T[k,0],X_T[k,1],"ro",alpha=0.4,ms=10.0)
             else:
                plt.plot(X_T[k,0],X_T[k,1],"bo",alpha=0.4,ms=10.0)

      plt.show()
      '''


  def process_data(self,X,w,b):
      newX = np.zeros((X.shape[0],X.shape[1]),dtype=float)
      for k in range(X.shape[0]):
          for i in range(X.shape[1]):
              newX[k,i] = np.tanh(np.sum(X[k,i,:]*w)+b) 
      return newX

  def process_data2(self,X,w1,b1,w2,b2):
      newX = np.zeros((X.shape[0],X.shape[1]),dtype=float)
      for k in range(X.shape[0]):
          for i in range(X.shape[1]):
              #newX[k,i] = ((np.sum(X[k,i,:]*w1)+b1)/(np.sum(X[k,i,:]*w2)+b2))
              newX[k,i] = np.tanh((np.sum(X[k,i,:]*w1)+b1)/(np.sum(X[k,i,:]*w2)+b2)) 
      return newX

  def filter_data(self,X,w):

      X_new = np.zeros(X.shape,dtype=float)

      for k in range(X.shape[0]):
          for b in range(X.shape[2]):
                f_temp = np.fft.fft(X[k,:,b])
                f_temp[w:-w] = 0
                X_new[k,:,b] = np.real(np.fft.ifft(f_temp))
      return X_new   
      

   
if __name__ == "__main__":
   np.random.seed(30)
            
   red_object = RedClass() 
     
   #LOADING DATASET
   veg,bwt = red_object.loadDataSet()
   #print(veg.shape)

   #CONCAT_DATASETS
   X,y = red_object.concatDataSets(veg,bwt)
   plt.plot(X[0,:,0]/10000.0,"b") 

   X = red_object.filter_data(X,25)

   #print(X.shape)
   X_ndvi = X[:,:,-1]
   X = X[:,:,:-1]/10000.0

   plt.plot(X[0,:,0],"r")
   plt.show()

   idx = np.random.randint(2,size=X.shape[0])

   #print(idx)

   X_train, X_valid, y_train, y_valid = X[idx==0,:,:],X[idx==1,:,:],y[idx==0],y[idx==1]

   print(X_train.shape)
   #print(X_valid.shape)
   
   print(tf.__version__)
   
   #red_object.exp1(X_train,X_valid,y_train,y_valid) 
   #red_object.exp2(X_train,X_valid,y_train,y_valid) 
   red_object.exp4(X_train,X_valid,y_train,y_valid) 
   #X_new = red_object.process_data(X,np.array([0.7249732,0.16765448,0.6030529,0.41907814,0.5606056,-0.29389688,-0.7113738]),-0.10607813)
   #X_new = red_object.process_data2(X,np.array([0.92509115,-0.31727937,0.23246963,0.24132209,-0.17650443,0.08887789,-0.35802707]),0.04901218,np.array([-0.38472643,-0.5096011,0.22391927,0.11414693,-0.14565101,0.39224625,0.01539364]),0.03469806)
   #X_new = red_object.process_data2(X,np.array([0.92509115,-0.31727937,0.23246963,0.24132209,-0.17650443,0.08887789,-0.35802707]),0.04901218,np.array([-0.38472643,-0.5096011,0.22391927,0.11414693,-0.14565101,0.39224625,0.01539364]),0.03469806)
   #X_new = X_ndvi

   #X_veg = X_new[y==1,:]

   #X_set = X_new[y==0,:]

   #plt.plot(X_veg[0,:],"r")
   #plt.plot(X_set[0,:],"b")
   #plt.show()

   #v = np.mean(X_veg,axis=0)
   #s = np.mean(X_set,axis=0)

   #v_std = np.std(X_veg,axis=0)
   #s_std = np.std(X_set,axis=0)

   #plt.plot(v,"r")
   #plt.plot(v+v_std,"r:")
   #plt.plot(v-v_std,"r:")
   #plt.plot(s,"b")
   #plt.plot(s+s_std,"b:")
   #plt.plot(s-s_std,"b:")
   #plt.show() 

   #pca = PCA(n_components=X_new.shape[1],whiten=False)
   #X_T = pca.fit(X_new).transform(X_new)

   #for k in range(X_new.shape[0]):
   #    if k % 2 == 0:
   #       if y[k] == 0:
   #          plt.plot(X_T[k,0],X_T[k,1],"ro",alpha=0.4,ms=10.0)
   #       else:
   #          plt.plot(X_T[k,0],X_T[k,1],"bo",alpha=0.4,ms=10.0)

   #plt.show()



 
