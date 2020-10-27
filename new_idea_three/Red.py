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
import os
import random
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
      model.add(keras.layers.Dense(45,activation="relu",kernel_initializer="he_normal",kernel_regularizer=keras.regularizers.l2(1e-3)))
      model.add(keras.layers.Dense(1,activation="sigmoid"))
      model.summary() 

      #COMPILE
      model.compile(loss="binary_crossentropy",optimizer="sgd",metrics=["accuracy"])

      #FIT
      history = model.fit(X_train,y_train,epochs=30,batch_size=5,validation_data=(X_valid,y_valid))

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

  def exp5(self,X_train,X_valid,y_train,y_valid):

      idx = np.random.randint(2,size=X_valid.shape[0])
      X_test = X_valid[idx==0,:,:]
      y_test = y_valid[idx==0]

      X_valid = X_valid[idx==1,:,:]
      y_valid = y_valid[idx==1]

      print(X_test.shape)
      print(X_valid.shape)

      #Adding an extra dimension to use convolution layer
      X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
      X_valid = X_valid.reshape((X_valid.shape[0],X_valid.shape[1],X_valid.shape[2],1))
      X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))


      #BUILDING THE MODEL
      model = keras.models.Sequential()
      #model.add(keras.layers.experimental.preprocessing.Normalization(input_shape=[368,7,1]))
      #model.add(keras.layers.BatchNormalization(input_shape=[368,7,1]))
      model.add(keras.layers.Conv2D(1,(1,7),activation="tanh",input_shape=[368,7,1],padding="valid",kernel_initializer="glorot_normal"))
      model.add(tf.keras.layers.Reshape((368, 1), input_shape=(368,1,1)))
      #model.add(keras.layers.BatchNormalization())
      model.add(tf.keras.layers.Conv1D(1,45,activation="tanh",strides=45,padding="valid",kernel_initializer="glorot_normal"))
      model.add(keras.layers.Flatten())
      #model.add(keras.layers.BatchNormalization())
      #model.add(keras.layers.Dense(368,activation="relu"))
      model.add(keras.layers.Dense(8,activation="tanh",kernel_initializer="glorot_normal"))
      #model.add(keras.layers.BatchNormalization())
      model.add(keras.layers.Dense(1,activation="sigmoid"))
      model.summary() 
      #optimizers.SGD(lr=10e-4)
      #COMPILE
      model.compile(loss="binary_crossentropy",optimizer=keras.optimizers.Adam(lr=1e-2),metrics=["accuracy"])
          
      

      #FIT
      history = model.fit(X_train,y_train,epochs=30,batch_size=10,validation_data=(X_valid,y_valid))

      model.evaluate(X_test,y_test,batch_size=1)

      #w,b = model.layers[0].get_weights()
      #print(w[0,:,0,0])
      #print(b)

      '''
      Model: "sequential_99"
      _________________________________________________________________
      Layer (type)                 Output Shape              Param #   
      =================================================================
      flatten_99 (Flatten)         (None, 2576)              0         
      _________________________________________________________________
      dense_149 (Dense)            (None, 1)                 2577      
      =================================================================
      Total params: 2,577
      Trainable params: 2,577
      Non-trainable params: 0
      _________________________________________________________________
      '''

  def denseModel(self,X_train,X_valid,y_train,y_valid,lr=1e-2,r=1e-3,e=10,bs=10):

      #CREATE TEST SET
      ################# 
      idx = np.random.randint(2,size=X_valid.shape[0])
      X_test = X_valid[idx==0,:,:]
      y_test = y_valid[idx==0]

      X_valid = X_valid[idx==1,:,:]
      y_valid = y_valid[idx==1]
      #################

      #BUILDING THE MODEL
      model = keras.models.Sequential()
      model.add(keras.layers.Flatten(input_shape=[368,7]))
      #model.add(keras.layers.Dense(2,activation="relu",kernel_initializer="he_normal",kernel_regularizer=keras.regularizers.l2(r)))
      model.add(keras.layers.Dense(1,activation="sigmoid",kernel_regularizer=keras.regularizers.l2(r)))
      model.summary() 

      #COMPILE
      model.compile(loss="binary_crossentropy",optimizer=keras.optimizers.Adam(lr=lr),metrics=["accuracy"])

      #FIT
      history = model.fit(X_train,y_train,epochs=e,batch_size=bs,validation_data=(X_valid,y_valid))

      #EVALUATE
      model.evaluate(X_test,y_test,batch_size=1)
      y_pred = np.zeros(y_test.shape,dtype=int)
      y_pred[model.predict(X_test).flatten()>=0.5]=1
      cm=confusion_matrix(y_test,y_pred)
      cm[0,:] = cm[0,:]/sum(cm[0,:])*100 #settlement
      cm[1,:] = cm[1,:]/sum(cm[1,:])*100 #vegetation
      cm = cm.astype(float)
      print(cm)

      return cm 
      '''
      _________________________________________________________________
      Layer (type)                 Output Shape              Param #   
      =================================================================
      conv2d_49 (Conv2D)           (None, 368, 1, 1)         8         
      _________________________________________________________________
      reshape_49 (Reshape)         (None, 368, 1)            0         
      _________________________________________________________________
      conv1d_49 (Conv1D)           (None, 8, 1)              46        
      _________________________________________________________________
      flatten_98 (Flatten)         (None, 8)                 0          
      _________________________________________________________________
      dense_147 (Dense)            (None, 8)                 72        
      _________________________________________________________________
      dense_148 (Dense)            (None, 1)                 9         
      =================================================================
      Total params: 135
      Trainable params: 135
      Non-trainable params: 0
      _________________________________________________________________

      '''  
  def parsimonuous_TCN(self,X_train,X_valid,y_train,y_valid,lr=1e-2,r=1e-3,e=10,bs=10):
      
      #CREATE TEST SET
      ################# 
      idx = np.random.randint(2,size=X_valid.shape[0])
      X_test = X_valid[idx==0,:,:]
      y_test = y_valid[idx==0]

      X_valid = X_valid[idx==1,:,:]
      y_valid = y_valid[idx==1]
      #################

      #Adding an extra dimension to use convolution layer
      X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
      X_valid = X_valid.reshape((X_valid.shape[0],X_valid.shape[1],X_valid.shape[2],1))
      X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))


      #BUILDING THE MODEL USING THE SEQ API
      #################
      model = keras.models.Sequential()
      model.add(keras.layers.Conv2D(1,(1,7),activation="tanh",input_shape=[368,7,1],padding="valid",kernel_initializer="glorot_normal",kernel_regularizer=keras.regularizers.l2(r)))
      model.add(tf.keras.layers.Reshape((368, 1), input_shape=(368,1,1)))
      model.add(tf.keras.layers.Conv1D(1,45,activation="tanh",strides=45,padding="valid",kernel_initializer="glorot_normal",kernel_regularizer=keras.regularizers.l2(r)))
      model.add(keras.layers.Flatten())
      model.add(keras.layers.Dense(8,activation="tanh",kernel_initializer="glorot_normal",kernel_regularizer=keras.regularizers.l2(r)))
      model.add(keras.layers.Dense(1,activation="sigmoid"))
      model.summary() 
      #################    
      

      #COMPILE AND FIT
      model.compile(loss="binary_crossentropy",optimizer=keras.optimizers.Adam(lr=lr),metrics=["accuracy"])
      history = model.fit(X_train,y_train,epochs=e,batch_size=bs,validation_data=(X_valid,y_valid))

      #EVALUATE
      model.evaluate(X_test,y_test,batch_size=1)
      y_pred = np.zeros(y_test.shape,dtype=int)
      y_pred[model.predict(X_test).flatten()>=0.5]=1
      cm=confusion_matrix(y_test,y_pred)
      cm = cm.astype(float)
      
      #print(cm.shape)
      cm[0,:] = cm[0,:]/sum(cm[0,:])*100 #settlement
      cm[1,:] = cm[1,:]/sum(cm[1,:])*100 #vegetation

      print(cm) 

      #SAVING THE WEIGHTS
      #idx = np.zeros((11,45),dtype=int)
      #idx[0,:7] = 1
      #idx[1,:] = 1
      #idx[2:10,:8] = 1
      #idx[10,:8] = 1

      weights = np.zeros((11,45),dtype=float)
      w,b = model.layers[0].get_weights()
      weights[0,:7] = (w[0,:,0,0]-np.amin(w[0,:,0,0]))/(np.amax(w[0,:,0,0])-np.amin(w[0,:,0,0]))
      w,b = model.layers[2].get_weights()
      weights[1,:] = (w[:,0,0]-np.amin(w[:,0,0]))/(np.amax(w[:,0,0])-np.amin(w[:,0,0])) 
      w,b = model.layers[4].get_weights()
      weights[2:10,:8] = (w-np.amin(w))/(np.amax(w)-np.amin(w)) 
      #print(w.shape) 
      w,b = model.layers[5].get_weights()
      weights[10,:8] = (w[:,0]-np.amin(w[:,0]))/(np.amax(w[:,0])-np.amin(w[:,0])) 
      
      return cm,weights
            

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


def set_seeds(seed=10):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed=10, fast_n_close=False):
    """
        Enable 100% reproducibility on operations related to tensor and randomness.
        Parameters:
        seed (int): seed value for global randomness
        fast_n_close (bool): whether to achieve efficient at the cost of determinism/reproducibility
    """
    set_seeds(seed=seed)
    if fast_n_close:
        return

    #logging.warning("*******************************************************************************")
    #logging.warning("*** set_global_determinism is called,setting full determinism, will be slow ***")
    #logging.warning("*******************************************************************************")

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    # https://www.tensorflow.org/api_docs/python/tf/config/threading/set_inter_op_parallelism_threads
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    #from tfdeterminism import patch
    #patch()
 
      

   
if __name__ == "__main__":
   
   seed = 10
   ex = -1

   if ex != -1:

      set_global_determinism(seed)         
      red_object = RedClass() 
      filename = 'results'+str(seed)+".pkl"
      outfile = open(filename,'wb')
     
      #LOADING DATASET
      veg,bwt = red_object.loadDataSet()
      #print(veg.shape)

      #CONCAT_DATASETS
      X,y = red_object.concatDataSets(veg,bwt)
      plt.plot(X[0,:,0]/10000.0,"b") 

      #REMOVE NDVI AND RESCALE
      X_ndvi = X[:,:,-1]
      X = X[:,:,:-1]/10000.0

      tcn_weights = np.zeros((ex,11,45),dtype=float)
      tcn_cm = np.zeros((ex,2,2),dtype=int)
      dense_cm = np.zeros((ex,2,2),dtype=int)

      for e in range(ex):
  
         #CONSTRUCT TRAINING/VALIDATION SET   
         idx = np.random.randint(2,size=X.shape[0])
         X_train, X_valid, y_train, y_valid = X[idx==0,:,:],X[idx==1,:,:],y[idx==0],y[idx==1]

         #TCN EXPERIMENT
         print("TCN: "+str(e))
         tcn_cm[e,:,:],tcn_weights[e,:,:] = red_object.parsimonuous_TCN(X_train,X_valid,y_train,y_valid)
         print("DENSE: "+str(e))
         dense_cm[e,:,:] = red_object.denseModel(X_train,X_valid,y_train,y_valid)  

      pickle.dump(tcn_cm,outfile)
      pickle.dump(tcn_weights,outfile)
      pickle.dump(dense_cm,outfile)
      outfile.close()
   else:
      red_object = RedClass() 
      import matplotlib
      matplotlib.rcParams.update({'font.size': 30})  
      #LOADING DATASET
      veg,bwt = red_object.loadDataSet()
      #print(veg.shape)

      #CONCAT_DATASETS
      X,y = red_object.concatDataSets(veg,bwt)
      X = X[:,:,:-1]/10000.0

      fig, ax = plt.subplots(nrows=1, ncols=1)
      ax.plot(X[0,:,:],lw=3)
     
      fig.patches.extend([plt.Rectangle((0.16,0.11),0.11,0.75,
                                  fill=True, color='g', alpha=0.5, zorder=1000,
                                   transform=fig.transFigure,figure=fig)])
      fig.patches.extend([plt.Rectangle((0.65,0.11),0.01,0.75,
                                  fill=True, color='r', alpha=0.5, zorder=1000,
                                   transform=fig.transFigure,figure=fig)])
      ax.text(70,0.13, 'Temporal convolution (45)', style='italic', bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 10})
      ax.text(270,0.13, 'Spectral convolution (7)', style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
      ax.axes.xaxis.set_visible(False)
      ax.axes.yaxis.set_visible(False)

      plt.show()
      
      filename = 'results'+str(seed)+".pkl"
      infile = open(filename,'rb')
      tcn_cm = pickle.load(infile) 
      tcn_weights = pickle.load(infile) 
      dense_cm = pickle.load(infile)  

      print(np.mean(tcn_cm,axis=0))
      print(np.std(tcn_cm,axis=0))

      print(np.mean(dense_cm,axis=0))
      print(np.std(dense_cm,axis=0))

      plt.close(fig)
      matplotlib.rcParams.update({'font.size': 17}) 
      fig = plt.figure()
      ax = fig.add_subplot(1, 1, 1)
      
      #plt.clf()
      #plt.cla()
      
      #ax.plot(X[0,:,:],lw=2.2)
      im = ax.imshow(tcn_weights[0,:,:])
      plt.colorbar(im,orientation='horizontal')

      #ax = plt.gca()
      ax.axes.xaxis.set_visible(False)
      ax.axes.yaxis.set_visible(False)
      ax.text(-7.5,0.1, 'Spectral weights', style='italic', bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 5})
      ax.text(-7.5,1.1, 'Temporal weights', style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 5})
      ax.text(-7.5,6.1,'Dense weights', style='italic', bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 5})
      #ax.text(-4,6.1,'{', size=200,alpha=0.5)
      ax.annotate("", xy=(0, 1.5), xytext=(0, 9.5), arrowprops=dict(arrowstyle="<->"))
      ax.text(-7.5,10.1, 'Sigmoid weights', style='italic', bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 5})

      plt.show()
      infile.close()

   #print(X_train.shape)
   #print(X_valid.shape)
   
   #print(tf.__version__)
   #for k in range(1):
   #    print(k)
       #red_object.exp1(X_train,X_valid,y_train,y_valid) 
   #red_object.exp2(X_train,X_valid,y_train,y_valid) 
   #red_object.exp4(X_train,X_valid,y_train,y_valid)
   #    red_object.parsimonuous_TCN(X_train,X_valid,y_train,y_valid) 
   #    red_object.exp2(X_train,X_valid,y_train,y_valid) 
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



 
