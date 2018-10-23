import scipy.io
import pylab as plt
from sklearn.decomposition import PCA
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from matplotlib.patches import Ellipse
from termcolor import colored
from sklearn import mixture

M_LENGTH = 45

class MODIS():

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
        #print(Xv.shape)
        #print(Xb.shape)

        X_concat[:,:,c] = np.vstack((Xv,Xb))
        #print (X.shape)

      yv = np.ones((v_pixels,1),dtype=int)
      yb = np.zeros((b_pixels,1),dtype=int)
      y = np.vstack((yv,yb))[:,0]
 
      return X_concat,y

  def createDictionary(self,X,num_veg_pixels=592,num_set_pixels=333):

      EXTRA_TIME = 8

      #print(X.shape)

      #X_temp = np.reshape(X,(X.shape[1],X.shape[0],X.shape[2]))

      #print(X_temp.shape)

      X_45 = np.array([])

      start = 0
      stop = M_LENGTH  
   
      while stop < X.shape[1]:
            if len(X_45) == 0:
               X_45 = X[:,start:stop,:]
            else:
               X_45 = np.concatenate((X_45,X[:,start:stop,:]))
            
            start += M_LENGTH 
            stop += M_LENGTH 
            #print(stop)
            print(start)

      Y_model = {}

      start = X.shape[1]-EXTRA_TIME

      for i in range(M_LENGTH):
          temp_model = np.squeeze(X_45[:,i,:])
          if i <= EXTRA_TIME-1:
             temp_model = np.concatenate((temp_model,np.squeeze(X[:,start+i,:])))
          Y_model[i] = temp_model

      for key in Y_model.keys():
          print(key)
          print(Y_model[key].shape)
        
      
if __name__ == "__main__":
   m = MODIS()
   veg,bwt = m.loadDataSet(name="Gauteng_nochange.mat",province="Gauteng")
   X,y = m.concatDataSets(veg,bwt)
   #print(X.shape)
   #print(y.shape)
   
   m.createDictionary(X)
   
   
      
   
