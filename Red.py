import scipy.io
import pylab as plt
from sklearn.decomposition import PCA
import numpy as np


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
        #print(Xv.shape)
        #print(Xb.shape)

        X_concat[:,:,c] = np.vstack((Xv,Xb))
        #print (X.shape)

      yv = np.ones((v_pixels,1),dtype=int)
      yb = np.zeros((b_pixels,1),dtype=int)
      y = np.vstack((yv,yb))[:,0]
 
      return X_concat,y
  
  def PCATransform(self,X):
      chan = X.shape[2]
      timeslots = X.shape[1]
      pixels = X.shape[0]
      
      X_T = np.zeros((pixels,timeslots,chan),dtype=float)
      var_ratio = np.zeros((timeslots,chan),dtype=float)

      for c in range(chan):
          print("Running PCA on channel: "+str(c))
          pca = PCA(n_components=timeslots,whiten=False)
          X_T[:,:,c] = pca.fit(X[:,:,c]).transform(X[:,:,c])
          var_ratio[:,c] = np.array(pca.explained_variance_ratio_)
           
      return X_T,var_ratio
      
       
if __name__ == "__main__":
      red_object = RedClass() 
      
      #LOADING DATASET
      veg,bwt = red_object.loadDataSet()
      #print(veg.shape)

      #CONCAT_DATASETS
      X,y = red_object.concatDataSets(veg,bwt)
      print(X.shape)

      XT,var_ratio = red_object.PCATransform(X)
      print(XT.shape)

      #for k in range(var_ratio.shape[1]):
      #    plt.plot(np.cumsum(np.ones((var_ratio.shape[0],))),np.cumsum(var_ratio[:,k]))

      #plt.show()


      '''
      #PLOT A SINGLE VEGETATION MODIS PIXEL (over all time and bands - except NDVI) 
      for k in range(7):
          plt.plot(veg_gauteng[:,0,k])
      plt.title("VEGETATION")
      plt.show()
          
      #PLOT A SINGLE SETTLEMENT MODIS PIXEL (over all time and bands - except NDVI) 
      for k in range(7):
        plt.plot(bwt_gauteng[:,0,k])

      plt.title("SETTLEMENT") 
      plt.show()
      '''
