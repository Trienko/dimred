import scipy.io
import pylab as plt
from sklearn.decomposition import PCA
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

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
      #(pixels,time,band) - X
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

  def FFTTransform(self,X):
      #(pixels,time,band) - X
      chan = X.shape[2]
      timeslots = X.shape[1]
      pixels = X.shape[0]
      
      N = timeslots
      T = 1.0 #samples_per_year/2.0 #PERIOD OF SINUSOIDAL WAVE IS 45 OBSERVATIONS, F = 1/45 (2.0 is for Nyquist). 
      X_T = np.zeros((pixels,timeslots,chan),dtype=complex)
      xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
      #plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
      #plt.grid()
      #plt.show()

      for p in range(pixels):
          print("p = "+str(p))
          for c in range(chan):
              t_series_test = X[p,:,c]
      	      X_T[p,:,c] = fft(t_series_test)

      return X_T, xf

  def model(self,t, A, f, phi):
      return A*np.sin(2*np.pi*f*t+phi)  

  def residuals(self,phi, A, f, y, t):
      return y - self.model(t, A, f, phi)

  def findPhi(self,X,Xf):
      #x, flag = leastsq(residuals, x0, args=(waveform_1, t))
      chan = X.shape[2]
      timeslots = X.shape[1]
      pixels = X.shape[0]
      f = 1.0/45.0
      t = np.arange(timeslots)
 
      Par = np.zeros((pixels,3,chan),dtype=float)

      for p in range(pixels):
          for c in range(chan):
              print("p = ",p)
              test = X[p,:,c]
              testf = np.absolute(Xf[p,:,c])*(2.0/timeslots)
              testf = np.sort(testf)
              Pvec = testf[-2:]
              A = Pvec[0]
              M = Pvec[1]/2.0
              test = test - M
              x0 = 0
              x, flag = leastsq(self.residuals, x0, args=(A,f,test,t))      
              #print("x = "+str(x))
              #model_values = self.model(t,A,f,x)
              #plt.plot(t,test)
              #plt.plot(t,model_values)
              #plt.show()
              Par[p,0,c] = M
              Par[p,1,c] = A
              Par[p,2,c] = x[0]

      return Par 

  def FourierInfo(self,Xf):
      chan = Xf.shape[2]
      Xf_new = np.absolute(Xf)
      Xf_new = np.sort(Xf_new,axis=1)
      Xf_new = Xf_new[:,::-1,:]

      res = np.cumsum(Xf_new,axis=1)

      res = np.average(res,axis=0)

      for c in range(chan):
          res[:,c] = res[:,c]/res[-1,c]
      return res 
         
  def FFTExample(self):
      # Number of sample points
      N = 600
      # sample spacing
      T = 1.0 / 800.0
      x = np.linspace(0.0, N*T, N)
      y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
      #plt.plot(x,y)
      #plt.show()
      yf = fft(y)
      print(N//2)
      xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
      
      plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
      plt.grid()
      plt.show()
      
       
if __name__ == "__main__":
      red_object = RedClass() 
     
      #LOADING DATASET
      veg,bwt = red_object.loadDataSet()
      #print(veg.shape)

      #CONCAT_DATASETS
      X,y = red_object.concatDataSets(veg,bwt)
      print(X.shape)

      '''
      XT,var_ratio = red_object.PCATransform(X)
      print(XT.shape)
      '''

      Xf,x_f = red_object.FFTTransform(X)

      Par = red_object.findPhi(X,Xf)

      ratio = red_object.FourierInfo(Xf)
      for k in range(ratio.shape[1]):
          plt.plot(np.cumsum(np.ones((ratio.shape[0],))),ratio[:,k])

      plt.show()
      

      #red_object.FFTExample()

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
