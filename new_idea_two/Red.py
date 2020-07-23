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
      
import math

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
        #print(Xv.shape)
        #print(Xb.shape)

        X_concat[:,:,c] = np.vstack((Xv,Xb))
        #print (X.shape)

      yv = np.ones((v_pixels,1),dtype=int)
      yb = np.zeros((b_pixels,1),dtype=int)
      y = np.vstack((yv,yb))[:,0]
 
      return X_concat,y

  def create_change_data(self,veg,bwt,N):

      data = np.zeros((N,veg.shape[1]),dtype = float)

      alpha = np.linspace(0,1,22)
            
      for k in range(N):
	  veg_pixel = np.random.randint(veg.shape[0])          
          bwt_pixel = np.random.randint(bwt.shape[0]) 
          cp = np.random.randint(299)+1

          temp_data = (1-alpha)*veg[veg_pixel,cp:cp+22]+alpha*bwt[bwt_pixel,cp:cp+22]
          data[k,:] = np.concatenate([veg[veg_pixel,:cp],temp_data,bwt[bwt_pixel,cp+22:]])
      
      return data
          
   
  def create_train_test(self,veg,bwt):
      #print(veg.shape)

      arr = np.arange(veg.shape[1])
      np.random.shuffle(arr)
      #print(arr)
      veg = veg[:,arr,:]
 
      arr = np.arange(bwt.shape[1])
      np.random.shuffle(arr)
      #print(arr)
      bwt = bwt[:,arr,:]
 

      veg_temp = veg[:,:400,7].T
      #print(veg_temp.shape)

      veg_train = veg_temp[:200,:]
      veg_test = veg_temp[200:,:]
      #print(veg_train.shape)
      #print(veg_test.shape)


      veg_temp = veg[:,400:,7].T

      veg_c_train = veg_temp[:96,:]
      veg_c_test = veg_temp[96:,:]
      #print(veg_c_train.shape)
      #print(veg_c_test.shape)
      
      bwt_temp = bwt[:,:,7].T
 
      bwt_c_train = bwt_temp[:166,:]
      bwt_c_test = bwt_temp[166:,:]
      #print(bwt_c_train.shape)
      #print(bwt_c_test.shape)

      c_train = self.create_change_data(veg_c_train,bwt_c_train,200)
      #print(c_train.shape)
      #plt.plot(c_train[199,:])
      #plt.show()
      c_test = self.create_change_data(veg_c_test,bwt_c_test,200)
      
      return veg_train,veg_test,c_train,c_test

  def compute_z_score(self,veg_train,veg_test,c_train,c_test,c=1.2):

      z_train = np.zeros((veg_train.shape[0]+c_train.shape[0],),dtype = float)
      z_test = np.zeros((veg_test.shape[0]+c_test.shape[0],),dtype = float)
      y_train = np.zeros(z_train.shape,dtype = int)
      y_test = np.zeros(z_test.shape,dtype = int)


      d_train = np.zeros((veg_train.shape[0]+c_train.shape[0],7),dtype = float)
      d_test = np.zeros((veg_test.shape[0]+c_test.shape[0],7),dtype = float)

      for k in range(len(z_train)):
          if (k < veg_train.shape[0]):
             #Step 1: Lunetta --- filter time-series
             t = veg_train[k,:]
             f = np.fft.fft(t)
             f[2:] = 0
             #f[10:]=0
             t = np.real(np.fft.ifft(f))

             #Step 2: Sum years:
             y8 = np.zeros((8,),dtype=float)
             y7 = np.zeros((7,),dtype=float)

             for i in range(8):
                 y8[i] = np.sum(t[45*i:45*i+45])
             
             for i in range(1,8):
                 y7[i-1] = y8[i]-y8[i-1]

             d_train[k,:] = y7
          else:
             y_train[k] = 1
             #Step 1: Lunetta --- filter time-series
             t = c_train[k-veg_train.shape[0],:]
             f = np.fft.fft(t)
             f[2:] = 0
             #f[10:]=0
             t = np.real(np.fft.ifft(f))

             #Step 2: Sum years:
             y8 = np.zeros((8,),dtype=float)
             y7 = np.zeros((7,),dtype=float)

             for i in range(8):
                 y8[i] = np.sum(t[45*i:45*i+45])
             
             for i in range(1,8):
                 y7[i-1] = y8[i]-y8[i-1]

             d_train[k,:] = y7

      for k in range(len(z_test)):
          if (k < veg_test.shape[0]):
             #Step 1: Lunetta --- filter time-series
             t = veg_test[k,:]
             f = np.fft.fft(t)
             f[2:] = 0
             #f[10:]=0
             t = np.real(np.fft.ifft(f))

             #Step 2: Sum years:
             y8 = np.zeros((8,),dtype=float)
             y7 = np.zeros((7,),dtype=float)

             for i in range(8):
                 y8[i] = np.sum(t[45*i:45*i+45])
             
             for i in range(1,8):
                 y7[i-1] = y8[i]-y8[i-1]

             d_test[k,:] = y7
          else:
             y_test[k] = 1
             #Step 1: Lunetta --- filter time-series
             t = c_test[k-veg_test.shape[0],:]
             f = np.fft.fft(t)
             f[2:] = 0
             #f[10:]=0
             t = np.real(np.fft.ifft(f))

             #Step 2: Sum years:
             y8 = np.zeros((8,),dtype=float)
             y7 = np.zeros((7,),dtype=float)

             for i in range(8):
                 y8[i] = np.sum(t[45*i:45*i+45])
             
             for i in range(1,8):
                 y7[i-1] = y8[i]-y8[i-1]

             d_test[k,:] = y7

      #print(d_test)
      

      m_train = np.mean(d_train,axis=0)
      m_test = np.mean(d_test,axis=0)

      s_train = np.std(d_train,axis=0)
      s_test = np.std(d_test,axis=0)

      #print(m_test)
      #print(s_test)

      for k in range(len(z_train)):
          z_train[k] = np.max(np.absolute((d_train[k,:]-m_train)/s_train))

      for k in range(len(z_test)):
          z_test[k] = np.max(np.absolute((d_test[k,:]-m_test)/s_test))

      #plt.plot(z_train)
      #plt.plot(z_test)

      #plt.show()


      y_predict1 = np.zeros(y_train.shape,dtype = int)
      y_predict1[z_train>c] = 1

      y_predict2 = np.zeros(y_test.shape,dtype = int)
      y_predict2[z_test>c] = 1
      

      return confusion_matrix(y_train,y_predict1),confusion_matrix(y_test,y_predict2)     

      
  def do_GAF_change_experiment(self,veg,bwt):
      print colored('FIRST METHOD: Change GAF','cyan')
      cm_GAF = np.zeros((2,2),dtype=float)
      
      veg_train, veg_test, c_train, c_test = self.create_train_test(veg,bwt)  
      c_test = red_object.load_NDVI_change()
      GAF_matrix_train = np.zeros((veg_train.shape[0]+c_train.shape[0],veg_train.shape[1]*veg_train.shape[1]),dtype=float)
      y_train = np.zeros((veg_train.shape[0]+c_train.shape[0],),dtype=int)

      GAF_matrix_test = np.zeros((veg_test.shape[0]+c_test.shape[0],veg_test.shape[1]*veg_test.shape[1]),dtype=float)
      y_test = np.zeros((veg_test.shape[0]+c_test.shape[0],),dtype=int)

      for k in range(GAF_matrix_train.shape[0]):
          if k < veg_train.shape[0]:
             GAF_matrix_train[k,:] = self.transform(veg_train[k,:],no_scaling=True)[0].flatten()
          else:
             GAF_matrix_train[k,:] = self.transform(c_train[k-veg_train.shape[0],:],no_scaling=True)[0].flatten()
             #y_train[k-veg_train.shape[0]] = 1
             y_train[k] = 1 

      for k in range(GAF_matrix_test.shape[0]):
          if k < veg_test.shape[0]:
             GAF_matrix_test[k,:] = self.transform(veg_test[k,:],no_scaling=True)[0].flatten()
          else:
             GAF_matrix_test[k,:] = self.transform(c_test[k-veg_test.shape[0],:],no_scaling=True)[0].flatten()
             y_test[k] = 1

      clf = LogisticRegression(random_state=0).fit(GAF_matrix_train, y_train)
      y_pred = clf.predict(GAF_matrix_test)
      cm_GAF[:,:] = confusion_matrix(y_test,y_pred)
      print colored('CM'+str(cm_GAF[:,:]),'blue')

      return cm_GAF, GAF_matrix_train, GAF_matrix_test

  def do_time_series_change_experiment(self,veg,bwt):
      print colored('SECOND METHOD: Change time-series','red')
      cm_GAF = np.zeros((2,2),dtype=float)
      
      veg_train, veg_test, c_train, c_test = self.create_train_test(veg,bwt)  
      c_test = red_object.load_NDVI_change()
      GAF_matrix_train = np.zeros((veg_train.shape[0]+c_train.shape[0],veg_train.shape[1]),dtype=float)
      y_train = np.zeros((veg_train.shape[0]+c_train.shape[0],),dtype=int)

      GAF_matrix_test = np.zeros((veg_test.shape[0]+c_test.shape[0],veg_test.shape[1]),dtype=float)
      y_test = np.zeros((veg_test.shape[0]+c_test.shape[0],),dtype=int)

      for k in range(GAF_matrix_train.shape[0]):
          if k < veg_train.shape[0]:
             GAF_matrix_train[k,:] = veg_train[k,:]#self.transform(veg_train[k,:],no_scaling=True)[0].flatten()
          else:
             GAF_matrix_train[k,:] = c_train[k-veg_train.shape[0],:]#self.transform(c_train[k-veg_train.shape[0],:],no_scaling=True)[0].flatten()
             #y_train[k-veg_train.shape[0]] = 1
             y_train[k] = 1 

      for k in range(GAF_matrix_test.shape[0]):
          if k < veg_test.shape[0]:
             GAF_matrix_test[k,:] = veg_test[k,:]#self.transform(veg_test[k,:],no_scaling=True)[0].flatten()
          else:
             GAF_matrix_test[k,:] = c_test[k-veg_test.shape[0],:] #self.transform(c_test[k-veg_test.shape[0],:],no_scaling=True)[0].flatten()
             y_test[k] = 1

      clf = LogisticRegression(random_state=0).fit(GAF_matrix_train, y_train)
      y_pred = clf.predict(GAF_matrix_test)
      cm_GAF[:,:] = confusion_matrix(y_test,y_pred)
      print colored('CM'+str(cm_GAF[:,:]),'blue')

      return cm_GAF



  def tabulate(self,x, y, f):
      """Return a table of f(x, y). Useful for the Gram-like operations."""
      return np.vectorize(f)(*np.meshgrid(x, y, sparse=True))

  def cos_sum(self,a, b):
      """To work with tabulate."""
      return(math.cos(a+b))

  def transform(self, serie,min_=0,max_=10000,no_scaling=True):
      """Compute the Gramian Angular Field of an image"""
      # Min-Max scaling
      #min_ = np.amin(serie)
      #max_ = np.amax(serie)

      if not no_scaling:
         scaled_serie = (2*serie - max_ - min_)/(max_ - min_)
      else:
         scaled_serie = serie+0

      # Floating point inaccuracy!
      scaled_serie = np.where(scaled_serie >= 1., 1., scaled_serie)
      scaled_serie = np.where(scaled_serie <= -1., -1., scaled_serie)

      # Polar encoding
      phi = np.arccos(scaled_serie)
      # Note! The computation of r is not necessary
      r = np.linspace(0, 1, len(scaled_serie))

      # GAF Computation (every term of the matrix)
      gaf = self.tabulate(phi, phi, self.cos_sum)

      return(gaf, phi, r, scaled_serie)


  
  def PCATransform(self,X):
      #(pixels,time,band) - X
      chan = X.shape[2]
      timeslots = X.shape[1]
      pixels = X.shape[0]
      
      X_T = np.zeros((pixels,timeslots,chan),dtype=float)
      var_ratio = np.zeros((timeslots,chan),dtype=float)

      for c in range(chan):
          #print("Running PCA on channel: "+str(c))
          print colored("Running PCA on channel: "+str(c),'red')
          pca = PCA(n_components=timeslots,whiten=False)
          X_T[:,:,c] = pca.fit(X[:,:,c]).transform(X[:,:,c])
          var_ratio[:,c] = np.array(pca.explained_variance_ratio_)
           
      return X_T,var_ratio

  def FFTTransform(self,X):
      #print("Performing FFT...")
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

      for c in range(chan):
          #print("p = "+str(p))
          #print("Performing FFT on channel: "+str(c))
          print colored("Performing FFT on channel: "+str(c),'blue') 
          for p in range(pixels):
              t_series_test = X[p,:,c]
      	      X_T[p,:,c] = fft(t_series_test)

      return X_T, xf

  def model(self,t, A, f, phi):
      return A*np.sin(2*np.pi*f*t+phi)  

  def residuals(self,phi, A, f, y, t):
      return y - self.model(t, A, f, phi)

  def findDomFFTFeatures(self,X,Xf):
      print("Making FFT feature Cube...")
      #x, flag = leastsq(residuals, x0, args=(waveform_1, t))
      chan = X.shape[2]
      timeslots = X.shape[1]
      pixels = X.shape[0]
      f = 1.0/45.0
      t = np.arange(timeslots)
 
      Par = np.zeros((pixels,3,chan),dtype=float)

      for p in range(pixels):
          for c in range(chan):
              #print("p = ",p)
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
      #print(N//2)
      xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
      
      plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
      plt.grid()
      plt.show()

  def constructDensitiesAndPlot(self,features,boundary = 592,fourier=False):
      print("Constructing densities...")
      chan = features.shape[2]
      pixels = features.shape[0]

      meanFinal1 = np.zeros((2,chan),dtype=float)
      covFinal1 = np.zeros((2,2,chan),dtype=float)

      meanFinal2 = np.zeros((2,chan),dtype=float)
      covFinal2 = np.zeros((2,2,chan),dtype=float)
      
      for c in range(chan):

          X_veg = features[:boundary,:2,c]
          X_bwt = features[boundary:,:2,c]
     
          gmm1 = mixture.GaussianMixture(n_components=1, covariance_type='full').fit(X_veg)
          gmm2 = mixture.GaussianMixture(n_components=1, covariance_type='full').fit(X_bwt)

          self.plot_results(X_veg,X_bwt,gmm1.means_[0],gmm2.means_[0],gmm1.covariances_[0],gmm2.covariances_[0],c1="red",c2="blue",fourier=fourier,chan=c) 
          meanFinal1[:,c] = gmm1.means_[0]
          meanFinal2[:,c] = gmm2.means_[0]
          covFinal1[:,:,c] = gmm1.covariances_[0]
          covFinal2[:,:,c] = gmm2.covariances_[0]
      return meanFinal1,meanFinal2,covFinal1,covFinal2 
          
  def HD(self,mean1,Sigma1,mean2,Sigma2):
      #print(mean1)
      mean1 = mean1.reshape((2,1))
      mean2 = mean2.reshape((2,1))
      #print(mean1)
      #print(mean1.shape)
      #print(Sigma1.shape)
      
      M = (Sigma1+Sigma2)/2.0
      u = mean1-mean2
      
      Minv = np.linalg.inv(M)
      x = np.dot(np.dot(u.T,Minv),u)[0,0]*(-1)*(1.0/8.0)

      detM = np.linalg.det(M)
      det1 = np.linalg.det(Sigma1)
      det2 = np.linalg.det(Sigma2)

      num = (det1**(1.0/4.0))*(det2**(1.0/4.0))
      den = detM**(1.0/2.0)
      y = num/den

      H = 1-y*np.exp(x)
      return np.sqrt(H)

      
  def eigsorted(self,cov):
      vals, vecs = np.linalg.eigh(cov)
      order = vals.argsort()[::-1]
      return vals[order], vecs[:,order]
        
  def plot_results(self,X_veg,X_bwt,mean1,mean2,cov1,cov2,c1="red",c2="blue",fourier=False,chan=0):
      import matplotlib 
      matplotlib.rcParams.update({'font.size': 20})
      plt.clf()
      ax = plt.gca()
      nstd = 2.0
      
      ##ELLIPSE 1
      vals, vecs = self.eigsorted(cov1[:,:])
      theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
      w, h = 2 * nstd * np.sqrt(vals)
      ell1 = Ellipse(xy=(mean1[0], mean1[1]),
              width=w, height=h,
              angle=theta, edgecolor=c1,facecolor='white',fill=True,linewidth=3,zorder=1)
      ell1.set_facecolor('none')
      ax.add_artist(ell1)

      ##ELLIPSE 2
      vals, vecs = self.eigsorted(cov2[:,:])
      theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
      w, h = 2 * nstd * np.sqrt(vals)
      ell2 = Ellipse(xy=(mean2[0], mean2[1]),
              width=w, height=h,
              angle=theta, edgecolor=c2,facecolor='white',fill=True,linewidth=3,zorder=1)
      ell2.set_facecolor('none')
      ax.add_artist(ell2)

      for i in range(X_veg.shape[0]):
         #col = c1[0]
         ax.scatter(X_veg[i,0],X_veg[i,1],c=c1,zorder=3,alpha=0.2)

      for i in range(X_bwt.shape[0]):
         #col = c2[0]
         ax.scatter(X_bwt[i,0],X_bwt[i,1],c=c2,zorder=3,alpha=0.2)
      
      

      if fourier:
         ax.set_xlabel("$f_1$")
         ax.set_ylabel("$f_1$")
         if chan <> 7:
            #ax.set_title("FFT: Band "+str(chan+1))
            plt.tight_layout()
            plt.savefig("FFT_Band_"+str(chan+1)+".pdf",bbox_inches='tight')
         else:
            #ax.set_title("FFT: NDVI")
            plt.tight_layout()
            plt.savefig("FFT_NDVI.pdf",bbox_inches='tight')
      else:
         ax.set_xlabel("PC 1")
         ax.set_ylabel("PC 2")
         if chan <> 7:
            #ax.set_title("PCA: Band "+str(chan+1))
            plt.tight_layout()
            plt.savefig("PCA_Band_"+str(chan+1)+".pdf",bbox_inches='tight')
         else:
            #ax.set_title("PCA: NDVI")
            plt.tight_layout()
            plt.savefig("PCA_NDVI.pdf",bbox_inches='tight')

  '''
  def plot_results(X, Y_, means, covariances, index, title):
      splot = plt.subplot(2, 1, 1 + index)
      for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
          v, w = linalg.eigh(covar)
          v = 2. * np.sqrt(2.) * np.sqrt(v)
          u = w[0] / linalg.norm(w[0])
          # as the DP will not use every component it has access to
          # unless it needs it, we shouldn't plot the redundant
          # components.
          if not np.any(Y_ == i):
             continue
          plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

          # Plot an ellipse to show the Gaussian component
          angle = np.arctan(u[1] / u[0])
          angle = 180. * angle / np.pi  # convert to degrees
          ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
          ell.set_clip_box(splot.bbox)
          ell.set_alpha(0.5)
          splot.add_artist(ell)
   gmm = mixture.GaussianMixture(n_components=5, covariance_type='full').fit(X)
   plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,'Gaussian Mixture')
   '''   
  def plotBar(self,x1,x2):
       plt.clf()
       N = len(x1)
       ind = np.arange(N)    # the x locations for the groups
       width = 0.35       # the width of the bars: can also be len(x) sequence

       p1 = plt.bar(ind, x1, width)
       p2 = plt.bar(ind, x2, width, bottom=x1)
       p3 = plt.bar(ind, x2-x1, width, bottom=x2+x1)

       plt.ylabel('HD (Cumulative)')
       plt.title('Hellinger Distance Plot')
       plt.xticks(ind, ('1', '2', '3', '4', '5', '6', '7', 'NDVI'))
       plt.xlabel("MODIS Bands")
       #plt.yticks(np.arange(0, 81, 10))
       plt.legend((p1[0], p2[0],p3[0]), ('FFT', 'PCA', 'PCA-FFT'))
       plt.savefig('HD.pdf')
       plt.show() 

  def do_PCA_experiment(self,X,y):
      print colored('FIRST METHOD: PCA','red')
      cm_PCA = np.zeros((X.shape[2],2,2),dtype=float)
      for b in range(7,8):
          print colored('Band '+str(b),'red')
          pca = PCA(n_components=X.shape[1],whiten=False)
          X_PCA = pca.fit(X[:,:,b]).transform(X[:,:,b])
          X_PCA = X_PCA[:,:2]
          
          idx = np.random.randint(2, size=X.shape[0])
          X_PCA_train = X_PCA[idx==1,:]
          y_train = y[idx==1]

          X_PCA_test = X_PCA[idx==0,:]
          y_test = y[idx==0]

          clf = LogisticRegression(random_state=0).fit(X_PCA_train, y_train)
          y_pred = clf.predict(X_PCA_test)

          cm_PCA[b,:,:] = confusion_matrix(y_test,y_pred)
      print colored('CM'+str(cm_PCA[7,:,:]),'red')
      return cm_PCA[7,:,:]

  def do_time_experiment(self,X,y):
      print colored('Third METHOD: Time','green')
      cm_T = np.zeros((X.shape[2],2,2),dtype=float)
      for b in range(X.shape[2]):
          print colored('Band '+str(b),'green')
                    
          idx = np.random.randint(2, size=X.shape[0])
          X_train = X[idx==1,:,b]
          y_train = y[idx==1]

          X_test = X[idx==0,:,b]
          y_test = y[idx==0]

          clf = LogisticRegression(random_state=0).fit(X_train, y_train)
          y_pred = clf.predict(X_test)

          cm_T[b,:,:] = confusion_matrix(y_test,y_pred)
          print colored('CM'+str(cm_T[b,:,:]),'green')
      return cm_T


  def do_GAF_experiment(self,X,y):
      print colored('SECOND METHOD: GAF','blue')
      cm_GAF = np.zeros((X.shape[2],2,2),dtype=float)
      for b in range(7,8):
          print colored('Band '+str(b),'blue')
	  GAF_matrix = np.zeros((X.shape[0],X.shape[1]*X.shape[1]),dtype=float)
          #GAF loop
          for k in range(X.shape[0]):
          #print(k)
              if (b < 7):
                 GAF_matrix[k,:] = self.transform(X[k,:,b],no_scaling=False)[0].flatten()
              else:
                 GAF_matrix[k,:] = self.transform(X[k,:,b],no_scaling=True)[0].flatten()

          idx = np.random.randint(2, size=X.shape[0])
          X_GAF_train = GAF_matrix[idx==1,:]
          y_train = y[idx==1]

          X_GAF_test = GAF_matrix[idx==0,:]
          y_test = y[idx==0]

          clf = LogisticRegression(random_state=0).fit(X_GAF_train, y_train)
          y_pred = clf.predict(X_GAF_test)

          cm_GAF[b,:,:] = confusion_matrix(y_test,y_pred)
      print colored('CM'+str(cm_GAF[7,:,:]),'blue')
      return cm_GAF[7,:,:],GAF_matrix

  def plot_confusion_matrix(self,cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
      """
      This function prints and plots the confusion matrix.
      Normalization can be applied by setting `normalize=True`.
      """
      if normalize:
          cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
          print("Normalized confusion matrix")
      else:
         print('Confusion matrix, without normalization')

      print(cm)

      plt.imshow(cm, interpolation='nearest', cmap=cmap)
      plt.title(title)
      plt.colorbar()
      tick_marks = np.arange(len(classes))
      plt.xticks(tick_marks, classes, rotation=45)
      plt.yticks(tick_marks, classes)

      fmt = '.2f' if normalize else 'd'
      thresh = cm.max() / 2.
      for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
         plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

      plt.tight_layout()
      plt.ylabel('True label')
      plt.xlabel('Predicted label')

  def load_NDVI_change(self,name="Gauteng.mat"):
      mat = scipy.io.loadmat("Gauteng.mat")

      band1 = mat["change_band0_valid"]
      band2 = mat["change_band1_valid"]

      NDVI = (band2-band1)/(band1+band2)

      return NDVI[:,0:368]
        
       
if __name__ == "__main__":

      


      filename = 'results.pkl'
      outfile = open(filename,'wb')
      np.random.seed(30)
            
      #IMPORTANT EXPERIMENT FOR PAPER
      #CLASSIFICATION EXPERIMENTS

      #cf2[0,:] = cf2[0,:]/sum(cf2[0,:])*100



      red_object = RedClass() 
     
      #LOADING DATASET
      veg,bwt = red_object.loadDataSet()
      #print(veg.shape)

      #CONCAT_DATASETS
      X,y = red_object.concatDataSets(veg,bwt)
      #print(X.shape)
      print "Classification experiments"
      #PCA
      cm_PCA_10 = np.zeros((2,2,10),dtype=float)

      for k in range(10):
          print(k) 
          cm_PCA = red_object.do_PCA_experiment(X,y)
          cm_PCA[0,:] = cm_PCA[0,:]/sum(cm_PCA[0,:])*100 #settlement
          cm_PCA[1,:] = cm_PCA[1,:]/sum(cm_PCA[1,:])*100 #vegetation
          cm_PCA_10[:,:,k] = cm_PCA
      
      pickle.dump(cm_PCA_10,outfile)

      print(np.mean(cm_PCA_10,axis=2))
      print(np.std(cm_PCA_10,axis=2))

      #cm_time =  red_object.do_time_experiment(X,y)
      #GAF
      cm_GAF_10 = np.zeros((2,2,10),dtype=float)

      for k in range(10):
          print(k) 
          cm_GAF,m1 = red_object.do_GAF_experiment(X,y)
          #print(cm_GAF)
          cm_GAF[0,:] = cm_GAF[0,:]/sum(cm_GAF[0,:])*100 #settlement
          cm_GAF[1,:] = cm_GAF[1,:]/sum(cm_GAF[1,:])*100 #vegetation
          cm_GAF_10[:,:,k] = cm_GAF

      pickle.dump(cm_GAF_10,outfile)
      pickle.dump(m1[0,:].reshape((368,368)),outfile)#vegetation
      pickle.dump(m1[-1,:].reshape((368,368)),outfile)#settlement

      #print(np.mean(cm_GAF_10,axis=2))
      #print(np.std(cm_GAF_10,axis=2))
      
      #plt.imshow(m1[0,:].reshape((368,368)))
      #plt.show()

      #CHANGE DETECTION EXPERIMENTS
      print "Change detection experiments"
      cm_GAF_c_10 = np.zeros((2,2,10),dtype=float)

      #GAF
      for k in range(10):
          cm_GAF_c,mc1,mc2 = red_object.do_GAF_change_experiment(veg,bwt)
          cm_GAF_c[0,:] = cm_GAF_c[0,:]/sum(cm_GAF_c[0,:])*100 #no change
          cm_GAF_c[1,:] = cm_GAF_c[1,:]/sum(cm_GAF_c[1,:])*100 #change
          cm_GAF_c_10[:,:,k] = cm_GAF_c
      
      pickle.dump(cm_GAF_c_10,outfile)
      pickle.dump(mc1[-1,:].reshape((368,368)),outfile)#sim_change
      pickle.dump(mc2[-1,:].reshape((368,368)),outfile)#real change

      print(np.mean(cm_GAF_c_10,axis=2))
      print(np.std(cm_GAF_c_10,axis=2))
          
          #red_object.do_time_series_change_experiment(veg,bwt)

      #veg,bwt = red_object.loadDataSet()

      #BAND DIFFERENCING EXPERIMENT
      print colored('SECOND METHOD: DIFF', 'yellow')
      veg_train,veg_test,c_train,c_test = red_object.create_train_test(veg,bwt)
      veg_test = veg[:,:,7].T  
      c_test = red_object.load_NDVI_change()

      #cf1,cf2 =red_object.compute_z_score(veg_train,veg_test,c_train,c_test,c=1.2)

      #print(cf2)
        
      cv = np.linspace(1.0,2.0,1000)
      AE1 = np.zeros(cv.shape,dtype=float)
      PD2 = np.zeros(cv.shape,dtype=float)
      PFA2 = np.zeros(cv.shape,dtype=float) 
      AE2 = np.zeros(cv.shape,dtype=float)
      for k in range(len(cv)):
          #print(k)
          cf1,cf2 =red_object.compute_z_score(veg_train,veg_test,c_train,c_test,c=cv[k])
          AE1[k] = (cf1[0,1]/200.0 + cf1[1,0]/200.0)/2.0
          AE2[k] = (cf2[0,1]/592.0 + cf2[1,0]/180.0)/2.0 
          PD2[k]= cf2[1,1]/180.0
          PFA2[k] = cf2[0,1]/592.0
          
         
      idx = np.argmin(AE2)
      cf1,cf2 =red_object.compute_z_score(veg_train,veg_test,c_train,c_test,c=1.2)
      print colored(cf2,'yellow')
      cf2[0,:] = cf2[0,:]/sum(cf2[0,:])*100
      cf2[1,:] = cf2[1,:]/sum(cf2[1,:])*100
      pickle.dump(cf2,outfile)
      outfile.close()

      '''
      red_object = RedClass()
      
      veg,bwt = red_object.loadDataSet()

      veg_train,veg_test,c_train,c_test = red_object.create_train_test(veg,bwt)
      veg_test = veg[:,:,7].T  
      c_test = red_object.load_NDVI_change()

      #cf1,cf2 =red_object.compute_z_score(veg_train,veg_test,c_train,c_test,c=1.2)

      #print(cf2)
        
      cv = np.linspace(1.0,2.0,1000)
      AE1 = np.zeros(cv.shape,dtype=float)
      PD2 = np.zeros(cv.shape,dtype=float)
      PFA2 = np.zeros(cv.shape,dtype=float) 
      AE2 = np.zeros(cv.shape,dtype=float)
      for k in range(len(cv)):
          print(k)
          cf1,cf2 =red_object.compute_z_score(veg_train,veg_test,c_train,c_test,c=cv[k])
          AE1[k] = (cf1[0,1]/200.0 + cf1[1,0]/200.0)/2.0
          AE2[k] = (cf2[0,1]/592.0 + cf2[1,0]/180.0)/2.0 
          PD2[k]= cf2[1,1]/180.0
          PFA2[k] = cf2[0,1]/592.0
          
         
      idx = np.argmin(AE2)
      cf1,cf2 =red_object.compute_z_score(veg_train,veg_test,c_train,c_test,c=cv[idx])
      print(cf2) 
      plt.plot(cv,AE1)
      plt.plot(cv,AE2)
      plt.show()
      plt.plot(cv,PFA2)
      plt.plot(cv,PD2)
      plt.show()

      #
      
      
      #print colored('hello', 'red'), colored('world', 'green') 
      '''

      '''
      #IMPORTANT EXPERIMENT FOR PAPER
      
      red_object = RedClass() 
     
      #LOADING DATASET
      veg,bwt = red_object.loadDataSet()
      #print(veg.shape)

      #CONCAT_DATASETS
      X,y = red_object.concatDataSets(veg,bwt)
      #print(X.shape)

      cm_PCA = red_object.do_PCA_experiment(X,y)

      cm_time =  red_object.do_time_experiment(X,y)
      
      cm_GAF = red_object.do_GAF_experiment(X,y)

      '''
      '''
      print colored('FIRST METHOD: PCA','red')

      X_PCA,var_ratio = red_object.PCATransform(X)
      #print(X_PCA.shape)
      
      mean1,mean2,Sigma1,Sigma2 = red_object.constructDensitiesAndPlot(X_PCA)

      H_PCA = np.zeros((X.shape[2],),dtype=float)

      for c in xrange(X.shape[2]):
          H_PCA[c] = red_object.HD(mean1[:,c],Sigma1[:,:,c],mean2[:,c],Sigma2[:,:,c])
      #print("H_PCA = "+str(H_PCA))
      '''
      '''
      XT,var_ratio = red_object.PCATransform(X)
      print(XT.shape)
      
      print colored('SECOND METHOD: FFT','blue') 
      Xf,x_f = red_object.FFTTransform(X)

      X_FFT = red_object.findDomFFTFeatures(X,Xf)
      
      mean1,mean2,Sigma1,Sigma2 = red_object.constructDensitiesAndPlot(X_FFT,fourier=True)
      
      print("Computing HD ...")
      H_FFT = np.zeros((X.shape[2],),dtype=float)

      for c in xrange(X.shape[2]):
          H_FFT[c] = red_object.HD(mean1[:,c],Sigma1[:,:,c],mean2[:,c],Sigma2[:,:,c])
      #print("H_FFT = "+str(H_FFT))

      red_object.plotBar(H_FFT,H_PCA)
      
      print(np.average(H_PCA/H_FFT))

      '''
      '''
      ratio = red_object.FourierInfo(Xf)
      for k in range(ratio.shape[1]):
          plt.plot(np.cumsum(np.ones((ratio.shape[0],))),ratio[:,k])

      plt.show()
      '''
       
      #red_object.FFTExample()

      #for k in range(var_ratio.shape[1]):
      #    plt.plot(np.cumsum(np.ones((var_ratio.shape[0],))),np.cumsum(var_ratio[:,k]))

      #plt.show()
      '''
      gaf_cube = np.zeros((X.shape[0],X.shape[1],X.shape[1]),dtype=float)
     
      for k in range(X.shape[0]):
          print(k)
          gaf_cube[k,:,:] = red_object.transform(X[k,:,0],no_scaling=False)[0]


      #t = red_object.transform(X[0,:,7])
      #print t[0]

      gaf_veg = gaf_cube[:592,:,:]
      gaf_bwt = gaf_cube[592:,:,:]

      plt.imshow(np.mean(gaf_veg,axis=0))
      plt.show()
      plt.imshow(np.mean(gaf_bwt,axis=0))
      plt.show()
      plt.imshow(np.absolute(np.mean(gaf_veg,axis=0)-np.mean(gaf_bwt,axis=0)))
      plt.show()

      new_X = np.zeros((gaf_cube.shape[0],gaf_cube.shape[1]*gaf_cube.shape[1]),dtype=float)

      for k in range(X.shape[0]):
          new_X[k,:] = gaf_cube[k,:,:].flatten()


      #X_train = new_X[:296,:]
      #y_train = y[:296,:]

      idx = np.random.randint(2, size=X.shape[0])
      
      #idx_reverse = 1-idx

      X_train = new_X[idx==1,:]
      y_train = y[idx==1]

      #print(idx)
      #print(y)

      X_test = new_X[idx==0,:]
      y_test = y[idx==0] 

      #from sklearn.datasets import load_iris
      from sklearn.linear_model import LogisticRegression
      #>>> X, y = load_iris(return_X_y=True)
      clf = LogisticRegression(random_state=0).fit(X_train, y_train)
      y_pred = clf.predict(X_test)

      from sklearn.metrics import confusion_matrix
      cm = confusion_matrix(y_test,y_pred)
      c = np.array(["s","v"])
      red_object.plot_confusion_matrix(cm,c)
      plt.show()
      print(cm) 

      print(y_test)
      print(y_pred-y_test)

      X_train = X[idx==1,:,0]
      y_train = y[idx==1] 

      X_test = X[idx==0,:,0]
      y_test = y[idx==0] 

      clf = LogisticRegression(random_state=0).fit(X_train, y_train)
      y_pred = clf.predict(X_test)

      from sklearn.metrics import confusion_matrix
      cm = confusion_matrix(y_test,y_pred)
      c = np.array(["s","v"])
      red_object.plot_confusion_matrix(cm,c)
      plt.show()
      print(cm) 

      pca = PCA(n_components=new_X.shape[0],whiten=False)
      X_PCA = pca.fit(new_X).transform(new_X)

      for k in range(X_PCA.shape[0]):
          if y[k] == 0:
             plt.plot(X_PCA[k,0],X_PCA[k,1],"ro")
          else:
             plt.plot(X_PCA[k,0],X_PCA[k,1],"bo")

      plt.show()
      

      #>>> clf.predict(X[:2, :])
      #array([0, 0])
      #>>> clf.predict_proba(X[:2, :])
      #array([[9.8...e-01, 1.8...e-02, 1.4...e-08],
      #       [9.7...e-01, 2.8...e-02, ...e-08]])
      #>>> clf.score(X, y)

      #plt.plot(t[-1])
      #plt.show()
      
      #PLOT A SINGLE VEGETATION MODIS PIXEL (over all time and bands - except NDVI) 
      for k in range(7):
          plt.plot(X[0,:,k])
      plt.title("VEGETATION")
      plt.show()
          
      #PLOT A SINGLE SETTLEMENT MODIS PIXEL (over all time and bands - except NDVI) 
      for k in range(7):
        plt.plot(X[0,:,k])

      plt.title("SETTLEMENT") 
      plt.show()
      '''
