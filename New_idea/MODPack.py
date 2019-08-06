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
       
if __name__ == "__main__":
      

      #print colored('hello', 'red'), colored('world', 'green') 
     
      red_object = RedClass() 
     
      #LOADING DATASET
      veg,bwt = red_object.loadDataSet()
      #print(veg.shape)

      #CONCAT_DATASETS
      X,y = red_object.concatDataSets(veg,bwt)
      #print(X.shape)

      print colored('FIRST METHOD: PCA','red')

      X_PCA,var_ratio = red_object.PCATransform(X)
      #print(X_PCA.shape)
      
      mean1,mean2,Sigma1,Sigma2 = red_object.constructDensitiesAndPlot(X_PCA)

      H_PCA = np.zeros((X.shape[2],),dtype=float)

      for c in xrange(X.shape[2]):
          H_PCA[c] = red_object.HD(mean1[:,c],Sigma1[:,:,c],mean2[:,c],Sigma2[:,:,c])
      #print("H_PCA = "+str(H_PCA))

      '''
      XT,var_ratio = red_object.PCATransform(X)
      print(XT.shape)
      '''
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
