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
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import itertools
from matplotlib.colors import ListedColormap
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import os

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
M_LENGTH = 45

class MODIS():

  def __init__(self):
      pass

  #Function to help with plotting confidence interval
  def eigsorted(self,cov):
      vals, vecs = np.linalg.eigh(cov)
      order = vals.argsort()[::-1]
      return vals[order], vecs[:,order]

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

  
  def timeVaryingModel(self,X,y,bands=[1,6],algo="KMEANS"):
      true_set_model,set_model,true_veg_model,veg_model,cm_year = self.yearModel(np.copy(X),np.copy(y),bands=bands,algo="KMEANS")
      #model -> [time(45),bands]
     

      X = X[:,:,bands]
      model = []

      for k in range(45):
          if algo=="KMEANS":
             if len(bands) <> 1:    
                model.append(KMeans(n_clusters=2, random_state=0).fit(np.squeeze(X[:,k,:])))
             else:
                model.append(KMeans(n_clusters=2, random_state=0).fit(np.squeeze(X[:,k,:]).reshape(-1, 1)))
          else:
             if len(bands)<>1:
                model.append(mixture.GaussianMixture(n_components=2, random_state=0).fit(np.squeeze(X[:,k,:])))
             else:
                model.append(mixture.GaussianMixture(n_components=2, random_state=0).fit(np.squeeze(X[:,k,:]).reshape(-1, 1)))
 
       
      #DETERMINE BEST LABELS:
      #**********************

      model0_label = np.zeros((45,),dtype=int)   
      model1_label = np.ones((45,),dtype=int)

      for k in range(45):
          
          if algo=="KMEANS":
             model_0_values_k = model[k].cluster_centers_[0,:] 
             model_1_values_k = model[k].cluster_centers_[1,:]
          else:
             model_0_values_k = model[k].means_[0,:] 
             model_1_values_k = model[k].means_[1,:]

          e1 = np.sum((set_model[k,:]-model_0_values_k)**2)  
          e2 = np.sum((veg_model[k,:]-model_1_values_k)**2)  

          e3 = np.sum((set_model[k,:]-model_1_values_k)**2)  
          e4 = np.sum((veg_model[k,:]-model_0_values_k)**2)  
  
          if (e1+e2) >= (e3+e4):
             model0_label[k] = 1 #SETTLEMENT
             model1_label[k] = 0 #VEGETATION
 

          #e1 = (model[k].cluster_centers_[0,0]-model1_reshaped[k,0])**2+(model[k].cluster_centers_[0,1]-model1_reshaped[k,1])**2
          #e2 = (model[k].cluster_centers_[1,0]-model2_reshaped[k,0])**2+(model[k].cluster_centers_[1,1]-model2_reshaped[k,1])**2

          #e3 = (model[k].cluster_centers_[1,0]-model1_reshaped[k,0])**2+(model[k].cluster_centers_[1,1]-model1_reshaped[k,1])**2
          #e4 = (model[k].cluster_centers_[0,0]-model2_reshaped[k,0])**2+(model[k].cluster_centers_[0,1]-model2_reshaped[k,1])**2

          #if (e1+e2) >= (e3+e4):
          #   model1_label[k] = 1
          #   model2_label[k] = 0
   

      if len(bands) == 2:
         plt.plot(veg_model[:,0],veg_model[:,1],"b")
         plt.plot(set_model[:,0],set_model[:,1],"r")   
         c = ["ro","bo"]
         for k in range(45):
             
           if algo=="KMEANS":
                plt.plot(model[k].cluster_centers_[0,0],model[k].cluster_centers_[0,1],c[model0_label[k]])     
                plt.plot(model[k].cluster_centers_[1,0],model[k].cluster_centers_[1,1],c[model1_label[k]])
           else:
                plt.plot(model[k].means_[0,0],model[k].means_[0,1],c[model0_label[k]])     
                plt.plot(model[k].means_[1,0],model[k].means_[1,1],c[model1_label[k]])


         plt.show()

      
      #COMPUTE STD (A FIRST ATTEMPT)
      if algo == "KMEANS":
         d = np.zeros((45,X.shape[0]))
         for k in range(45):
             for p in range(X.shape[0]):
                 #X[p,k,bands]
                 #model[k].labels_[p]
                 #d[k,p]
                 d[k,p] = np.sqrt(np.sum((X[p,k,:] - model[k].cluster_centers_[model[k].labels_[p],:])**2))
                 
         d = np.std(d,axis=0) 
      else:
         d = -1   

      print("d = "+str(d))     
      
      return model, model0_label, model1_label, d  
  
  #1 is vegetation
  #0 is settlement
  ################
  def SPRT_supervised(self,X,y,vegmodel,setmodel,bands=[0,1]):
      #rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])  
      X = X[:,:,bands]

      
      
      vegetation = []
      settlement = []
      
      for k in range(45):
          vegetation.append(multivariate_normal(vegmodel[k].means_[0,:],vegmodel[k].covariances_[0,:,:]))
          settlement.append(multivariate_normal(setmodel[k].means_[0,:],setmodel[k].covariances_[0,:,:]))
       
      sprt_value = np.zeros((X.shape[0],X.shape[1]))   

      for t in range(X.shape[1]):
          print(t)
          
          for p in range(X.shape[0]):
              num = vegetation[t%45].pdf(np.squeeze(X[p,t,:]))
              den = settlement[t%45].pdf(np.squeeze(X[p,t,:]))
              #print(num)
              #print(den)
              if t == 0:
                 print(num)
                 print(den)
                 sprt_value[p,0] = np.log(num)-np.log(den)
              else:
                 sprt_value[p,t] = sprt_value[p,t-1] + (np.log(num)-np.log(den)) 
      
      y_pred = np.zeros(y.shape)
      y_pred[sprt_value[:,-1]>0] = 1
      
      c = ["r","b"]
      for p in range(X.shape[0]):
          plt.plot(sprt_value[p,:],c[int(y[p])],alpha=0.1)
      plt.show()

      cm = confusion_matrix(y,y_pred)
      self.plot_confusion_matrix(cm,["s","v"])
      plt.show()  
      

  
  #1 is vegetation
  #0 is settlement
  ################
  def SPRT_classifier(self,X,y,model,model0_label,model1_label, d, bands=[1,6]):
      #rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])  
      X = X[:,:,bands]
      
      vegetation = []
      settlement = []
      
      for k in range(45):
          d_mat = (d[k]**2)*np.diag(np.ones((len(bands),)))
          settlement.append(multivariate_normal(model[k].cluster_centers_[model0_label[k],:],d_mat))
          vegetation.append(multivariate_normal(model[k].cluster_centers_[model1_label[k],:],d_mat))
              
      sprt_value = np.zeros((X.shape[0],X.shape[1]))   
   
      for p in range(X.shape[0]):
          print(p)
          for t in range(X.shape[1]):
              num = vegetation[t%45].pdf(X[p,t,:])
              den = settlement[t%45].pdf(X[p,t,:])
              if t == 0:
                 sprt_value[p,0] = np.log(num/den)
              else:
                 sprt_value[p,t] = sprt_value[p,t-1] + np.log(num/den) 
      y_pred = np.zeros(y.shape)
      y_pred[sprt_value[:,-1]>0] = 1
      
      c = ["r","b"]
      for p in range(X.shape[0]):
          plt.plot(sprt_value[p,:],c[int(y[p])],alpha=0.1)
      plt.show()

      cm = confusion_matrix(y,y_pred)
      self.plot_confusion_matrix(cm,["s","v"])
      plt.show()  
  #1 is vegetation
  #0 is settlement
  ################
  def SPRT_classifierGMM(self,X,y,model,model0_label,model1_label, d, bands=[1,6]):
      #rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])  
      X = X[:,:,bands]
      
      #vegetation = []
      #settlement = []
      
      #for k in range(45):
      #    d_mat = (d[k]**2)*np.diag(np.ones((len(bands),)))
      #    settlement.append(multivariate_normal(model[k].cluster_centers_[model0_label[k],:],d_mat))
      #    vegetation.append(multivariate_normal(model[k].cluster_centers_[model1_label[k],:],d_mat))
              
      sprt_value = np.zeros((X.shape[0],X.shape[1]))   

      for t in range(X.shape[1]):
          resp = model[t%45].predict_proba(np.squeeze(X[:,t,:]))
          for p in range(X.shape[0]):
              num = resp[p,model1_label[t%45]]
              den = resp[p,model0_label[t%45]]

              if t == 0:
                 sprt_value[p,0] = np.log(num)-np.log(den)
              else:
                 sprt_value[p,t] = sprt_value[p,t-1] + (np.log(num)-np.log(den)) 
      
      y_pred = np.zeros(y.shape)
      y_pred[sprt_value[:,-1]>0] = 1
      
      c = ["r","b"]
      for p in range(X.shape[0]):
          plt.plot(sprt_value[p,:],c[int(y[p])],alpha=0.1)
      plt.show()

      cm = confusion_matrix(y,y_pred)
      self.plot_confusion_matrix(cm,["s","v"])
      plt.show()  
          

  #X - [observations,time (0-44),bands]
  #y - [observations]
  #vegetation -- 1 and settlement --- 0
  def yearModel(self,X,y,bands=[0,1],algo="KMEANS"):
      
      
      X = X[:,:,bands]

      X_reshaped = np.squeeze(X[:,:,0])

      for b in range(1,len(bands)):
          X_reshaped = np.concatenate((X_reshaped,np.squeeze(X[:,:,b])),axis=1)

      #X_reshaped --- (observations,bands*45)
      
      if algo == "GMM":
         kmeans = mixture.GaussianMixture(n_components=2, random_state=0)
         kmeans.fit(X_reshaped)
         model0 = kmeans.means_[0,:].reshape((len(bands),45)).T #ASSOCIATED WITH THE 0 LABEL
         model1 = kmeans.means_[1,:].reshape((len(bands),45)).T #ASSOCIATED WITH THE 1 LABEL
      else:
         kmeans = KMeans(n_clusters=2, random_state=0)
         kmeans.fit(X_reshaped)
         model0 = kmeans.cluster_centers_[0,:].reshape((len(bands),45)).T #ASSOCIATED WITH THE 0 LABEL
         model1 = kmeans.cluster_centers_[1,:].reshape((len(bands),45)).T #ASSOCIATED WITH THE 1 LABEL

      #--- need to align the two models ---
      #COMPUTE TRUE MEAN MODEL
      veg_model = np.mean(X_reshaped[y==1,:],axis=0).reshape((len(bands),45)).T
      set_model = np.mean(X_reshaped[y==0,:],axis=0).reshape((len(bands),45)).T

      #--- need to determine the label of model-0 and model-1 ---
      e1 = np.sum(model0-set_model)**2
      e2 = np.sum(model1-veg_model)**2
      
      e3 = np.sum(model1-set_model)**2
      e4 = np.sum(model0-veg_model)**2
      
      if (e1+e2) < (e3+e4):
         #model 0 is settlement
         #model 1 is vegetation
         if algo == "KMEANS":
            cm = confusion_matrix(y,kmeans.labels_)
         else:
            cm = confusion_matrix(y,kmeans.predict(X_reshaped))

         #plt.plot(set_model[:,0],set_model[:,1],"rx")
         #plt.plot(model0[:,0],model0[:,1],"ro")
         #plt.plot(veg_model[:,0],veg_model[:,1],"bx")
         #plt.plot(model1[:,0],model1[:,1],"bo")
         #plt.show()
         return set_model,model0,veg_model,model1,cm
      else:
         #model 1 is settlement
         #model 0 is vegetation
         if algo == "KMEANS":
            cm = confusion_matrix(y,np.absolute(kmeans.labels_-1))
         else:
            cm = confusion_matrix(y,np.absolute(kmeans.predict(X_reshaped)-1))

         #plt.plot(set_model[:,0],set_model[:,1],"rx")
         #plt.plot(model1[:,0],model1[:,1],"ro")
         #plt.plot(veg_model[:,0],veg_model[:,1],"bx")
         #plt.plot(model0[:,0],model0[:,1],"bo")
         #plt.show() 
         return set_model,model1,veg_model,model0,cm


  def createDictionary(self,X,y):

      EXTRA_TIME = 8

      #print(X.shape)

      #X_temp = np.reshape(X,(X.shape[1],X.shape[0],X.shape[2]))

      #print(X_temp.shape)

      X_45 = np.array([])
      y_45 = np.array([])

      start = 0
      stop = M_LENGTH  
   
      while stop < X.shape[1]:
            if len(X_45) == 0:
               X_45 = X[:,start:stop,:]
               y_45 = np.copy(y)
            else:
               X_45 = np.concatenate((X_45,X[:,start:stop,:]))
               y_45 = np.concatenate((y_45,y))
            
            start += M_LENGTH 
            stop += M_LENGTH 
            #print(stop)
            print(start)

      X_model = {}
      y_model = {}

      start = X.shape[1]-EXTRA_TIME

      for i in range(M_LENGTH):
          temp_model = np.squeeze(X_45[:,i,:])
          temp_y = np.copy(y_45)
          if i <= EXTRA_TIME-1:
             temp_model = np.concatenate((temp_model,np.squeeze(X[:,start+i,:])))
             temp_y = np.concatenate((temp_y,y))
          X_model[i] = temp_model
          y_model[i] = np.copy(temp_y)

      for key in X_model.keys():
          print(key)
          print(X_model[key].shape)
          print(y_model[key].shape)

      #plt.plot(y_model[0])
      #plt.show()

      return X_model,y_model,X_45,y_45

      #ax = plt.gca()
      #nstd = 2.0
      ##ELLIPSE 1
      #vals, vecs = eigsorted(cov[0,:,:])
      #theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
      #w, h = 2 * nstd * np.sqrt(vals)
      #ell1 = Ellipse(xy=(mean[0,0], mean[1,0]),width=w, height=h, angle=theta,edgecolor='green',facecolor='white',fill=True,linewidth=3,zorder=1)
      #ell1.set_facecolor('none')
      #ax.add_artist(ell1)

      ##ELLIPSE 2
      #vals, vecs = eigsorted(cov[1,:,:])
      #theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
      #w, h = 2 * nstd * np.sqrt(vals)
      #ell2 = Ellipse(xy=(mean[0,1], mean[1,1]),width=w, height=h,angle=theta, edgecolor='red',facecolor='white',fill=True,linewidth=3,zorder=1)
      #ell2.set_facecolor('none')
      #ax.add_artist(ell2)


      #for i in range(X.shape[1]):
      #   col = [[resp[i,1],resp[i,0],0]]
      #   ax.scatter(X[0,i],X[1,i],c=col,zorder=3,alpha=0.2)
      #   ax.set_xlabel("$x_1$")
      #   ax.set_ylabel("$x_2$")
      #   plt.xlim([-6,6])
      #   plt.ylim([-6,6])
      #plt.show()

  def plotEllipsesAllModels(self,X,y,bands=[0,1],sup_set,sup_veg,kmeans_cov,kmeans_set,kmeans_veg,gmm_cov_set,gmm_cov_veg,gmm_set,gmm_veg):
          os.system("mkdir BAND"+str(bands[0])+str(bands[1]))
          cmd = "cd BAND"+str(bands[0])+str(bands[1])
          #print(cmd)

          os.chdir("./BAND"+str(bands[0])+str(bands[1]))
      
          for k in range(45):
              for m in range(3):
                  ax = plt.gca()
                  nstd = 2.0

                  ##PLOTTING ELLIPSES
                  for t in range(2):
                      if t == 0:
                         if m == 0:
                            vals, vecs = self.eigsorted(sup_veg[k].covariances_[0,:,:])
                         elif m == 1:
                            vals, vec = self.eigsorted(kmeans_cov)
                         else:
                            vals, vec = self.eigsorted(gmm_cov_veg)
                      else:
                         if m == 0:
                            vals, vecs = self.eigsorted(sup_set[k].covariances_[0,:,:])
                         elif m == 1:
                            vals, vec = self.eigsorted(kmeans_cov)
                         else:
                            vals, vec = self.eigsorted(gmm_cov_set)

                         
                      theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
                      w, h = 2 * nstd * np.sqrt(vals)
                      if t==0:
                         if m == 0:
                            ell1 = Ellipse(xy=(sup_veg[k].means_[0,0], sup_veg[k].means_[0,1]),width=w,height=h,angle=theta,edgecolor='green',facecolor='white',fill=True,linewidth=3,zorder=1)
                         elif m == 1:
                            ell1 = Ellipse(xy=(kmeans_veg[k][0,0], kmeans_veg[k][0,1]),width=w,height=h,angle=theta,edgecolor='green',facecolor='white',fill=True,linewidth=3,zorder=1,linestyle="--")
                         else:
                            ell1 = Ellipse(xy=(gmm_veg[k][0,0], gmm_veg[k][0,1]),width=w,height=h,angle=theta,edgecolor='green',facecolor='white',fill=True,linewidth=3,zorder=1,linestyle="..")
 
                      else:
                         if m == 0:
                            ell1 = Ellipse(xy=(sup_set[k].means_[0,0], sup_set[k].means_[0,1]),width=w,height=h,angle=theta,edgecolor='red',facecolor='white',fill=True,linewidth=3,zorder=1)
                         elif m == 1:
                            ell1 = Ellipse(xy=(kmeans_set[k][0,0], kmeans_set[k][0,1]),width=w,height=h,angle=theta,edgecolor='red',facecolor='white',fill=True,linewidth=3,zorder=1,linestyle="--")
                         else:
                            ell1 = Ellipse(xy=(gmm_set[k][0,0], gmm_set[k][0,1]),width=w,height=h,angle=theta,edgecolor='red',facecolor='white',fill=True,linewidth=3,zorder=1,linestyle="..")

                      ell1.set_facecolor('none')
                      ax.add_artist(ell1)
                      col = ["r","b"]

                      ax.plot(X[y==0,k,0],X[y==0,k,1],"ro",zorder=3,alpha=0.05)
                      ax.plot(X[y==1,k,0],X[y==1,k,1],"go",zorder=3,alpha=0.05)

              
                      plt.savefig(str(k)+".png")
                      ax.cla()
          os.chdir("..")
      

  def convertGMMmodel(self,model,model0_label,model1_label, bands=[1,6]):
          
      veg_cov = []
      set_cov = []
      vegetation = []
      settlement = []
      
      for k in range(45):
          set_cov.append(model[k].covariances_[model0_label[k],:,:])
          veg_cov.append(model[k].covariances_[model1_label[k],:,:])

          settlement.append(model[k].means_[model0_label[k],:])
          vegetation.append(model[k].means_[model1_label[k],:])

      return veg_cov, set_cov, vegetation,settlement
 
  def convertKmeansmodel(self,model,model0_label,model1_label, d, bands=[1,6]):
            
      k_cov = []
      vegetation = []
      settlement = []
      
      for k in range(45):
          k_cov.append((d[k]**2)*np.diag(np.ones((len(bands),))))
          settlement.append(model[k].cluster_centers_[model0_label[k],:])
          vegetation.append(model[k].cluster_centers_[model1_label[k],:])

      return k_cov,vegetation,settlement
          
  def createSupervisedYearModel(self,X,y,bands=[0,1]):
      X = X[:,:,bands]
      vegetation_model = []
      settlement_model = []

      for k in range(45):
          vegetation_model.append(mixture.GaussianMixture(n_components=1).fit(np.squeeze(X[y==1,k,:])))
          settlement_model.append(mixture.GaussianMixture(n_components=1).fit(np.squeeze(X[y==0,k,:])))

      ##PLOT SUPERVISED MODEL AS AN EXCERCISE
      os.system("mkdir BAND"+str(bands[0])+str(bands[1]))
      cmd = "cd BAND"+str(bands[0])+str(bands[1])
      #print(cmd)

      os.chdir("./BAND"+str(bands[0])+str(bands[1]))
          
      for k in range(45):
          print(k)
          ax = plt.gca()
          nstd = 2.0

          ##PLOTTING ELLIPSES
          for t in range(2):
              if t == 0:
                 vals, vecs = self.eigsorted(vegetation_model[k].covariances_[0,:,:])
              else:
                 vals, vecs = self.eigsorted(settlement_model[k].covariances_[0,:,:])
              theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
              w, h = 2 * nstd * np.sqrt(vals)
              if t==0:
                 ell1 = Ellipse(xy=(vegetation_model[k].means_[0,0], vegetation_model[k].means_[0,1]),width=w,height=h,angle=theta,edgecolor='green',facecolor='white',fill=True,linewidth=3,zorder=1)
              else:
                 ell1 = Ellipse(xy=(settlement_model[k].means_[0,0], settlement_model[k].means_[0,1]),width=w,height=h,angle=theta,edgecolor='red',facecolor='white',fill=True,linewidth=3,zorder=1)

              ell1.set_facecolor('none')
              ax.add_artist(ell1)
              col = ["r","b"]

          ax.plot(X[y==0,k,0],X[y==0,k,1],"ro",zorder=3,alpha=0.05)
          ax.plot(X[y==1,k,0],X[y==1,k,1],"go",zorder=3,alpha=0.05)

          #for i in range(X.shape[0]):
              #col = [[resp[i,1],resp[i,0],0]]
              
          #    ax.plot(X[i,k,0],X[i,k,1],col[y[i]],zorder=3,alpha=0.2)
          #    ax.set_xlabel("$x_1$")
          #    ax.set_ylabel("$x_2$")
              
          plt.savefig(str(k)+".png")
          ax.cla()
      os.chdir("..")
      return vegetation_model,settlement_model

  def multi_kmeans_45(self,X,y,bands=np.array([1,6])):
      X = X[:,:,bands]

      X_reshaped = np.squeeze(X[:,:,0])

      for b in range(1,len(bands)):
          X_reshaped = np.concatenate((X_reshaped,np.squeeze(X[:,:,b])),axis=1)

      print(X_reshaped.shape)

      kmeans = KMeans(n_clusters=2, random_state=0)
      kmeans.fit(X_reshaped)
      model1 = kmeans.cluster_centers_[0,:]
      model2 = kmeans.cluster_centers_[1,:]

      veg_model = np.mean(X_reshaped[y==1,:],axis=0)
      set_model = np.mean(X_reshaped[y==0,:],axis=0)

      print("HALLO")
      print(veg_model.shape)

      plt.plot(veg_model[:45],veg_model[45:],"b") 
      plt.plot(set_model[:45],set_model[45:],"r") 
 
      plt.plot(model1[:45],model1[45:],"rx")
      plt.plot(model2[:45],model2[45:],"bx")

      model1_reshaped = np.zeros((45,2))
      model2_reshaped = np.zeros((45,2))     

      model1_reshaped[:,0] = model1[:45]
      model1_reshaped[:,1] = model1[45:]
      model2_reshaped[:,0] = model2[:45]
      model2_reshaped[:,1] = model2[45:]
      
      #plt.show() 

      ##TEST CODE --- JUST TO ILLUSTRATE THE IDEA ---

      model = []

      for k in range(45):
          model.append(KMeans(n_clusters=2, random_state=0).fit(np.squeeze(X[:,k,:])))
          #model.append(mixture.GaussianMixture(n_components=2, random_state=0).fit(np.squeeze(X[:,k,:])))

      #determine best labels
      model1_label = np.zeros((45,),dtype=int)   
      model2_label = np.ones((45,),dtype=int)

      for k in range(45):
          e1 = (model[k].cluster_centers_[0,0]-model1_reshaped[k,0])**2+(model[k].cluster_centers_[0,1]-model1_reshaped[k,1])**2
          e2 = (model[k].cluster_centers_[1,0]-model2_reshaped[k,0])**2+(model[k].cluster_centers_[1,1]-model2_reshaped[k,1])**2

          e3 = (model[k].cluster_centers_[1,0]-model1_reshaped[k,0])**2+(model[k].cluster_centers_[1,1]-model1_reshaped[k,1])**2
          e4 = (model[k].cluster_centers_[0,0]-model2_reshaped[k,0])**2+(model[k].cluster_centers_[0,1]-model2_reshaped[k,1])**2

          if (e1+e2) >= (e3+e4):
             model1_label[k] = 1
             model2_label[k] = 0
 

      #c = ["ro","bo"]

      for k in range(45):
          plt.plot(model[k].cluster_centers_[model1_label[k],0],model[k].cluster_centers_[model1_label[k],1],"ro") 
          plt.plot(model[k].cluster_centers_[model2_label[k],0],model[k].cluster_centers_[model2_label[k],1],"bo")
          
          #plt.plot(model[k].means_[0,0],model[k].means_[0,1],"gx")
          #plt.plot(model[k].means_[1,0],model[k].means_[1,1],"mo")

      #plt.show() 

      model = []

      for k in range(45):
          #model.append(KMeans(n_clusters=2, random_state=0).fit(np.squeeze(X[:,k,:])))
          model.append(mixture.GaussianMixture(n_components=2).fit(np.squeeze(X[:,k,:])))

      #determine best labels
      model1_label = np.zeros((45,),dtype=int)   
      model2_label = np.ones((45,),dtype=int)

      for k in range(45):
          e1 = (model[k].means_[0,0]-model1_reshaped[k,0])**2+(model[k].means_[0,1]-model1_reshaped[k,1])**2
          e2 = (model[k].means_[1,0]-model2_reshaped[k,0])**2+(model[k].means_[1,1]-model2_reshaped[k,1])**2

          e3 = (model[k].means_[1,0]-model1_reshaped[k,0])**2+(model[k].means_[1,1]-model1_reshaped[k,1])**2
          e4 = (model[k].means_[0,0]-model2_reshaped[k,0])**2+(model[k].means_[0,1]-model2_reshaped[k,1])**2

          if (e1+e2) >= (e3+e4):
             model1_label[k] = 1
             model2_label[k] = 0
 

      #c = ["ro","bo"]

      for k in range(45):
          #plt.plot(model[k].cluster_centers_[model1_label[k],0],model[k].cluster_centers_[model1_label[k],1],"ro") 
          #plt.plot(model[k].cluster_centers_[model2_label[k],0],model[k].cluster_centers_[model2_label[k],1],"bo")
          
          plt.plot(model[k].means_[model1_label[k],0],model[k].means_[model1_label[k],1],"rs")
          plt.plot(model[k].means_[model2_label[k],0],model[k].means_[model2_label[k],1],"bs")

      plt.show()   



      


  def kmeans45(self,X,y):
      X = X[:,0:45,:]
      print("K-MEANS")
      model = []

      true_mean_model = []

      for b in range(X.shape[2]):
          model.append(KMeans(n_clusters=2, random_state=0).fit(X[:,:,b]))
          temp_mean = np.zeros((2,X.shape[1]))
          temp_mean[0,:] = np.mean(X[y==0,:,b],axis=0)
          temp_mean[1,:] = np.mean(X[y==1,:,b],axis=0)
          true_mean_model.append(temp_mean)  
                    
          #plt.plot(model[b].cluster_centers_[0,:],"rx")
          #plt.plot(model[b].cluster_centers_[1,:],"bo")
          #plt.plot(temp_mean[0,:],"rs")
          #plt.plot(temp_mean[1,:],"bp")
          #plt.show()
          
          mod1 = np.sum(model[b].cluster_centers_[0,:] - temp_mean[0,:])**2
          mod2 = np.sum(model[b].cluster_centers_[1,:] - temp_mean[1,:])**2 
          
          mod3 = np.sum(model[b].cluster_centers_[0,:] - temp_mean[1,:])**2
          mod4 = np.sum(model[b].cluster_centers_[1,:] - temp_mean[0,:])**2

          if (mod1+mod2) <= (mod3+mod4):
             print("CORRECT LABELS")
             cm = confusion_matrix(y,model[b].labels_)
          else:
             #print(str(mod1+mod2))
             #print(str(mod3+mod4))
             print("SWOP LABELS")
             cm = confusion_matrix(y,np.absolute(model[b].labels_-1))
             
          self.plot_confusion_matrix(cm,["s","v"])
          #plt.show()
          '''
          if (mod1+mod2) <= (mod3+mod4):
             plt.plot(model[b].cluster_centers_[0,:],"rx")
             plt.plot(model[b].cluster_centers_[1,:],"bo")
             plt.plot(temp_mean[0,:],"r")
             plt.plot(temp_mean[1,:],"b")
             plt.show()
          else:
             plt.plot(model[b].cluster_centers_[1,:],"rx")
             plt.plot(model[b].cluster_centers_[0,:],"bo")
             plt.plot(temp_mean[0,:],"r")
             plt.plot(temp_mean[1,:],"b")
             plt.show()
          '''   
           
         
          #if b == 7:
          #   plt.show() 
          #   plt.plot(model[b].labels_,"rx",alpha=0.1)
          #   #plt.plot(y,"bo",alpha=0.1)
          #   plt.show()    

  def gmm45(self,X,y):
      X = X[:,0:45,:]
      print("GMM")
      model = []
      true_mean_model = []

      for b in range(X.shape[2]):
          model.append(mixture.GaussianMixture(n_components=2,covariance_type='spherical').fit(X[:,:,b]))
          temp_mean = np.zeros((2,X.shape[1]))
          temp_mean[0,:] = np.mean(X[y==0,:,b],axis=0)
          temp_mean[1,:] = np.mean(X[y==1,:,b],axis=0)
          true_mean_model.append(temp_mean)  
          
          mod1 = np.sum(model[b].means_[0,:] - temp_mean[0,:])**2
          mod2 = np.sum(model[b].means_[1,:] - temp_mean[1,:])**2 
          
          mod3 = np.sum(model[b].means_[0,:] - temp_mean[1,:])**2
          mod4 = np.sum(model[b].means_[1,:] - temp_mean[0,:])**2

          

          if (mod1+mod2) <= (mod3+mod4):
             print("CORRECT LABELS")
             print(b)
             #print(model[b].predict(X[:,:,b]))
             cm = confusion_matrix(y,model[b].predict(X[:,:,b]))
          else:
             #print(str(mod1+mod2))
             #print(str(mod3+mod4))
             print("SWOP LABELS")
             #print(b)
             cm = confusion_matrix(y,np.absolute(model[b].predict(X[:,:,b])-1))
             
          self.plot_confusion_matrix(cm,["s","v"])
          #plt.show()
         

          #cm = confusion_matrix(y,np.absolute(model[b].predict(X[:,:,b])-1))
          #self.plot_confusion_matrix(cm,["v","s"])
          #plt.show()
          '''
          if (mod1+mod2) <= (mod3+mod4):
             plt.plot(model[b].means_[0,:],"r")
             plt.plot(model[b].means_[1,:],"b")
             plt.plot(temp_mean[0,:],"rs")
             plt.plot(temp_mean[1,:],"bp")
          else:
             plt.plot(model[b].means_[1,:],"r")
             plt.plot(model[b].means_[0,:],"b")
             plt.plot(temp_mean[0,:],"rs")
             plt.plot(temp_mean[1,:],"bp")
          
          plt.show()
          '''
          #if b == 7:
          #   plt.show() 
          #   plt.plot(model[b].labels_,"rx",alpha=0.1)
          #   #plt.plot(y,"bo",alpha=0.1)
          #   plt.show()

  def sequential_k_means(self,Xdict,ydict,X,y):
      band = 4

      temp_mean = np.zeros((2,45))
      #temp_mean[0,:] = np.mean(X[y==0,0:45,band],axis=0)
      #temp_mean[1,:] = np.mean(X[y==1,0:45,band],axis=0)

      yearly_k_means = []

      first_model_label = []
      second_model_label = []

      yearly_mean = np.zeros((2,45))

      for key in Xdict.keys():
          Xnew = Xdict[key]
          ynew = ydict[key]
          temp_mean[0,key] = np.mean(Xnew[ynew==0,band]) 
          temp_mean[1,key] = np.mean(Xnew[ynew==1,band])
          yearly_k_means.append(KMeans(n_clusters=2, random_state=0).fit(Xnew[:,band].reshape(-1, 1))) 
          yearly_mean[0,key] = yearly_k_means[key].cluster_centers_[0]
          yearly_mean[1,key] = yearly_k_means[key].cluster_centers_[1]

          if key == 0:
             first_model_label.append(0)
             second_model_label.append(1)
          else:
             c1 = (yearly_mean[first_model_label[key-1],key-1]-yearly_mean[0,key])**2
             c2 = (yearly_mean[second_model_label[key-1],key-1]-yearly_mean[0,key])**2
             if (c1 <= c2):
                first_model_label.append(0)
                second_model_label.append(1)
             else:
                first_model_label.append(1)
                second_model_label.append(0)
               

      
      plt.plot(temp_mean[0,:],"r")
      plt.plot(temp_mean[1,:],"b")
      for k in range(45):
          plt.plot(k,yearly_mean[first_model_label[k],k],"rx")
          plt.plot(k,yearly_mean[second_model_label[k],k],"bo")
      plt.show()

      overall_label_1 = 0
      overall_label_2 = 1

      d1=0
      d2=0
      d3=0
      d4=0

      for k in range(45):
          d1 += (yearly_mean[first_model_label[k],k]-temp_mean[0,k])**2
          d2 += (yearly_mean[second_model_label[k],k]-temp_mean[1,k])**2

          d3 += (yearly_mean[first_model_label[k],k]-temp_mean[1,k])**2
          d4 += (yearly_mean[second_model_label[k],k]-temp_mean[0,k])**2

      if (d1+d2) <= (d3+d4):
         overall_label_1 = 0
         overall_label_2 = 1
      else:
         overall_label_1 = 1
         overall_label_2 = 0

      c = ["r","b"]

      plt.plot(temp_mean[0,:],"r")
      plt.plot(temp_mean[1,:],"b")
      for k in range(45):
          plt.plot(k,yearly_mean[first_model_label[k],k],c[overall_label_1]+"x")
          plt.plot(k,yearly_mean[second_model_label[k],k],c[overall_label_2]+"o")
      plt.show()

  def validtionCurveTest(self,X_model,y_model,X_old,y_old):
       from sklearn.linear_model import LogisticRegression
       '''
       for t in range(45):
           X = X_model[t]
           y = y_model[t]

           c = ["r","b"]
           c2= ["r","b"]


           plt.plot(X[y==0,1],X[y==0,5],c[0]+"o",alpha=0.1)
           plt.plot(X[y==1,1],X[y==1,5],c[1]+"o",alpha=0.1)
           plt.plot(X_old[y_old==0,t+45,1],X_old[y_old==0,t+45,5],c2[0]+"s",alpha=0.5)
           plt.plot(X_old[y_old==1,t+45,1],X_old[y_old==1,t+45,5],c2[1]+"s",alpha=0.5)
           plt.title(str(t))
           plt.show()
        '''

       X = X_model[12]
       y = y_model[12]

       bands = np.array([False,True,False,False,False,True,False,False])
       
       X = X[:,bands]

       param_range = np.logspace(-10, 10, 10)
       train_scores, test_scores = validation_curve(LogisticRegression(), X, y, param_name="C", param_range=param_range,
       cv=10, scoring="accuracy", n_jobs=1)
       train_scores_mean = np.mean(train_scores, axis=1)
       train_scores_std = np.std(train_scores, axis=1)
       test_scores_mean = np.mean(test_scores, axis=1)
       test_scores_std = np.std(test_scores, axis=1)

       plt.title("Validation Curve with Logistic Regression")
       plt.xlabel("$\gamma$")
       plt.ylabel("Score")
       plt.ylim(0.0, 1.1)
       lw = 2
       plt.semilogx(param_range, train_scores_mean, label="Training score",
              color="darkorange", lw=lw)
       plt.fill_between(param_range, train_scores_mean - train_scores_std,
                  train_scores_mean + train_scores_std, alpha=0.2,
                  color="darkorange", lw=lw)
       plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
              color="navy", lw=lw)
       plt.fill_between(param_range, test_scores_mean - test_scores_std,
                  test_scores_mean + test_scores_std, alpha=0.2,
                  color="navy", lw=lw)
       plt.legend(loc="best")
       plt.show()



if __name__ == "__main__":
   m = MODIS()
   veg,bwt = m.loadDataSet(name="Gauteng_nochange.mat",province="Gauteng")
   X,y = m.concatDataSets(veg,bwt)

   #m.kmeans45(X,y)
   #m.gmm45(X,y)
   
   Xdic,ydic,X45,y45 = m.createDictionary(X,y)
   #print(X45.shape)
   #print(y45.shape)
   #m.multi_kmeans_45(X45,y45)
   #m.yearModel(X45,y45,bands=[0,1])

   vegmodel,setmodel = m.createSupervisedYearModel(X45,y45,bands=[4,5])
   m.SPRT_supervised(X,y,vegmodel,setmodel,bands=[4,5])
   

   #x1,x2,x3,x4,c = m.yearModel(X45,y45,bands=[0,1,2,3],algo="GMM")
   #m.plot_confusion_matrix(c,["s","v"])
   #plt.show()
   #model, model0_label, model1_label, d = m.timeVaryingModel(X45,y45,bands=[0,1,2,3],algo="KMEANS")
   #m.SPRT_classifier(X,y,model,model0_label,model1_label, d, bands=[0,1,2,3])
   #model, model0_label, model1_label, d = m.timeVaryingModel(X45,y45,bands=[0,1,2,3],algo="GMM")
   #m.SPRT_classifierGMM(X,y,model,model0_label,model1_label, d, bands=[0,1,2,3])

   '''
   counter = 0
   a = 0
   for j in range(7):
           x1,x2,x3,x4,c = m.yearModel(X45,y45,bands=[j])
           m.plot_confusion_matrix(c,["s","v"])
           a += 1.0*(c[0,0]+c[1,1])/np.sum(c)
           counter += 1
   print("AVERAGE")
   print(a/counter)   

   counter = 0
   a = 0

   for j in range(7):
       for k in range((j+1),7):
           print(str(j)+" "+str(k))
           x1,x2,x3,x4,c = m.yearModel(X45,y45,bands=[j,k])
           m.plot_confusion_matrix(c,["s","v"])
           a += 1.0*(c[0,0]+c[1,1])/np.sum(c)
           counter += 1

   print("AVERAGE")
   print(a/counter)  
      
   #print((c/(1.0*np.sum(c)))*100)

   #m.sequential_k_means(Xdic,ydic,X,y)
   
   #print(X.shape)
   #print(y.shape)
   
   #Xdic,ydic = m.createDictionary(X,y)

   #m.validtionCurveTest(Xdic,ydic,X,y)
   '''
   
   
      
   
