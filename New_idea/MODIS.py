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

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
M_LENGTH = 45

class MODIS():

  def __init__(self):
      pass

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

      return X_model,y_model

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
      band = 7

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
          d1 += (yearly_mean[first_model_label[k],k]-temp_mean[0,:])**2
          d2 += (yearly_mean[second_model_label[k],k]-temp_mean[1,:])**2

          d3 += (yearly_mean[first_model_label[k],k]-temp_mean[1,:])**2
          d4 += (yearly_mean[second_model_label[k],k]-temp_mean[0,:])**2

      if d


      
      
  
      

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
   
   Xdic,ydic = m.createDictionary(X,y)
   m.sequential_k_means(Xdic,ydic,X,y)

   #print(X.shape)
   #print(y.shape)
   
   #Xdic,ydic = m.createDictionary(X,y)

   #m.validtionCurveTest(Xdic,ydic,X,y)

   
   
      
   
