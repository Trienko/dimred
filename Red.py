import scipy.io
import pylab as plt


class RedClass():

  def __init__(self):
      pass

  def loadDataSet(self,name="Gauteng_nochange.mat",province="Gauteng"):

      mat = scipy.io.loadmat(name)
      #print(mat.keys())

      if province == "Gauteng":
         veg = mat['veg_gauteng']
         bwt = mat['bwt_gauteng']
    
      #(time,pixels,band)
      #(time,pixels,7) - NDVI
                   
      return veg,bwt


if __name__ == "__main__":
      red_object = RedClass() 
      veg,bwt = red_object.loadDataSet()
      print(veg.shape)
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
