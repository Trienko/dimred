import scipy.io
import pylab as plt


class Mat():

      def __init__(self):
          pass

      def plot_mat(self,name=""):
          mat = scipy.io.loadmat(name)
          print mat.keys()
          veg_gauteng = mat['veg_gauteng']
          
          #(time,pixels,band)
          #(time,pixels,7) - NDVI
          print veg_gauteng.shape

          #(time,pixels,band)
          #(time,pixels,7) - NDVI
          bwt_gauteng = mat['bwt_gauteng']
          #bldg_gauteng = mat['bldg_gauteng']
          print bwt_gauteng.shape
                   
          #PLOT A SINGLE VEGETATION MODIS PIXEL (over all time and bands - except NDVI) 
          for k in xrange(7):
              plt.plot(veg_gauteng[:,0,k])
          plt.title("VEGETATION")
          plt.show()
          
          #PLOT A SINGLE SETTLEMENT MODIS PIXEL (over all time and bands - except NDVI) 
          for k in xrange(7):
              plt.plot(bwt_gauteng[:,0,k])

          plt.title("SETTLEMENT") 
          plt.show()      
                
             
if __name__ == "__main__":
   f = Mat()
   f.plot_mat(name="Gauteng_nochange.mat")
