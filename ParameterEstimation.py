import distribution
import numpy as np
import scipy
from scipy.stats import weibull_min


class parameter_estimation:
    def __init__(self):
        pass
    '''Function takes in a 2D array (image) and a patch size as input parameters and divides the 
      array into smaller patches of the given patch size, extracts each patch, and stores it in a list.
      Finally, it returns the list of patches.'''

    def patchify(self,array,patch_size):
        w,h=array.shape[:2]
        patch=[]
        for i in range(0,w,patch_size):
            for j in range(0,h,patch_size):
                ptc=array[i:i+patch_size,j:j+patch_size]
                patch.append(ptc)
        return patch
    '''Defining a function to estimate the Weibull parameters from the obtained distribution.
        Input parameters:
            img1: A NumPy array representing the first image.
            img2: A NumPy array representing the second image.
            window: A PyTorch tensor representing the 2D Gaussian window used for the weighted sums.
            patch_size: An integer representing the size of the patches used to divide the images.

    '''
    def fit_hist(self,img1,img2,window,patch_size):
        data=distribution.distribution().distribute(img1,img2,patch_size,window)
        
        data=parameter_estimation.patchify(self,data,patch_size)
        m=[]
        for y in data:
            m.append(y.mean())
        #_, bins, _ = plt.hist(m,20, density=2, alpha=0.5)  
        params=weibull_min.fit(m)
        alpha,loc,scale=params
        #alpha,loc,scale=round(alpha,3),round(loc,3),round(scale,3)
        mu, sigma = scipy.stats.norm.fit(m)
        #best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
        #plt.axis('off')
        #plt.plot(bins, best_fit_line)
        #print(alpha,loc,scale)
        #plt.show()
        #print("mean:",mu,"sigma:",sigma)
            
        return params
    
   
                