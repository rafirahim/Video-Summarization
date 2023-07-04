import OpticalFlow
import numpy as np
import torch
import math
import torch
import torch.nn.functional as F

class distribution:
    
    def __init__(self):
        pass
    ''' Gradient function'''
    def gradient(self,vect):
        Fx,Fy=vect[0],vect[1]
        x_gradient=np.gradient(Fx,axis=0)
        y_gradient=np.gradient(Fy,axis=1)
        return (x_gradient,y_gradient)
    
    ''' Function to return 1D tensor containing values sampled from a Gaussian distribution.
        Input parameters:
            win_size: The size of the Gaussian window.
            sigma: The standard deviation of the Gaussian distribution.
    '''
    def gaussian(self,win_size,sigma):
        gauss =  torch.Tensor([math.exp(-(x - win_size//2)**2/float(2*sigma**2)) for x in range(win_size)])
        return gauss/gauss.sum()
    
    ''' Function to convert Vector to Tensor
    '''
    def cvt_tensor(self,vect):
        vect=torch.tensor(vect)
        vect=vect[None,:]
        return vect
    
    ''' Function returning Gaussian Window by taking the parametrs 
            gauss': A 1D tensor containing values sampled from a Gaussian distribution.
            win_size: Size of the Gaussian Window
            channel: The number of channels in the image.
            sigma : Standard deviation
    '''
    def create_window(self,gauss,win_size, channel):

        # Generate an 1D tensor containing values sampled from a gaussian distribution
        _1d_window = gauss.unsqueeze(1)
        
        # Converting to 2D  
        _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
         
        window = torch.Tensor(_2d_window.expand(channel, 1, win_size, win_size).contiguous())
    
        return window
    
    ''' Function used to map the weighted sum of the x and y components of the opatical floe and its respective gradients to a numpy array.

    
    '''
    def distribute(self,img1,img2,patch_size,window):
        flow1=OpticalFlow.opticalflow().optical_flow_vector(img1,img2,patch_size)
        #X and Y components of the flow Vector
        Fx_1,Fy_1=torch.tensor(flow1[0])[None,:],torch.tensor(flow1[1])[None,:]
        Fx_1,Fy_1=Fx_1,Fy_1
        
        # Gradient adn Divergence of the flow1 vector in x and y direction
        Gx_1,Gy_1=distribution.gradient(self,flow1)
        Gx_1,Gy_1=distribution.cvt_tensor(self,Gx_1),distribution.cvt_tensor(self,Gy_1)
        Gx_1,Gy_1=Gx_1,Gy_1
        #Padding 
        pad=11//2
        #Weighted sums from flow
        mu_sum_Fx_1=F.conv2d(Fx_1,window,padding=pad)
        mu_sum_Fy_1=F.conv2d(Fy_1,window,padding=pad)
        mu_sum_Gx_1=F.conv2d(Gx_1,window,padding=pad)
        mu_sum_Gy_1=F.conv2d(Gy_1,window,padding=pad)   
        
        #Mapping
        fun=(mu_sum_Fx_1* mu_sum_Gx_1 + mu_sum_Fy_1*mu_sum_Gy_1)
        fun=fun.numpy()
        return fun
    
    