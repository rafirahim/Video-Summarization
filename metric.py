import numpy as np
import OpticalFlow
import cv2
import ParameterEstimation
import math
import matplotlib.pyplot as plt
class metric:
    def __inint__(self):
        pass
    '''Defining a function to return the sigmoid value of the input
    '''
    def sigmoid(self,x):
        z = 1/(1 + np.exp(-x))
        return z
    '''Defining a function to return the tanh value of the input
    '''
    def tanh(self,x):
        x=math.tanh(x)
        return x
    
    '''Defining the proposed metric to take the eztimated parameters of the distribution as input,
        to return the optical flow similarity between two pairs of images (im1, im2) and (im3,im4).
        The function return the metric value, tanh of the metric value and the sigmoid of the metric value.
    '''
    def metric(self,alpha1,loc1,scale1,alpha2,loc2,scale2):
        s=1-((loc1-loc2)**2+0.01)/((scale1-scale2)**2+0.01)
        s=np.abs(s)
        t=metric.tanh(self,s)
        z=metric.sigmoid(self,s)
        return s,t,z
    

    def score_val(self,im1,im2,im3,im4,window,patch_size):
        #plot frames
        #plt.imshow(np.hstack((im1,im2,im3,im4)))
        #plt.show()
        # op1=cv2.calcOpticalFlowFarneback(cv2.cvtColor(im1,cv2.COLOR_RGB2GRAY),cv2.cvtColor(im2,cv2.COLOR_RGB2GRAY),None,0.5,3,15,3,5,1.2,0)
        # op2=cv2.calcOpticalFlowFarneback(cv2.cvtColor(im3,cv2.COLOR_RGB2GRAY),cv2.cvtColor(im4,cv2.COLOR_RGB2GRAY),None,0.5,3,15,3,5,1.2,0)
        # flow1=OpticalFlow.opticalflow().draw_flow(np.zeros_like(im1),op1,patch_size)
        # flow2=OpticalFlow.opticalflow().draw_flow(np.zeros_like(im3),op2,patch_size)
        # plt.imshow(np.hstack((flow1,flow2)))
        # plt.show()
        ''' Estimating the weibull parametrs of the distribution of optical flow betwwen the respective pairs of video frame.
        '''
        a1,b1,c1=ParameterEstimation.parameter_estimation().fit_hist(im1,im2,window,patch_size)
        a2,b2,c2=ParameterEstimation.parameter_estimation().fit_hist(im3,im4,window,patch_size)
        s=metric().metric(a1,b1,c1,a2,b2,c2)
        return s
