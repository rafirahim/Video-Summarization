import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class  videoframes:
    def __init__(self):
        pass
    '''determine whether a given filename refers to a video file based on its file extension ['.mp4','.mpg','.mpeg'].'''
    def is_video_file(self,filename):
        video_file_extensions = ('.mp4','.mpg','.mpeg','.webm')
        if filename.endswith((video_file_extensions)):
            print("Video file detected")
        
            return True
        else:
            print("Unsupporter file format.")
            print("Supported file formats are '.mp4','.mpg','.mpeg','.webm'")

    '''Capturing the video frames'''
    def video_to_frames(self,filename):
        if (self.is_video_file(filename)==True):
            os.system('cls')
            frames=[]
            video=cv2.VideoCapture(filename)
            ret,frame=video.read()
            count=0
            while True:
                if (count%1==0.0):
                    frames.append(frame)
                count+=1
                ret_new,frame=video.read()
                if not ret_new:
                    print("Frame Capturing Completed")
                    break
            return frames


class opticalflow(videoframes):
    def __init__(self):
        super(opticalflow, self).__init__()
        
    def draw_flow(self,img,flow,step):
        step=step/2
        h,w=img.shape[:2]
        y,x=np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
        fx,fy=flow[y,x].T
        lines=np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
        lines=np.int32(lines+0.5)
        vis=img
        cv2.polylines(vis,lines, 0, (0,255,0))
        for (x1,y1),(x2,y2) in lines:
            cv2.circle(vis,(x1,y1),1,(0,255,0),-1)
        return vis
    
    
    '''
    Calculate the optical flow between two images, 'img1' and 'img2', using the Farneback method provided by OpenCV
    Input parameters 'img1', 'img2', and 'patch_size'

    Farneback method. The method takes the following parameters:
    
    current: This is the first frame, represented as a grayscale image. It is the image at time t.

    new: This is the second frame, also represented as a grayscale image. It is the image at time t+1.

    flow: This is the output array that will contain the calculated optical flow vectors. It has the same size as the input images and should have a type of np.float32.

    pyr_scale: This parameter specifies the scale factor between each level of the Gaussian pyramid used to calculate the optical flow. It is a float value between 0 and 1, with smaller values resulting in more levels in the pyramid.

    levels: This parameter specifies the number of levels in the Gaussian pyramid. The optical flow is calculated at the finest level and then upsampled to the coarser levels. Larger values of this parameter result in more levels and a finer optical flow estimate.

    winsize: This parameter specifies the size of the window used to average the flow between pixels. Larger values of this parameter result in a smoother optical flow field.

    iterations: This parameter specifies the number of iterations performed at each pyramid level to refine the flow estimate.

    poly_n: This parameter specifies the size of the pixel neighborhood used to find polynomial expansion of the flow. Larger values of this parameter result in a more complex flow estimate.

    poly_sigma: This parameter specifies the standard deviation of the Gaussian used to smooth derivatives used as a basis for the polynomial expansion. Larger values of this parameter result in a smoother flow estimate.

    flags: This parameter specifies various flags used to control the algorithm. Possible values include 0, cv2.OPTFLOW_USE_INITIAL_FLOW, cv2.OPTFLOW_FARNEBACK_GAUSSIAN, and cv2.OPTFLOW_FARNEBACK_HAAR.
    '''

    def optical_flow_vector(self,img1,img2,patch_size):
        pyr_scale=0.5
        levels=3
        winsize=15
        iterations=3
        poly_n=5
        poly_sigma=1.2
        flags=0

        current=cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
        new=cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
#         current=cv2.cvtColor(cv2.pyrDown(img1),cv2.COLOR_RGB2GRAY)
#         new=cv2.cvtColor(cv2.pyrDown(img2),cv2.COLOR_RGB2GRAY)
        flow=cv2.calcOpticalFlowFarneback(current,new,None,pyr_scale,levels,winsize,iterations,poly_n,poly_sigma,flags)
        #plt.axis('off')
        #plt.imshow(draw_flow(np.zeros_like(img1),flow,patch_size))
        mag,ang=cv2.cartToPolar(flow[...,0],flow[...,1])
        directions=ang*180/np.pi/2
        Fx,Fy=flow[:,:,0],flow[:,:,1]
        F=[Fx,Fy]
        #vect=mag*np.cos(directions)
        return F
    
