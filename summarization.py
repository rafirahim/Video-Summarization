import torch.cuda
import numpy as np
import cv2
from scipy.signal import argrelextrema
import time
import os
from tqdm.notebook import tqdm
import torch
import torch.cuda
import matplotlib.pyplot as plt
from rich.progress import track
import pandas as pd
import OpticalFlow
import distribution
import metric

class summary:
    def __inint__(self):
        self.win_size=11
        self.sigma=1.5
        self.channel=1
        self.window=distribution.distribution().create_window(distribution.distribution().gaussian(self.win_size,self.sigma),self.win_size,self.channel)
        self.patch_size=16

        pass
    '''Defining a function to return the sigmoid value of the input
    '''
    def frame_extraction(self,video,sparsity):
        of=OpticalFlow.videoframes()
        #Capturing frames from the video
        Frames=of.video_to_frames(video)
        flag=0
        score=[]
        for i,f in enumerate(track(Frames[:-2])):
                    ref_frame=Frames[flag]
                    # Calculateing the saptio-temporal similary metric vlaue
                    s,t,z=metric.metric().score_val(Frames[i],Frames[i+1],Frames[i+1],Frames[i+2],self.window,self.patch_size)
                    score.append(s)
        # Sorting based on the clculated metric value            
        sorted_indices = np.argsort(score)
        # Spersifying 
        frame_indices=np.sort(sorted_indices[:int(len(Frames)*sparsity)])
        return (Frames, score, frame_indices)
    
    def create_video_from_frames(self,frames,index, output_filename, fps=30):
        # Determine the dimensions of the frames
        height, width, _ = frames[0].shape

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for AVI format
        video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

        # Write the frames to the video file
        for x in index:
            video_writer.write(frames[x])

        # Release the video writer and close the output file
        video_writer.release()

        print(f"Video '{output_filename}' has been created successfully.")

    def save_key_frames(self,Frames, indeces, path):
        i=0
        for x in indeces:
            # Saving key_frames
            cv2.imwrite(path+"/frame"+str(i+1)+".jpg", Frames[x])
            i+=1
        print("Key Frames saved in ", path)

    def plot_score(self,file_name,score, indices):
        score=np.array(score)
        fig = plt.figure(figsize=(100,20))
        # Plotting score values of the metric
        plt.plot(score)
        # Plotiing index values of the key frames above the plot of score value
        plt.scatter(x=indices,y=score[indices], color='r', marker='*')
        # Savig the plot
        plt.savefig(file_name+"_score_plot")
        print("PyPot saved in", file_name)

    def video_summary(self, video, sparsity):
        start_time = time.time()
        Frames,score,indices=self.frame_extraction(video, sparsity)
        # Making an output directory with the name of the input video to save the results
        output_directory=os.path.splitext(video)[0]
        os.mkdir(output_directory)
        # Definig the file name for the summary of the input video
        output_filename=output_directory+"/summary"+".mp4"
        # Saving video summary 
        self.create_video_from_frames(Frames,indices,output_filename)
        # Saving key_frames
        self.save_key_frames(Frames,indices,output_directory)
        # Plotting score
        self.plot_score(output_directory,score, indices)
        # Calculating the time take 
        elapsed_time = time.time() - start_time
        return(len(Frames), len(indices),elapsed_time)  
    
    def get_summary(self,folder,sparsity):
        '''
    Calculate the optical flow between two images, 'img1' and 'img2', using the Farneback method provided by OpenCV
    Input parameters 'img1', 'img2', and 'patch_size'

    Farneback method. The method takes the following parameters:
    
    folder:Folder in which the videos are stored.

    sparsity: Sparsity of the summary. The percentage of frames in the summary.

    '''
        data=[]
        object=os.scandir(folder)
        for entry in object:
            if entry.is_file():
                video=folder+"/"+entry.name
                print("Begin summarization of ", video)
                print()
                len_frames, len_key_frames, time_taken=self.video_summary(video,sparsity)
                print(['File Name','No. of Farames in vidoe','No. of Frames in Summary','Sparsity', 'Execution Time'],)

                print([entry.name,len_frames, len_key_frames,100-(len_key_frames*100/len_frames),time_taken])
                data.append([entry.name,len_frames, len_key_frames,100-(len_key_frames*100/len_frames),time_taken])
                pd.DataFrame(np.array(data)).to_csv(folder+'_output.csv', index_label = "Index", header  = ['File Name','No. of Frames in video','No. of Frames in Summary','Sparsity', 'Execution Time'])    
        