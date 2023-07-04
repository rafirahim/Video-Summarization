import argparse
import distribution
import summarization
import os
from tqdm.notebook import tqdm

if __name__=="__main__":
    parser = argparse.ArgumentParser()
       
    parser.add_argument('-f', '--folder',required=True)
    parser.add_argument('-k', '--sparsity',required=True)
    
    args = parser.parse_args()
    path = args.folder
    print(path)
    '''
    Defining the input parameters
    '''
    summarization.get_summary(args.folder, args.sparsity)
    