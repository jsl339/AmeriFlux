import numpy as np
import pandas as pd
import os
import sys

def load_data(dir=None):

    # check if final file is already downloaded
    if dir is not None:

        #check if file is in directory provided
        if os.path.exists(dir+"/phenocam_data_final.csv"):
            print("Reading data from directory provided")
            return pd.read_csv(dir+"/phenocam_data_final.csv")
        else:
            print("Data not found in directory provided")
            sys.exit()

    else:

        print("Checking whether data has been downloaded and saved previously")
        if os.path.exists("../data/phenocam_data_final.csv"):
            print("It has, reading data from data directory")
            return pd.read_csv("../phenocam_data_final.csv")
        else:
            print("Data not found, downloading to data directory")
            # download data
            
            # preprocess data

            # save to data directory

