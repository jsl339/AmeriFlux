import numpy as np
import pandas as pd
import os
import sys


def load_data(user_id, user_email, dir=None):

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

            # checking if Ameriflux username and email have been provided
            if user_id == None or user_email == None:
                print("Must provide registered user and email to download AmeriFlux data")
                sys.exit()

            # download and preprocess data
            download_data_using_R(user_id, user_email)
            
            # save to data directory


def download_data_using_R(user_id_arg, user_email_arg):
    current_path = os.getcwd() 
    data_path = current_path[:-4]+"data/"
    os.system("Rscript data_download.R -u "+str(user_id_arg)+" -e "+str(user_email_arg)+" -d "+str(data_path))


if __name__ == "__main__":
    load_data("test", "email@email.com")