import glob
import pandas as pd
import os

#"/Users/johnleland/Desktop/Ameriflux_Data/XGregression/LE50/KFold/LE50Metrics_KFold0.csv"

print("p")
path = "/Users/johnleland/Desktop/Ameriflux_Data/XGregression"

def collection(path_name):
    files = glob.glob((path_name))
    pdm = {}
    pdm = [pd.read_csv(f) for f in files]
    merged = pd.concat(pdm)
    merged = merged.sort_values(by=["Fold"])
    merged.to_csv(path + "/FC50/Metrics/LOGOResults.csv")
    return merged

collection( "/Users/johnleland/Desktop/Ameriflux_Data/XGregression/FC50/LOGO/*.csv")
print("All Done")