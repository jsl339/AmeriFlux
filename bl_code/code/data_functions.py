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

            # download Ameriflux data
            # if not os.path.exists("../data/all_sites_Ameriflux.csv"):
            if not os.path.exists("/Users/bml438/Desktop/data/all_sites_Ameriflux.csv"): ###### <------- delete this in the future!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # checking if Ameriflux username and email have been provided
                if user_id == None or user_email == None:
                    print("Must provide registered user and email to download AmeriFlux data")
                    sys.exit()
                download_Ameriflux_data_using_R(user_id, user_email)
            
            # download phenocam data
            # if not os.path.exists("../data/all_sites_phenocam.csv"):
            if not os.path.exists("/Users/bml438/Desktop/data/all_sites_phenocam.csv"): ###### <------- delete this in the future!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                download_phenocam_data_using_R()
            
            merge_flux_and_phenocam_data()


def download_Ameriflux_data_using_R(user_id_arg, user_email_arg):
    current_path = os.getcwd() 
    data_path = current_path[:-4]+"data/"
    os.system("Rscript Ameriflux_data_dl.R -u "+str(user_id_arg)+" -e "+str(user_email_arg)+" -d "+str(data_path))


def download_phenocam_data_using_R():
    current_path = os.getcwd()
    data_path = current_path[:-4]+"data/"
    os.system("Rscript pheno_data_dl.R -d "+str(data_path))


def merge_flux_and_phenocam_data():
    # Read in site data csv
    print("Reading flux data")
    # flux_df = pd.read_csv("../data/all_sites_Ameriflux.csv", header=0, low_memory=False)
    flux_df = pd.read_csv("/Users/bml438/Desktop/data/all_sites_Ameriflux.csv", header=0, low_memory=False) ###### <------- delete this in the future!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    site_names = list(np.unique(flux_df['Site']))
    #Read in pheno data csv
    # pheno_df = pd.read_csv("../data/all_sites_phenocam.csv", header=0, low_memory=False)
    pheno_df = pd.read_csv("/Users/bml438/Desktop/data/all_sites_phenocam.csv", header=0, low_memory=False) ###### <------- delete this in the future!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print("Reading phenocam data")

    # Partition large datsets into new datasets by site name
    flux_df_grouped = flux_df.groupby(flux_df['Site'])
    pheno_df_grouped = pheno_df.groupby(pheno_df['site'])
    dflist= []
    gcclist=[]
    for site in uni:
        dfg1 = grouped.get_group(site)
        dflist.append(dfg1)
        dfg2= grouped2.get_group(site)
        gcclist.append(dfg2)


    # #%%
    # #Quick way to line up datasets by getting a string of Year-DOY
    # cleandflist = []
    # for i,df in enumerate(gcclist):
    #     print(f'Attempting to clean DF{i+1}')
    #     df['YDOY'] = df['year'].astype(str)+ df['doy'].astype(str)
    #     print(f'DF{i+1} -- Complete')
    #     cleandflist.append(df.loc[:,['YDOY','midday_gcc','midday_rcc','gcc_50','gcc_75','gcc_90','rcc_50','rcc_75','rcc_90']])
    # for df in dflist:
    #     df['YDOY'] = df['YEAR'].astype(str)+ df['DOY'].astype(str)


    # #%%
    # #Merge on 'YDOY', maintain size of larger DF
    # #Copies rows of GCC data to match 
    # merge_list = []
    # for df, df1 in zip(cleandflist, dflist):
    #     #print(f'df shape: {df.shape}, df1 shape {df1.shape}')
    #     merge_df = df1.merge(df,how = 'left', on = 'YDOY')
    #     #print(f'merged shape: {merge_df.shape}')
    #     merge_list.append(merge_df)

    # # %%
    # #Concatenate to form one large dataframe
    # final_df = pd.concat(merge_list)

    # #%%
    # final_df.reset_index(inplace = True)
    # final_df.drop(['YDOY'], axis = 1, inplace = True)

    # final_df.drop(['index'], axis = 1, inplace = True)
    # #%%
    # #Free up space
    # del merge_list, cleandflist, gcclist, dflist, df, df2
    # #%%
    # #Columns to keep
    # keep_cols = ['TS_1_1_1',
    # 'TS_1_2_1',
    # 'PPFD',
    # 'Tair',
    # 'VPD',
    # 'SWC_1_1_1',
    # 'HourCos',
    # 'HourSin',
    # 'DOYCos',
    # 'DOYSin',
    # 'FC50',
    # 'Site',
    # 'PPFD_OUT', 
    # 'PPFD_BC_IN_1_1_1',
    # 'Rh',
    # 'NETRAD',
    # 'SW_DIF',
    # 'SW_IN_1_1_1',
    # 'SW_OUT',
    # 'Ustar',
    # 'midday_gcc',
    # 'midday_rcc',
    # 'gcc_50',
    # 'gcc_75',
    # 'gcc_90',
    # 'rcc_50',
    # 'rcc_75',
    # 'rcc_90'
    # ]
    # #Create FC50 
    # data = final_df[final_df.Ustar >= final_df.U50]
    # final_df["FC50"] = data["FC"]

    # #Final df of kept columns
    # final_df = final_df.loc[:,keep_cols]
    # # %%
    # #Remove FC50 outliers
    # def outlier(data):
    #     data.dropna(inplace=True)
    #     x = sorted(data)
    #     q1, q3= np.percentile(x,[25,75])
    #     outside = 6
    #     iqr = q3 - q1
    #     lower_bound = q1 -(outside * iqr) 
    #     upper_bound = q3 +(outside * iqr) 
    #     return lower_bound,upper_bound


    # x = outlier(final_df["FC50"])
    # d = final_df[final_df["FC50"] <= x[1]]

    # d = d[d["FC50"] >= x[0]]

    # dpreds = d.drop(["FC50"], axis = 1)
    # dresponse = d["FC50"]


    # dpreds = dpreds.interpolate()

    # dat = dpreds.join(dresponse)

    # #Drop NA values
    # df2 = dat.dropna()
    # #%%
    # #Save to csv. Change file path
    # df2.to_csv(r"C:\Users\jeffu\Documents\Condensed43+GCC.csv")


if __name__ == "__main__":
    merge_flux_and_phenocam_data()