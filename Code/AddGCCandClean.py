#%%
# Read in CSV of all Neon site data
import numpy as np
import pandas as pd
#Read in Site data csv
df = pd.read_csv(r"C:\Users\jeffu\Documents\44Datasets.csv\44Datasets.csv", header = 0, low_memory=False)
uni = list(np.unique(df['Site']))
#Read in GCC csv
df2 = pd.read_csv(r"C:\Users\jeffu\OneDrive\Documents\43GCCData.csv", header =0, low_memory=False)
#Uncomment to see list of site names
#print(uni)
#%%
#Partition large datsets into new datasets by site name
grouped = df.groupby(df['Site'])
grouped2 = df2.groupby(df2['site'])
dflist= []
gcclist=[]
for site in uni:
    dfg1 = grouped.get_group(site)
    dflist.append(dfg1)
    dfg2= grouped2.get_group(site)
    gcclist.append(dfg2)


#%%
#Quick way to line up datasets by getting a string of Year-DOY
cleandflist = []
for i,df in enumerate(gcclist):
    print(f'Attempting to clean DF{i+1}')
    df['YDOY'] = df['year'].astype(str)+ df['doy'].astype(str)
    print(f'DF{i+1} -- Complete')
    cleandflist.append(df.loc[:,['YDOY','midday_gcc','midday_rcc','gcc_50','gcc_75','gcc_90','rcc_50','rcc_75','rcc_90']])
for df in dflist:
    df['YDOY'] = df['YEAR'].astype(str)+ df['DOY'].astype(str)


#%%
#Merge on 'YDOY', maintain size of larger DF
#Copies rows of GCC data to match 
merge_list = []
for df, df1 in zip(cleandflist, dflist):
    #print(f'df shape: {df.shape}, df1 shape {df1.shape}')
    merge_df = df1.merge(df,how = 'left', on = 'YDOY')
    #print(f'merged shape: {merge_df.shape}')
    merge_list.append(merge_df)

# %%
#Concatenate to form one large dataframe
final_df = pd.concat(merge_list)

#%%
final_df.reset_index(inplace = True)
final_df.drop(['YDOY'], axis = 1, inplace = True)

final_df.drop(['index'], axis = 1, inplace = True)
#%%
#Free up space
del merge_list, cleandflist, gcclist, dflist, df, df2
#%%
#Columns to keep
keep_cols = ['TS_1_1_1',
 'TS_1_2_1',
'PPFD',
 'Tair',
 'VPD',
 'SWC_1_1_1',
 'HourCos',
 'HourSin',
 'DOYCos',
 'DOYSin',
 'FC50',
 'Site',
 'PPFD_OUT', 
'PPFD_BC_IN_1_1_1',
'Rh',
'NETRAD',
'SW_DIF',
'SW_IN_1_1_1',
'SW_OUT',
'Ustar',
 'midday_gcc',
 'midday_rcc',
 'gcc_50',
 'gcc_75',
 'gcc_90',
 'rcc_50',
 'rcc_75',
 'rcc_90'
 ]
#Create FC50 
data = final_df[final_df.Ustar >= final_df.U50]
final_df["FC50"] = data["FC"]

#Final df of kept columns
final_df = final_df.loc[:,keep_cols]
# %%
#Remove FC50 outliers
def outlier(data):
    data.dropna(inplace=True)
    x = sorted(data)
    q1, q3= np.percentile(x,[25,75])
    outside = 6
    iqr = q3 - q1
    lower_bound = q1 -(outside * iqr) 
    upper_bound = q3 +(outside * iqr) 
    return lower_bound,upper_bound


x = outlier(final_df["FC50"])
d = final_df[final_df["FC50"] <= x[1]]

d = d[d["FC50"] >= x[0]]

dpreds = d.drop(["FC50"], axis = 1)
dresponse = d["FC50"]


dpreds = dpreds.interpolate()

dat = dpreds.join(dresponse)

#Drop NA values
df2 = dat.dropna()
#%%
#Save to csv. Change file path
df2.to_csv(r"C:\Users\jeffu\Documents\Condensed43+GCC.csv")

# %%
