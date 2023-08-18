#%%
# Read in CSV of all Neon site data
import numpy as np
import pandas as pd
#Read in Site data csv
df = pd.read_csv(r"C:\Users\jeffu\Documents\LE+LW 2.csv\LE+LW.csv", header = 0, low_memory=False)
uni = list(np.unique(df['Site']))
#Read in GCC csv
dfgcc = pd.read_csv(r"C:\Users\jeffu\OneDrive\Documents\43GCCData.csv", header =0, low_memory=False)
dfndvi = pd.read_csv(r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Summer 2023 Research\NDVI.csv")
dfprcp = pd.read_csv(r"C:\Users\jeffu\OneDrive\Documents\Jeff's Math\Summer 2023 Research\prcp.csv")
#%%
keep_cols = ['YEAR','DOY', 'HOUR', 'LW_OUT','U50',
'TS_1_1_1',
 'TS_1_2_1',
'PPFD',
 'Tair',
 'VPD',
 'SWC_1_1_1',
 'FC',
 'LE',
 'Site',
 'PPFD_OUT', 
'PPFD_BC_IN_1_1_1',
'Rh',
'NETRAD',
'Ustar']
df = df.loc[:,keep_cols]
#Uncomment to see list of site names
#print(uni)
#%%
extradf = df
#%%
df = extradf
#Partition large datsets into new datasets by site name
grouped = df.groupby(df['Site'])
grouped2 = dfgcc.groupby(dfgcc['site'])
grouped3 = dfndvi.groupby(dfndvi['Site'])
grouped4 = dfprcp.groupby(dfprcp['Site'])
dflist= []
gcclist=[]
ndvilist = []
prcplist = []
for site in uni:
    dfg1 = grouped.get_group(site)
    dflist.append(dfg1)
    dfg2= grouped2.get_group(site)
    gcclist.append(dfg2)
    dfg3=grouped3.get_group(site)
    ndvilist.append(dfg3)
    dfg4 = grouped4.get_group(site)
    prcplist.append(dfg4)


#%%
#Quick way to line up datasets by getting a string of Year-DOY
cleandflist = []
for i,df in enumerate(gcclist):
    df.loc[:,'YDOY'] = df.loc[:,'year'].astype(str)+ df.loc[:,'doy'].astype(str)
    cleandflist.append(df.loc[:,['YDOY','gcc_50','rcc_50']])
for df in dflist:
    df.loc[:,'YDOY'] = df.loc[:,'YEAR'].astype(str)+ df.loc[:,'DOY'].astype(str)

for df in ndvilist:
    df.loc[:,'YDOY'] = df.loc[:,'YEAR'].astype(str) + df.loc[:,'DOY'].astype(str)
for df in prcplist:
    df.loc[:,'YDOY'] = df.loc[:,'YEAR'].astype(str) + df.loc[:,'DOY'].astype(str)
#%%
#Merge on 'YDOY', maintain size of larger DF
#Copies rows of GCC data to match 
merge_list = []
for df, df1 in zip(cleandflist, dflist):
    #print(f'df shape: {df.shape}, df1 shape {df1.shape}')
    merge_df = df1.merge(df,how = 'left', on = 'YDOY')
    #merge_df = merge_df.merge(df2, how = 'left', on='YDOY')
    #print(f'merged shape: {merge_df.shape}')
    merge_list.append(merge_df)

# %%
merge_list_2 = []
for df, df1, df2 in zip(merge_list,ndvilist,prcplist):
    df1=df1.drop(['YEAR','DOY','Site'],axis=1)
    merge_df = df.merge(df1, how = 'left', on = 'YDOY')
    
    df2 = df2.drop(['YEAR','DOY','Site'],axis=1)
    merge_df = merge_df.merge(df2, how='left', on = 'YDOY')
    merge_list_2.append(merge_df)
#%%
#Concatenate to form one large dataframe
final_df = pd.concat(merge_list_2)

#%%
final_df.reset_index(inplace = True)
final_df.drop(['YDOY','YEAR'], axis = 1, inplace = True)

final_df.drop(['index'], axis = 1, inplace = True)
#%%
#Free up space
del merge_list, cleandflist, gcclist, dflist, df
#%%
#Columns to keep

#Create FC50 and LE50
data = final_df[final_df.Ustar >= final_df.U50]

final_df["FC50"] = data["FC"]
final_df["LE50"] = data["LE"]
#%%
#Final df of kept columns
final_df = final_df.drop(['FC','LE','U50'],axis=1)
#%%
#Add in phenocam site metadata 
new_cols = pd.read_csv(r"C:\Users\jeffu\OneDrive\Documents\Newcolumns.csv")
new_cols.rename(columns={'flux_sitenames':'Site'},inplace=True)
new_cols.drop(['site_code'],axis=1,inplace=True)
#Replace Secondary Veg NA with primary
newer = new_cols.secondary_veg_type.fillna(new_cols.primary_veg_type)
new_cols.secondary_veg_type = newer
#%%
#Leftmerge final with phenocam metadata
final_df = final_df.merge(new_cols, how='left', on='Site')

#Scale DOY and HOUR
final_df.loc[:,'DOY'] /= 366
final_df.loc[:,'HOUR'] /= 24

#%%
final_df.loc[:,['EVI','NDVI']] = final_df.loc[:,['EVI','NDVI']].ffill()
#%%
final_df = final_df[final_df.EVI!=-3000.0]
#%%
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
dFC = final_df[final_df["FC50"] <= x[1]]

dFC = dFC[dFC["FC50"] >= x[0]]

dFC.drop(['LW_OUT'],axis=1,inplace=True)

#Remove LE50 outliers separately 
x = outlier(final_df["LE50"])
dLE= final_df[final_df["LE50"] <= x[1]]

dLE = dLE[dLE["LE50"] >= x[0]]
#%%
#Interpolate and create FC50 Dataset
dpreds = dFC.drop(["FC50","LE50"], axis = 1)
dresponse = dFC["FC50"]
dpreds = dpreds.interpolate()

dat = dpreds.join(dresponse)
dat= dat.join(pd.get_dummies(dat.primary_veg_type, prefix='PVeg'))
dat = dat.join(pd.get_dummies(dat.secondary_veg_type,prefix='SVeg'))
dat = dat.join(pd.get_dummies(dat.domain, prefix = 'domain'))
dat.drop(['primary_veg_type','secondary_veg_type','domain'],axis=1,inplace=True)
dfFC50 = dat.dropna()


#Interpolate and create LE50 Dataset
dpreds = dLE.drop(["FC50","LE50"], axis = 1)
dresponse = dLE["LE50"]
dpreds = dpreds.interpolate()

dat = dpreds.join(dresponse)
dat= dat.join(pd.get_dummies(dat.primary_veg_type, prefix='PVeg'))
dat = dat.join(pd.get_dummies(dat.secondary_veg_type,prefix='SVeg'))
dat = dat.join(pd.get_dummies(dat.domain, prefix = 'domain'))
dat.drop(['primary_veg_type','secondary_veg_type','domain'],axis=1,inplace=True)
dfLE50 = dat.dropna()

#%%
#Save to csv. Change file path
dfFC50.to_csv(r"C:\Users\jeffu\Documents\FC50DataVer4.csv",index=False)
dfLE50.to_csv(r"C:\Users\jeffu\Documents\LE50DataVer4.csv",index=False)
# %%

