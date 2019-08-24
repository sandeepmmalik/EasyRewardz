
# coding: utf-8

# # Easy Rewardz: A study of sales through consumer behaviours

# <h3>Column Description</h3>
# 
# <table>
#     <tr>
#         <th> Column Name </th> <th> Description </th>
#     </tr>
#     <tr> <td>ModifiedStore</td> <td>	Store Name </td>
#     </tr>
#     <tr> <td> ModifiedStoreCode </td> <td>		Store Code</td>
#     </tr>
#     <tr> <td> ModifiedTxnDate </td> <td>		Txn Date</td>
#     </tr>
#     <tr> <td> ModifiedTxnTime </td> <td>		Txn Time </td>
#     </tr>
#     <tr> <td> ModifiedBillNo </td> <td>		Bill No</td>
#     </tr>
#     <tr> <td> UniqueItemName </td> <td>		Item Name</td>
#     </tr>
#     <tr> <td> UniqueItemCode </td> <td>		Item Code </td>
#     </tr>
#     <tr> <td> Barcode </td> <td>		Barcode</td>
#     </tr>
#     <tr> <td> ItemQty </td> <td>		Item Qty</td>
#     </tr>
#     <tr> <td> ItemMRP </td> <td>		Item MRP</td>
#     </tr>
#     <tr> <td> ItemGrossAmount </td> <td>		Item Gross Amount</td>
#     </tr>
#     <tr> <td> ItemNetAmount </td> <td>	ItemNetAmount</td>
#     </tr>
#     <tr> <td> ItemDiscountAmount </td> <td>	ItemDiscountAmount</td>
#     </tr>
#     <tr> <td> Customer Id </td> <td>	Customer Id</td>
#     </tr>
#     
#     
# </table>
# 
# 
# 

# <H3> Analysis Points (Use Cases)</H3>
# 
# <UL>
#     <LI> Growth Trend </LI>
#     <UL>
#         <LI> New Customers Acquired </LI>
#         <LI> Total Transacting Customers </LI>
#         <LI> Repeat Customers </LI>
#         <LI> Bills </LI>
#         <LI> Sales </LI>
#         <LI> Product Brought </LI>
#         <LI> Avg. Bill Value </LI>        
#         <LI> Avg. Products in each Bill </LI>
#         <LI> Avg. Single Product Price </LI>
#     </UL>       
#     <LI> Customer Fallout Rate </LI>
#     <LI> Average Gap between 2 successive visits for Repeat Customers </LI>
#     <LI> Product Affinity, Category Popularity, Weekday Vs Weekend Trend </LI>    
# </UL>    

# In[344]:

# Import various modules

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib as mlp
import matplotlib.pyplot as plt
import datetime as dt 

# import, instantiate, fit
from sklearn.linear_model import LinearRegression

#import train test split from sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[345]:

# Read data 

file_loc = "D:/DataScience/ProjectData/EasyRewardz/"
file_name = "ER Excel Case Study_October 2018.csv"

bf_data = pd.read_csv(file_loc+file_name, sep=',')

file_name = "item master.xlsx"

item_master = pd.read_excel(file_loc+file_name)


# In[346]:

# summary of the data (mean, STD, Min, Max)
bf_data.describe()


# In[347]:

item_master.describe()


# In[348]:

bf_data.columns


# In[349]:

item_master.columns


# In[350]:

# how data is distributed 
bf_data.info()


# In[351]:

item_master.info()


# In[352]:

#top 5 rows from dataset
bf_data.head()


# In[353]:

item_master.head()


# In[354]:

#bottom 5 rows from dataset
bf_data.tail()


# In[355]:

item_master.tail()


# In[356]:

# Print Unique Store Names and count number of occurances for each

print('Unique Store Names: ', bf_data['ModifiedStore'].unique(), bf_data.groupby('ModifiedStore').size())
print('-----------------------------------------------------------------------------------')
print('Unique Stores Count: ', bf_data['ModifiedStore'].nunique())


# In[357]:

customer_store=bf_data[['ModifiedStore','Customer Id']].drop_duplicates()
customer_store
customer_store.groupby(['ModifiedStore'])['Customer Id'].aggregate('count').reset_index().sort_values('Customer Id', ascending=False)


# In[358]:

# Count orders with negative value in ItemQty col

bf_data[(bf_data['ItemQty'] < 0)].ItemQty.count() 


# In[359]:

NOW = dt.datetime(2018,10,1)


# In[360]:

bf_data['InvoiceDate'] = pd.to_datetime(bf_data['ModifiedTxnDate'])


# In[361]:

bf_data.dtypes


# <h2>RFM Customer Segmentation</h2>

# In[362]:

bf_data.head()


# In[363]:

# Create RFM Table

rfmTable = bf_data.groupby(['Customer Id','ModifiedStore']).agg({'InvoiceDate': lambda x: (NOW - x.max()), 'ModifiedBillNo': lambda x: len(x), 'ItemNetAmount': lambda x: x.sum()})
#rfmTable['InvoiceDate'] = rfmTable['InvoiceDate'].astype(int)
rfmTable.rename(columns={'InvoiceDate': 'recency', 
                         'ModifiedBillNo': 'frequency', 
                         'ItemNetAmount': 'monetary_value'}, inplace=True)


# In[364]:

rfmTable.head()


# In[365]:

rfmTable.dtypes

rfmTable['recency'] = (rfmTable['recency']/ np.timedelta64(1, 'D')).astype(int)


# In[366]:

rfmTable.head()


# In[367]:

# We will use the 80% quantile for each feature
#quantiles = rfmTable.quantile(q=[0.8])
quantiles = rfmTable.quantile(q=[0.25,0.5,0.75], axis=0)
quantiles = quantiles.to_dict()

segmented_rfm = rfmTable

print(quantiles)
segmented_rfm.head()


# In[368]:

def RScore(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
    
def FMScore(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1


# In[369]:

segmented_rfm['r_quartile'] = segmented_rfm['recency'].apply(RScore, args=('recency',quantiles,))
segmented_rfm['f_quartile'] = segmented_rfm['frequency'].apply(FMScore, args=('frequency',quantiles,))
segmented_rfm['m_quartile'] = segmented_rfm['monetary_value'].apply(FMScore, args=('monetary_value',quantiles,))
segmented_rfm.head()


# In[370]:

segmented_rfm['RFMScore'] = segmented_rfm.r_quartile.map(str) + segmented_rfm.f_quartile.map(str) + segmented_rfm.m_quartile.map(str)
segmented_rfm.head()


# In[371]:

# top 10 best customers

segmented_rfm[segmented_rfm['RFMScore']=='111'].sort_values('monetary_value', ascending=False).head(10)


# In[372]:

New_Cust = bf_data.groupby(['Customer Id', 'ModifiedStore']).agg({'ModifiedTxnDate': lambda x: x.min(),'InvoiceDate': lambda x: (NOW - x.max()), 'ModifiedBillNo': lambda x: len(x), 'ItemNetAmount': lambda x: x.sum()})


# In[373]:

New_Cust.head()


# In[374]:

New_Cust.rename(columns={'ModifiedTxnDate':'FirstTransDate',
                         'InvoiceDate': 'recency', 
                         'ModifiedBillNo': 'frequency', 
                         'ItemNetAmount': 'monetary_value'}, inplace=True)


# In[375]:

New_Cust.head()


# In[376]:

New_Cust.dtypes
New_Cust['FirstTransDate'] = pd.to_datetime(New_Cust['FirstTransDate'])
New_Cust['recency'] = (New_Cust['recency']/ np.timedelta64(1, 'D')).astype(int)


# In[377]:

New_Cust = New_Cust[New_Cust['frequency'] >= 1]
#New_Cust = New_Cust[New_Cust['recency'] < 90]
New_Cust = New_Cust[New_Cust['FirstTransDate'] >= (NOW- dt.timedelta(days=90))] 
New_Cust.head()


# In[378]:

# Customer count for Customers who did their first transaction within last 90 days
New_Cust.count()


# In[379]:

# Customers with Transaction within last 90 days

Trans_Cust = bf_data.groupby(['Customer Id', 'ModifiedStore']).agg({'ModifiedTxnDate': lambda x: x.max(),'InvoiceDate': lambda x: (NOW - x.max()), 'ModifiedBillNo': lambda x: len(x), 'ItemNetAmount': lambda x: x.sum()})

Trans_Cust.rename(columns={'ModifiedTxnDate':'LastTransDate',
                         'InvoiceDate': 'recency', 
                         'ModifiedBillNo': 'frequency', 
                         'ItemNetAmount': 'monetary_value'}, inplace=True)

Trans_Cust['LastTransDate'] = pd.to_datetime(Trans_Cust['LastTransDate'])

Trans_Cust = Trans_Cust[Trans_Cust['LastTransDate'] >= (NOW- dt.timedelta(days=90))] 


# In[380]:

Trans_Cust.head()


# In[381]:

# Customer count for Customers who did their last transaction within last 90 days
Trans_Cust.count()


# In[389]:

Avg_Bill = bf_data.groupby(['ModifiedStore']).agg({'InvoiceDate': lambda x: (NOW - x.max()), 'ModifiedBillNo': lambda x: len(x), 'ItemNetAmount': lambda x: x.mean()})
Avg_Bill.sort_values('ItemNetAmount', ascending=False).head(10)


# In[390]:

bf_data['ItemNetAmount'].mean()


# <h3> Visualizing the Data on various parameters </h3>

# In[393]:

get_ipython().magic('matplotlib inline')


# In[394]:

plt.figure(figsize=(20,15))
sns.countplot(segmented_rfm['RFMScore'])


# In[ ]:




# In[ ]:

# Calculate RFM score and sort customers
# To do the 2 x 2 matrix we will only use Recency & Monetary
df_RFM = segmented_rfm
df_RFM.head()


# In[ ]:

df_RFM['RMScore'] = segmented_rfm.m_quartile.map(str)+segmented_rfm.r_quartile.map(str)
df_RFM = df_RFM.reset_index()
df_RFM.head()


# In[ ]:

df_RFM_SUM = df_RFM.groupby('RMScore').agg({'Customer Id': lambda y: len(y.unique()),
                                        'frequency': lambda y: round(y.mean(),0),
                                        'recency': lambda y: round(y.mean(),0),
                                        'r_quartile': lambda y: round(y.mean(),0),
                                        'm_quartile': lambda y: round(y.mean(),0),
                                        'monetary_value': lambda y: round(y.mean(),0)})


# In[ ]:

df_RFM_SUM.head()


# In[ ]:

df_RFM_SUM = df_RFM_SUM.reset_index()

df_RFM_SUM = df_RFM.groupby('RMScore').agg({'Customer Id': lambda y: len(y.unique()),
                                        'frequency': lambda y: round(y.mean(),0),
                                        'recency': lambda y: round(y.mean(),0),
                                        'r_quartile': lambda y: round(y.mean(),0),
                                        'm_quartile': lambda y: round(y.mean(),0),
                                        'monetary_value': lambda y: round(y.mean(),0)})

df_RFM_SUM.head()


# In[ ]:

#
df_RFM_SUM = df_RFM_SUM.reset_index()
df_RFM_SUM = df_RFM_SUM.sort_values('RMScore', ascending=False)

# Visualize the Value Matrix and explore some key numbers
# 1) Average Monetary Matrix
df_RFM_M = df_RFM_SUM.pivot(index='m_quartile', columns='r_quartile', values='monetary_value')
df_RFM_M= df_RFM_M.reset_index().sort_values(['m_quartile'], ascending = False).set_index(['m_quartile'])
print(df_RFM_M)
# 2) Number of Customer Matrix
df_RFM_C = df_RFM_SUM.pivot(index='m_quartile', columns='r_quartile', values='Customer Id')
df_RFM_C= df_RFM_C.reset_index().sort_values(['m_quartile'], ascending = False).set_index(['m_quartile'])
print(df_RFM_C)
# 3) Recency Matrix
df_RFM_R = df_RFM_SUM.pivot(index='m_quartile', columns='r_quartile', values='recency')
df_RFM_R= df_RFM_R.reset_index().sort_values(['m_quartile'], ascending = False).set_index(['m_quartile'])
print(df_RFM_R)

