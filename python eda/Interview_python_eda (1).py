#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
from time import strftime
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


# In[2]:


# Reading the dataset which i've taken from kaggle as a Dataframe 

base_data = pd.read_csv('Data.csv')


# In[6]:


base_data.head()


# In[9]:


base_data.shape


# In[11]:


base_data.info()


# In[12]:


#modifying the date and time from object type to date time format
base_data['ScheduledDay'] = pd.to_datetime(base_data['ScheduledDay']).dt.date.astype('datetime64[ns]')
base_data['AppointmentDay'] = pd.to_datetime(base_data['AppointmentDay']).dt.date.astype('datetime64[ns]')


# In[15]:


base_data.info()


# In[16]:


# 5 is Saturday, 6 is Sunday 

base_data['sch_weekday'] = base_data['ScheduledDay'].dt.dayofweek


# In[17]:


base_data['app_weekday'] = base_data['AppointmentDay'].dt.dayofweek


# In[20]:


base_data['sch_weekday'].value_counts()
# Clearly implies that there are no scheduled dates on a Sunday


# In[21]:


base_data['app_weekday'].value_counts()
#Similarly there are no appointment dates on Sundays


# In[22]:


base_data.columns


# In[23]:


#changing the name of some columns to clear them of typo errors
base_data= base_data.rename(columns={'Hipertension': 'Hypertension', 'Handcap': 'Handicap', 'SMS_received': 'SMSReceived', 'No-show': 'NoShow'})


# In[24]:


base_data.columns


# In[25]:


base_data.info()


# In[26]:


# dropping some columns which have no significance
base_data.drop(['PatientId', 'AppointmentID', 'Neighbourhood'], axis=1, inplace=True)


# In[28]:


#denotion of only numerical values and not categorical values

base_data.describe()


# In[30]:


#utilising matplotlib for data visualization of target variable:No Show
base_data['NoShow'].value_counts().plot(kind='barh', figsize=(8, 6))
plt.xlabel("Count", labelpad=14)
plt.ylabel("Target Variable", labelpad=14)
plt.title("Count of TARGET Variable per category", y=1.02);


# In[31]:


# calculating the % of appointments or not 
100*base_data['NoShow'].value_counts()/len(base_data['NoShow'])


# In[32]:


#The inference from above plot is that the data is highly imbalanced towards no show:No


# In[36]:


# Having a look that data contains missing values or not
missing = pd.DataFrame((base_data.isnull().sum())*100/base_data.shape[0]).reset_index()
plt.figure(figsize=(16,5))
ax = sns.pointplot('index',0,data=missing)
plt.xticks(rotation =90,fontsize =7)
plt.title("Percentage of Missing values")
plt.ylabel("PERCENTAGE")
plt.show()


# In[38]:


#The initial intuition is that we don't have any missing data.


# In[41]:


"""General Thumb Rules:
For features with less missing values- can use regression to predict the missing values or fill with the mean of the values present, depending on the feature.
For features with very high number of missing values- it is better to drop those columns as they give very less insight on analysis.
As there's no thumb rule on what criteria do we delete the columns with high number of missing values, but generally you can delete the columns, if you have more than 30-40% of missing values.
"""


# In[42]:


#DATA CLEANING


# In[43]:


new_data=base_data.copy()


# In[44]:


new_data.info()


# In[46]:


#As we don't have any null records, there's no data cleaning required


# In[45]:


# Get the max tenure of age
print(base_data['Age'].max())


# In[47]:


# Group the tenure in bins of 12 months to make the data look more meaningful and organized
labels = ["{0} - {1}".format(i, i + 20) for i in range(1, 118, 20)]

base_data['Age_group'] = pd.cut(base_data.Age, range(1, 130, 20), right=False, labels=labels)


# In[48]:


base_data.drop(['Age'], axis=1, inplace=True)


# In[49]:


#Data Exploration


# In[50]:


list(base_data.columns)


# In[66]:


#having a see-through into the values of count of each column and there count with respect to No-Show column
for i, predictor in enumerate(base_data.drop(columns=['NoShow'])):
    print('-'*10,predictor,'-'*10)
    print(base_data[predictor].value_counts())    
    plt.figure(i)
    sns.countplot(data=base_data, x=predictor, hue='NoShow')


# In[84]:


#predictor--column of base_data dataset
#base_data['NoShow'] = np.where(base_data.NoShow == 'Yes',1,0)


# In[85]:


#base_data.NoShow.value_counts()


# In[59]:


#Convert all the categorical variables into dummy variables
base_data_dummies = pd.get_dummies(base_data)
base_data_dummies.head()


# In[60]:


#Build a corelation of all predictors with 'NoShow'


# In[83]:


"""plt.figure(figsize=(20,8))
base_data_dummies.corr()['NoShow'].sort_values(ascending = False).plot(kind='bar')"""


# In[68]:


plt.figure(figsize=(12,12))
sns.heatmap(base_data_dummies.corr(), cmap="Paired")


# In[74]:


#BIVARIATE ANALYSIS


# In[75]:


new_df1_target0=base_data.loc[base_data["NoShow"]==0]
new_df1_target1=base_data.loc[base_data["NoShow"]==1]


# In[2]:


def uniplot(df,col,title,hue =None):
    
    sns.set_style('whitegrid')
    sns.set_context('talk')
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.titlepad'] = 30
    
    
    temp = pd.Series(data = hue)
    fig, ax = plt.subplots()
    width = len(df[col].unique()) + 7 + 4*len(temp.unique())
    fig.set_size_inches(width , 8)
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.title(title)
    ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue,palette='bright') 
        
    plt.show()


# In[82]:


#uniplot(new_df1_target1,col='Hypertension',title='Distribution of Gender for NoShow Customers',hue='Gender')


# In[81]:


uniplot(new_df1_target0,col='Hypertension',title='Distribution of Gender for NoShow Customers',hue='Gender')


# In[87]:


#uniplot(new_df1_target1,col='Age_group',title='Distribution of Age for NoShow Customers',hue='Gender')


# In[88]:


uniplot(new_df1_target0,col='Age_group',title='Distribution of Age for NoShow Customers',hue='Gender')


# 
# Findings
# 1.Female patients have taken more appointments then male patients
# 2.Ratio of Nohow and Show is almost equal for age group except Age 0 and Age 1 with 80% show rate for each age group
# 3.Each Neighbourhood have almost 80% show rate
# 4.There are 99666 patients without Scholarship and out of them around 80% have come for the visit and out of the 21801 patients with Scholarship around 75% of them have come for the visit.
# 5.There are around 88,726 patients without Hypertension and out of them around 78% have come for the visit and Out of the 21801 patients with Hypertension around 85% of them have come for the visit.
# 6.There are around 102,584 patients without Diabetes and out of them around 80% have come for the visit and Out of the 7,943 patients with Diabetes around 83% of them have come for the visit.
# 7.There are around 75,045 patients who have not received SMS and out of them around 84% have come for the visit and out of the 35,482 patients who have received SMS around 72% of them have come for the visit.
# 8.There is no appointments on sunday and on saturday appointments are very less in comparision to other week days
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




