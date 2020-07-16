#!/usr/bin/env python
# coding: utf-8

# In[1]:


import turicreate as tc


# # Regression using house price data

# In[2]:


#reading data
sf=tc.SFrame('home_data2.sframe')


# In[3]:


sf


# # Visualization

# In[158]:


#general overall summary of house
sf.show()


# In[159]:


#2d plot
tc.show(sf[1:5000]['bedrooms'],sf[1:5000]['price'])


# # Regression starts here

# In[160]:


train_set, test_set = sf.random_split(.7,seed=0)


#  ### First, we build a simple linear model for predicting house price

# In[161]:


model1 = tc.linear_regression.create(train_set,target='price',features=['sqft_living'])


# #### Evaluation of model

# In[162]:


print (model1.evaluate(test_set))


# In[163]:


print (model1.predict(test_set))


# In[164]:


a1=tc.SArray(data=model1.predict(test_set))
a1=a1.astype(dtype=int)


# In[165]:


a2=tc.SArray(test_set['price'])


# In[166]:


import pandas as pd


# In[167]:


data = {'predicted value':a1, 'real value':a2} 


# In[168]:


z=tc.SFrame(pd.DataFrame(data))


# In[169]:


z


# ### Finding Coefficients and plots

# In[170]:


model1.coefficients


# In[171]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(test_set['sqft_living'],test_set['price'],'.',
        test_set['sqft_living'],model1.predict(test_set),'-')


# ## Application of Multiple independent features(Multiple Regression)

# In[172]:


x=['sqft_living','sqft_above','sqft_lot15','bedrooms','bathrooms','zipcode','sqft_lot','floors','sqft_living15','sqft_basement','condition','view','waterfront']


# In[173]:


sf[x].show()


# In[174]:


tc.show(sf['zipcode'],sf['price'])


# In[175]:


model2 = tc.linear_regression.create(train_set,target='price',features=x)


# ### Comparison of both models

# In[176]:


print (model1.evaluate(test_set))
print (model2.evaluate(test_set))


# ### Evaluation of performance of model2

# In[178]:


p2=model2.predict(test_set)
p2


# In[179]:


a3=tc.SArray(data=model2.predict(test_set))
a3=a3.astype(dtype=int)


# In[180]:


data2 = {'predicted value':a3, 'real value':a2}


# In[181]:


z2=tc.SFrame(pd.DataFrame(data2))


# In[182]:


z2


# ## Prediction of a particular house

# In[206]:


h1=sf[sf['id']=='1321400060']


# In[207]:


h1


# In[203]:


print (h1['price'])


# In[204]:


print (model1.predict(h1))


# In[205]:


print (model2.predict(h1))


# <img src=house-2008000270.png>

# #### Finding mean house price of a particular neighbourhood

# In[127]:


z=sf[sf['zipcode']==98039]


# In[128]:


z


# In[129]:


z['price'].mean()


# ### Finding fraction of houses of square feet of living space >=2000 sqft but <4000 sqft

# In[87]:


z=sf[sf['sqft_living']>2000]


# In[88]:


z=z[z['sqft_living']<4000]


# In[89]:


z


# In[90]:


z=z.add_row_number(column_name='No.', start=1)


# In[96]:


f1=z['No.'].tail(1)


# In[93]:


sf=sf.add_row_number(column_name='No.', start=1)


# In[97]:


f2=sf['No.'].tail(1)


# In[98]:


f=f1/f2


# In[99]:


f


# ### Finding predictions with new set of Features

# In[6]:


my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']


# In[7]:


advanced_features = [
'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house
'grade', # measure of quality of construction
'waterfront', # waterfront property
'view', # type of view
'sqft_above', # square feet above ground
'sqft_basement', # square feet in basement
'yr_built', # the year built
'yr_renovated', # the year renovated
'lat', 'long', # the lat-long of the parcel
'sqft_living15', # average sq.ft. of 15 nearest neighbors 
'sqft_lot15', # average lot size of 15 nearest neighbors 
]


# In[8]:


train_data, test_data = sf.random_split(.8,seed=0)


# In[9]:


model3 = tc.linear_regression.create(train_data,target='price',features=my_features,validation_set=None)


# In[10]:


model4 = tc.linear_regression.create(train_data,target='price',features=advanced_features,validation_set=None)


# #### Find difference in errors amongst two models

# In[11]:


e1=model3.evaluate(test_data)
e1


# In[12]:


e2=model4.evaluate(test_data)
e2


# In[13]:


e=e1['rmse']-e2['rmse']


# In[14]:


e

