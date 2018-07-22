
# coding: utf-8

# In[1]:


import matplotlib.pyplot as py
import seaborn as sb
import pandas as pd
import numpy as np


# In[55]:


df = sb.load_dataset('tips')


# In[56]:


df.to_csv('tips1.csv')
df.head(20)


# In[57]:


df.info()


# In[58]:


df.describe()


# In[59]:


df.sample(5)


# In[60]:


df.groupby('day').count()


# In[61]:


df2 = df.groupby('day').sum()
df2


# In[62]:


df2.drop('size',inplace = True,axis = 1)


# In[63]:


df2['percent'] = df2['tip']/df2['total_bill'] * 100


# In[64]:


df2.to_csv('df2.csv')
df2


# In[65]:


df3 = df.groupby('smoker').sum()
df3


# In[66]:


df3['percent'] = df3['tip']/df3['total_bill'] * 100
df3.to_csv('df3.csv')
df3


# In[67]:


df4 = df.groupby(['day','size']).sum()
print(df4)
df4['percent'] = df4['tip']/df4['total_bill'] * 100
df4.to_csv('df4.csv')
df4.dropna()


# In[68]:


sb.countplot(x = 'day',data = df)
py.show()


# In[69]:


sb.countplot(x = 'day',hue = 'size',data = df)


# In[70]:


df3


# In[71]:


sb.countplot(x = 'percent',hue = 'total_bill',data = df3)


# In[75]:


df.replace({'sex':{'Male':0,'Female' : 1},'smoker':{'No':0,'Yes':1}},inplace= True)
df.to_csv('df_replace.csv')
df


# In[76]:


days = pd.get_dummies(df['day'])
days.sample(5)


# In[77]:


days = pd.get_dummies(df['day'],drop_first = True)
days.sample(5)
times = pd.get_dummies(df['time'],drop_first = True)
times.sample(5)


# In[78]:


days = pd.get_dummies(df['day'],drop_first = True)
df =pd.concat([df,days],axis = 1)


# In[87]:


df.head()


# In[94]:


df = pd.concat([df,times],axis = 1)
df.head()


# In[82]:


df.drop(['day','time'],inplace = True,axis = 1)


# In[83]:


df.to_csv('df_drop.csv')


# In[85]:


df.head()


# In[92]:


df.drop(['Dinner'],inplace = True,axis = 1)


# In[95]:


df.head()


# In[97]:


x = df[['sex','smoker','size','Fri','Sat','Sun','Dinner']]
Y = df[['tip']]


# In[107]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
x_train,x_test,y_train,y_test = train_test_split(x,Y,test_size = 0.25,random_state = 20)


# In[119]:


model = LinearRegression()
model.fit(x_train,y_train)


# In[109]:


predict = model.predict(x_test)


# In[118]:


sb.distplot(y_test-predict)


# In[120]:


my_vals = np.array([0,1,3,1,0,0,0]).reshape(1,-1)
print(my_vals)
my_vals1 = np.array([0,1,3,1,0,0,0])
print(my_vals1)


# In[121]:


model.predict(my_vals)

