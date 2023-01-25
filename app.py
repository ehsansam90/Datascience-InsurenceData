#!/usr/bin/env python
# coding: utf-8

# In[122]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler


# In[208]:


df = pd.read_csv("insurance.csv")


# In[209]:


df.head()


# In[210]:


df['children'] = df['children'].astype("object")


# In[211]:


df.info()


# In[212]:


df.shape


# In[213]:


df.describe()


# In[214]:


df.smoker.value_counts()


# In[ ]:





# In[215]:


categorical = [column for column in df.columns if df[column].dtype == "O"]
numeric = [column for column in df.columns if column not in categorical]


# In[216]:


df.describe()


# In[217]:


sns.set()

for column in numeric:
    plt.figure(figsize=(7,7))
    sns.distplot(df[column])
    plt.title(column)
    plt.show()


# In[218]:


for column in categorical:
    plt.figure(figsize=(7,7))
    sns.countplot(x=column, data=df)
    plt.title(column)
    plt.show()


# In[220]:


df


# In[238]:


sns.heatmap(df.corr())


# In[222]:


sns.boxplot(x='sex', y='charges', data=df)


# In[223]:


sns.boxplot(x='smoker', y='charges', data=df)


# In[224]:


sns.boxplot(x='region', y='charges', data=df)


# In[225]:


sns.lmplot(x='bmi',y='charges',data=df,aspect=2,height=6)
plt.xlabel('BMI$(kg/m^2)$')
plt.ylabel('Insurance Charges')
plt.title('Charge Vs BMI')


# In[226]:


chargesMoreThan40000 = df[df['charges'] > 40000]


# In[241]:


sns.boxplot(x='sex', y='charges', data=chargesMoreThan40000)


# In[228]:


sns.boxplot(x='smoker', y='charges', data=chargesMoreThan40000)


# In[229]:


sns.boxplot(x='sex', y='charges', data=chargesMoreThan40000)


# In[230]:


df.replace({"sex":{"male":1,"female":2}, "smoker":{"yes":1,"no":0}},inplace=True)


# In[231]:


df = pd.concat([df, pd.get_dummies(df['region'])], axis=1).drop(['region'],axis=1)
df = pd.concat([df, pd.get_dummies(df['children'])], axis=1).drop(['children'],axis=1)


# In[232]:


df.head()


# In[233]:


df.info()


# In[234]:


#implement Standard scaler


# In[235]:


s = StandardScaler()


# In[236]:


s_df = s.fit_transform(df[['bmi','age']])


# In[237]:


s_df


# In[151]:


s_df = pd.DataFrame(s_df, columns=df[['bmi','age']].columns.values)


# In[153]:


df1 = df.drop(columns=['age','bmi'],axis=1)


# In[154]:


df1 = pd.concat([df1, s_df], axis=1)


# In[155]:


df1.head()


# In[156]:


X = df1.drop(['charges'],axis=1)
y = df1['charges']


# In[157]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=42) 


# In[158]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[159]:


lg = LinearRegression()


# In[160]:


ml1 = lg.fit(X_train, y_train)


# In[161]:


ml1.score(X_test,y_test)


# In[162]:


y_predlr1 = ml1.predict(X_test)


# In[163]:


msereg1 = mean_squared_error(y_predlr1, y_test)
r2reg1 = r2_score(y_predlr1, y_test)


# In[164]:


r2_score(y_predlr1, y_test)


# In[165]:


# implement min_max_scaler


# In[166]:


mm = MinMaxScaler()


# In[167]:


mm_df = mm.fit_transform(df[['bmi','age']])


# In[168]:


mm_df = pd.DataFrame(mm_df, columns=df[['bmi','age']].columns.values)


# In[169]:


df2 = df.drop(columns=['age','bmi'],axis=1)


# In[170]:


df2 = pd.concat([df2, mm_df], axis=1)


# In[171]:


df2.head()


# In[172]:


X = df2.drop(['charges'],axis=1)
y = df2['charges']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size = 0.2,random_state=42) 


# In[173]:


ml2 = lg.fit(X_train2,y_train2)


# In[174]:


ml2.score(X_test2,y_test2)


# In[175]:


y_predlr2 = ml2.predict(X_test2)


# In[176]:


msereg2 = mean_squared_error(y_predlr2, y_test2)
r2reg2 = r2_score(y_predlr2, y_test2)


# In[177]:


#implement SVM
from sklearn.tree import DecisionTreeRegressor


# In[178]:


tr = DecisionTreeRegressor()


# In[179]:


tr.fit(X_train, y_train)


# In[180]:


y_preddt1 = tr.predict(X_test)


# In[181]:


msedt1 = mean_squared_error(y_preddt1, y_test)
r2dt1 = r2_score(y_preddt1, y_test)


# In[182]:


tr.fit(X_train2, y_train2)


# In[183]:


y_preddt2 = tr.predict(X_test2)


# In[184]:


msedt2 = mean_squared_error(y_preddt2, y_test2)
r2dt2 = r2_score(y_preddt2, y_test2)


# In[185]:


from sklearn.linear_model import SGDRegressor


# In[186]:


sgdr = SGDRegressor() 


# In[187]:


sgdr.fit(X_train, y_train)


# In[188]:


y_predsgd1 = sgdr.predict(X_test)


# In[189]:


msesgd1 = mean_squared_error(y_predsgd1, y_test)
r2sgd1 = r2_score(y_predsgd1, y_test)


# In[190]:


sgdr.fit(X_train2, y_train2)


# In[191]:


y_predsgd2 = sgdr.predict(X_test2)
msesgd2 = mean_squared_error(y_predsgd2, y_test2)
r2sgd2 = r2_score(y_predsgd2, y_test2)


# In[192]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor


# In[193]:


reg1 = GradientBoostingRegressor(random_state=1)
reg2 = RandomForestRegressor(random_state=1)
reg3 = LinearRegression()
ereg = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])
ereg = ereg.fit(X_train, y_train)


# In[194]:


y_predereg1 = ereg.predict(X_test)
mseereg1 = mean_squared_error(y_predereg1, y_test)
r2ereg1 = r2_score(y_predereg1, y_test)


# In[195]:


ereg = ereg.fit(X_train2, y_train2)


# In[196]:


y_predereg2 = ereg.predict(X_test)
mseereg2 = mean_squared_error(y_predereg2, y_test2)
r2ereg2 = r2_score(y_predereg2, y_test2)


# In[197]:


results = pd.DataFrame({"MSE_standardScaled": [msereg1,msesgd1,msedt1,mseereg1], "R2_standardScaled":[r2reg1,r2sgd1,r2dt1,r2ereg1], 
                         "MSE_MinMaxScaled":[msereg2,msesgd2,msedt2,mseereg2],"R2_MinMaxScaled":[r2reg2,r2sgd2,r2dt2,r2ereg2] }
                     ,index = ['LinearRegression','SGDRegressor','DecisionTreeRegressor','VotingRegressor'])    


# In[203]:


results = results.T


# In[204]:


results


# In[246]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.lineplot(data=results[['MSE_standardScaled','MSE_MinMaxScaled']])
plt.ylabel('MSE')


# In[247]:


sns.lineplot(data=results[['R2_standardScaled','R2_MinMaxScaled']])
plt.ylabel('R2 Score')


# In[253]:


dict_models ={"DecissionTreeRegressor - Standard Scaled":y_preddt1,
              "DecissionTreeRegressor - MinMax Scaled":y_preddt2,
              "VotingRegressor - Standard Scaled":y_predereg1,
              "VotingRegressor - MinMax Scaled":y_preddt2,
              "LinearRegression - Standard Scaled":y_predlr1,
              "Linearregression - MinMax Scaled":y_predlr2,
              "SGDregression - Standard Scaled":y_predsgd1,
              "SGD regression MinMax Scaled":y_predsgd2}


# In[258]:


for pred in ["DecissionTreeRegressor - Standard Scaled"
             ,"DecissionTreeRegressor - MinMax Scaled",
             "VotingRegressor - Standard Scaled",
             "VotingRegressor - MinMax Scaled",
             "LinearRegression - Standard Scaled",
             "Linearregression - MinMax Scaled",
             "SGDregression - Standard Scaled",
             "SGD regression MinMax Scaled"]:
    sns.residplot(dict_models[pred], y_test2)
    plt.xlabel('Predicted values for {}'.format(pred), fontsize=15)
    plt.ylabel('Residuals')
    plt.show()


# In[ ]:




