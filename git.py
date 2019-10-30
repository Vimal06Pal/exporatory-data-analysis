#importing libraries

import numpy as npS
import pandas as pd
import matplotlib.pyplot as plt
import  seaborn as sns

#Reading the file
df=pd.read_csv('train.csv')

#Exporatory data analysis

sns.heatmap(df.isnull(),yticklabels=False)
sns.set_style('whitegrid')
sns.countplot('Survived',data=df)

sns.countplot('Survived',hue='Sex',data=df)

sns.countplot('Survived',hue='Pclass',data=df)

sns.distplot(df['Age'].dropna(),kde=False,bins=20)

sns.countplot('SibSp',data=df)

sns.distplot(df['Fare'],kde=False,bins=20,color='red')


####data cleaning   #######################

#how to remove null values
sns.boxplot(x='Pclass',y='Age',data=df)

def impute_Age(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        if Pclass==1:
            return(37)
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age
df['Age']=df[['Age','Pclass']].apply(impute_Age,axis=1)
sns.heatmap(df.isnull(),yticklabels=False)

df.drop('Cabin',axis=1,inplace=True)  
sns.heatmap(df.isnull(),yticklabels=False)  

embarked=pd.get_dummies(df['Embarked'],drop_first=True)
sex=pd.get_dummies(df['Sex'],drop_first=True)

df.drop(['Ticket','Name','Sex','Embarked'],axis=1,inplace=True)
dataset=pd.concat([df,sex,embarked],axis=1)
