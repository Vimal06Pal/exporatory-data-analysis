#importing libraries

import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from scipy.stats import norm


# some self created function 
def iqr(x):
    return x.quantile(q=0.75) - x.quantile(q=0.25)

#outlier >75th %tile + 1.5IQR & < 25th %tile - 1.5IQR
def remove_outlier(x):
    upper_out = x.quantile(q=0.75) + 1.5 * iqr(x)
    lower_out = x.quantile(q=0.25) - 1.5 * iqr(x)
    return(x[{x <=upper_out} &{x>= lower_out}])

class UnivriatePlotUtility(object):
    def __init__(self,x):
        self.x = x
    
    def num_plot(self):
        bins = max(round(self.x.nunique()/100),min(round(self.x.nunique(),10)))
        ax = sns.distplot(x,kde =True, hist = True,bins= bins,rug = False,color = 'bool')
        plt.xticks(rotation = 45)
        plt.show()

    def  non_outlier_num_plot(self):
        x = remove_outlier(self.x)
        bins = max(round(self.x.nunique()/100),min(round(self.x.nunique(),10)))
        ax = sns.distplot(x,kde =True, hist = True,bins= bins,rug = False,color = 'bool')
        plt.xticks(rotation = 45)
        plt.show()

    def cat_plot(self):
        plt.hist(self.x)
        plt.xticks(rotation = 50)
        plt.show()

    def distribution_plot(self):
        if self.x.dtypes == 'object' or self.x.dtypes == 'bool':
            self.cat_plot()
        else:
            self.num_plot()
        
    def non_outlier_distribution_plot(self):
        if self.x.dtypes == 'object' or self.x.dtypes == 'bool':
            self.cat_plot()
        else:
            self.non_outlier_num_plot()

df_train = pd.read_csv('./train.csv')
print(df_train.columns)

#understanding data 
''' univariate study
. Overview of the data
. how is my target variable
. how other variable llok like 
'''
# data overview 

def data_information(df,id_cols): 
    # removing id_column
    df = df.drop(columns = id_cols,axis=1)

    #creating empty DataFrame 
    data_info = pd.DataFrame(np.random.randn(0,12) * 0,
                            columns = ['No of observations (Mrow)',
                                       'No of variables (Ncol)',
                                       'No of numeric variables',
                                       'No of factor variables',
                                       'No of categorical variable',
                                       'No of logical variable',
                                       'No of Data variable',
                                       'No of Zero variance Variables (Uniform)',
                                       '% of variables having complete cases',
                                       '% of variables having <=50% missing cases',
                                       '% of variables having >50% missing cases',
                                       '% of variables having >90% missing cases'])  

    # data information 
    data_info.loc[0,'No of observations (Mrow)'] = df.shape[0] 
    data_info.loc[0,'No of variables (Ncol)'] = df.shape[1] 
    data_info.loc[0,'No of numeric variables'] = df._get_numeric_data().shape[1] 
    data_info.loc[0,'No of factor variables'] = df.select_dtypes(include = 'category').shape[1] 
    data_info.loc[0,'No of logical variables'] = df.select_dtypes(include = 'bool').shape[1] 
    data_info.loc[0,'No of categorical variables'] = df.select_dtypes(include = 'object').shape[1] 
    data_info.loc[0,'No of Gata variables'] = df.select_dtypes(include = 'datetime64').shape[1] 
    # data_info.loc[0,'No of Zero variance Variables (Uniform)'] = df.loc[:,df.apply(pd.Series.nunique)] 

    null_percent = pd.DataFrame(df.isnull().sum()/df.shape[0])
    print(null_percent)

id_cols = 'Id'
data_information(df_train,id_cols)

                        