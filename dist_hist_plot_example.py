''' distplot using seaborn '''

#importing libraries

mport seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import scipy as sci

# loading dataset
from sklearn.datasets import load_breast_cancer
cancer_dataset = load_breast_cancer()
# print(cancer_dataset)

# preparring dataset
cancer_df = pd.DataFrame(np.c_[cancer_dataset['data'],cancer_dataset['target']],
columns = np.append(cancer_dataset['feature_names'],['target']))
print(cancer_df)

# sns.distplot(cancer_df['mean radius'])
def distplot_inc_bins(cancer_df):
    ''' bins helps in divide the data into no of bins column'''
    sns.distplot(cancer_df['mean radius'])
    plt.show()

def distplot_only_kde(cancer_df):
    ''' by default hist = true,it gives us histogrsm as well as kde
    but when hist = false it give only kde no histogram'''
    sns.distplot(cancer_df['mean radius'],hist = False)
    plt.show() 

def distplot_only_histogram(cancer_df):
    ''' by default kde = true,it gives us histogram as well as kde
    but when kde = false it give only histogram  no kde'''
    sns.distplot(cancer_df['mean radius'],kde = False)
    plt.show()

def distplot_rug(cancer_df):
    ''' to draw a rugplot on the support axis.'''
    sns.distplot(cancer_df['mean radius'],rug = True)
    plt.show()

def displot_with_two_kde_using_norm_in_scipy(cancer_df):
    '''plotting  kde  and normalised line plot using norm in scipy.stats '''
    sns.distplot(cancer_df['mean radius'],fit = sci.stats.norm)
    plt.show() 

def color_changing_distplot(cancer_df):
    ''' changing color of histogram'''
    sns.distplot(cancer_df['mean radius'],color ='r')
    plt.show()

if __name__ == "__main__":
    distplot_inc_bins(cancer_df)

    distplot_only_kde(cancer_df)

    distplot_only_histogram(cancer_df)

    distplot_rug(cancer_df)

    displot_with_two_kde_using_norm_in_scipy(cancer_df)

    color_changing_distplot(cancer_df)
