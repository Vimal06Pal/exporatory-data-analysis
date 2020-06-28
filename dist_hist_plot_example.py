''' distplot using seaborn '''

#importing libraries

import seaborn as sns
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

def increase_size_of_fig(cancer_df):
    ''' increase the fig size and help to introduce grid cells
    in the background'''
    plt.figure(figsize=(16,9))
    sns.set()
    sns.distplot(cancer_df['mean radius'],label = 'mean radius')
    plt.title = ('histogram of mean radius')
    plt.legend()
    plt.show()

def to_show_bins_on_the_x_axis_according_to_parameters(cancer_df):
    '''we get the bar within the range of the specified bins'''
    bins = [1,5,10,15,20,25,30]
    sns.set()
    sns.distplot(cancer_df['mean radius'],bins = bins)
    plt.xticks = bins
    plt.title = ('histogram of mean radius')
    plt.show() 

def showing_dist_kws(cancer_df):
    ''' showing kws for hist'''
    sns.set()
    sns.distplot(cancer_df['mean radius'],hist_kws = {'color': '#DC143C','edgecolor':'#aaff00',
    'linewidth':1,'linestyle':'--','alpha':0.9})
    plt.show() 

def showing_kde_kws(cancer_df):
    ''' showing kws for kde '''
    sns.set()
    sns.distplot(cancer_df['mean radius'],hist_kws = {'color': '#DC143C','edgecolor':'#aaff00',
    'linewidth':1,'linestyle':'--','alpha':0.9},
    kde_kws = {'color': '#8e00ce',          
    'linewidth':1,'linestyle':'--','alpha':0.9})        # all parameter of kde_kws  is similar to hist_kws but not edgecolor

    plt.show()

def showing_rugplot_kws(cancer_df):
    ''' showing kws for kde '''
    sns.set()
    sns.distplot(cancer_df['mean radius'],hist_kws = {'color': '#DC143C','edgecolor':'#aaff00',
    'linewidth':1,'linestyle':'--','alpha':0.9},

    kde_kws = {'color': '#8e00ce',          
    'linewidth':1,'linestyle':'--','alpha':0.9},        # all parameter of kde_kws  is similar to hist_kws but not edgecolor
    
    rug = True,
    
    rug_kws = {'color': '#0426d0','edgecolor':'#00dbff',
    'linewidth':3,'linestyle':'--','alpha':0.9})
    
    plt.show()

def showing_fit_kws(cancer_df):
    ''' showing kws for kde '''
    plt.figure(figsize=(16,9))
    sns.set()
    sns.distplot(cancer_df['mean radius'],hist_kws = {'color': '#DC143C','edgecolor':'#aaff00',
    'linewidth':1,'linestyle':'--','alpha':0.9},

    kde =False,
    fit = sci.stats.norm,
    fit_kws = {'color': '#8e00ce',
    'linewidth':12,'linestyle':'--','alpha':0.4},
    
    rug = True,
    
    rug_kws = {'color': '#0426d0','edgecolor':'#00dbff',
    'linewidth':5,'linestyle':'--','alpha':0.9})
    
    plt.show()


if __name__ == "__main__":
    # distplot_inc_bins(cancer_df)

    # distplot_only_kde(cancer_df)

    # distplot_only_histogram(cancer_df)

    # distplot_rug(cancer_df)

    # displot_with_two_kde_using_norm_in_scipy(cancer_df)

    # color_changing_distplot(cancer_df)

    # increase_size_of_fig(cancer_df)

    # print(cancer_df['mean radius'].sort_values()) # to get the range of the bins

    # to_show_bins_on_the_x_axis_according_to_parameters(cancer_df)

    # showing_dist_kws(cancer_df)

    # showing_kde_kws(cancer_df)

    # showing_rugplot_kws(cancer_df)

    showing_fit_kws(cancer_df)
