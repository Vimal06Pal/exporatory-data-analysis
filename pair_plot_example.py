''' pair plot is similar to scatter plot but in sactter plot we 
can plot one by one feature at a time but in pair plot we can take a sequence of parameter 
and print the scatter plot
'''
# importing libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

#importing dataset
from sklearn.datasets import load_breast_cancer
cancer_dataset = load_breast_cancer()
# print(cancer_dataset)

# preparing dataset
cancer_df = pd.DataFrame(np.c_[cancer_dataset['data'],cancer_dataset['target']],
columns = np.append(cancer_dataset['feature_names'],['target']))
print(cancer_df)

# sns.pairplot(cancer_df)
def plot_using_various_parameter(cancer_df):
    ''' plot graph by selected some perticular parameter'''
    sns.pairplot(cancer_df,vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
           'mean smoothness'])
    plt.show()


def plot_using_hue_parameter(cancer_df):
    sns.pairplot(cancer_df,vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
           'mean smoothness'],hue = 'target')
    plt.show()


def plot_using_palette(cancer_df):
    sns.pairplot(cancer_df,vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness'],hue = 'target',palette='cividis')
    plt.show()


def plot_according_to_our_require_param(cancer_df):

    '''
    if we want to plot the pair plot according to our requirement 
    as x parameter and y parameter
    '''
    sns.pairplot(cancer_df, hue = 'target',
    x_vars=['mean radius', 'mean texture'],y_vars=['mean radius'])
    plt.show()


# managing_diagonal_graph
def for_read(cancer_df):
    '''in diagornal ,graphs are automatically generated as kde(kernel distribution estimator)
and historgam, if we want to generate according to our choice we can use 
diag_kind = 'hist' orf 'kde'
'''
    sns.pairplot(cancer_df, vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area'],hue = 'target',
    kind='reg')
    plt.show()


def for_hist(cancer_df):
    sns.pairplot(cancer_df, vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area'],hue = 'target',
    diag_kind='hist')
    plt.show()


def marker_parameter(cancer_df):
    ''' marker is used to plot datapoint shapes that denotes the point on the graph
    '''
    sns.pairplot(cancer_df, vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area'],hue = 'target',
    diag_kind='hist',markers=['*','<'])
    plt.show()


def fig_size_increase(cancer_df):
    '''
    if we want to increase the size of the pair plot
    '''
    sns.pairplot(cancer_df, vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area'],hue = 'target',
    diag_kind='hist',markers=['*','<'],height = 20)
    plt.show()



if __name__ == "__main__":
    plot_using_various_parameter(cancer_df)

    plot_using_hue_parameter(cancer_df)

    plot_using_palette(cancer_df)

    plot_according_to_our_require_param(cancer_df)

    for_read(cancer_df)

    for_hist(cancer_df)

    marker_parameter(cancer_df)

    fig_size_increase(cancer_df)