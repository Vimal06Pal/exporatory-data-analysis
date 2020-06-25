''' pair plot is similar to scatter plot but in sactter plot we 
can plot one by one feature at a time but in pair plot we can take a sequence of parameter 
and print the scatter plot
'''

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.datasets import load_breast_cancer
cancer_dataset = load_breast_cancer()
# print(cancer_dataset)

cancer_df = pd.DataFrame(np.c_[cancer_dataset['data'],cancer_dataset['target']],
columns = np.append(cancer_dataset['feature_names'],['target']))
print(cancer_df)

# sns.pairplot(cancer_df)
# sns.pairplot(cancer_df,vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
#        'mean smoothness'])
# sns.pairplot(cancer_df,vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
    #    'mean smoothness'],hue = 'target')
# sns.pairplot(cancer_df,vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
#        'mean smoothness'],hue = 'target',palette='cividis')

'''
if we want to plot the pair plot according to our requirement 
as x parameter and y parameter
'''
# sns.pairplot(cancer_df, hue = 'target',
# x_vars=['mean radius', 'mean texture'],y_vars=['mean radius'])

# sns.pairplot(cancer_df, vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area'],hue = 'target',
# kind='reg')
'''
in diagornal ,graphs are automatically generated as kde(kernel distribution estimator)
and historgam, if we want to generate according to our choice we can use 
diag_kind = 'hist' orf 'kde'
'''
# sns.pairplot(cancer_df, vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area'],hue = 'target',
# diag_kind='hist')

''' marker is used to plot datapoint shapes that denotes the point on the graph
'''
# sns.pairplot(cancer_df, vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area'],hue = 'target',
# diag_kind='hist',markers=['*','<'])

'''
if we want to increase the size of the pair plot
'''
sns.pairplot(cancer_df, vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area'],hue = 'target',
diag_kind='hist',markers=['*','<'],height = 20)
plt.show()