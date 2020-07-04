# importing libraries

import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import scipy as sci

# loading dataset

from sklearn.datasets import load_breast_cancer
cancer_dataset = load_breast_cancer()
# print(cancer_dataset.data)


# preparring dataset

cancer_df = pd.DataFrame(np.c_[cancer_dataset['data'],cancer_dataset['target']],
columns = np.append(cancer_dataset['feature_names'],['target']))
print(cancer_df)

# to show all the columns we use pd.set_option
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
print(cancer_df.head())

# to get information of dataset

print(cancer_df.info())

def distplot_inc_bins(cancer_df):
    ''' bins helps in divide the data into no of bins columns '''
    sns.set()
    sns.distplot(cancer_df['mean radius'])
    plt.show()

def scatterplot(cancer_df):
    sns.set()
    sns.scatterplot(x='mean radius',y='mean texture',data=cancer_df, hue = 'target',style = 'target')
    plt.show()

def mean_texture(cancer_df):
    sns.set()
    sns.distplot(cancer_df['mean texture'])
    plt.show()

def correlation(cancer_df):
    corr_df = cancer_df.corr()
    print(corr_df)



if __name__ == "__main__":
    distplot_inc_bins(cancer_df)
    mean_texture(cancer_df)
    scatterplot(cancer_df)

    correlation(cancer_df)

    x = cancer_df.drop(['target'], axis = 1)
    y = cancer_df['target']

    from sklearn.model_selection import train_test_split

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 ,random_state = 0)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()

    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)


    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(x_train, y_train)

    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm,annot = True)
    plt.show()

    print(cm)