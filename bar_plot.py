import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns



tips = sns.load_dataset('tips')
print(tips.head())
''' ----------------bar plots----------- '''
'''
hue = on what basis you want to plot graph on x,y axis
order= which order you want to follow on x axis
estimator = wheather you want median etc but by default it is mean
'''
'''
day vs tip
'''
# sns.barplot(x='day',y='tip',data=tips)

'''
day vs tip according to their sex and   hue_order
sns.barplot(x='day',y='total_bill',data = tips,hue = 'sex',hue_order = ['Female','Male'])

and plotting a graph in order= ['Sat','Fri','Sun','Thur']
'''
# sns.barplot(x='day',y='tip',data=tips,hue='smoker',order=['Sat','Fri','Sun','Thur'])
'''
day vs total bill 
'''
# sns.barplot(x='day',y='total_bill',data = tips)



'''
applying estimator it is the range we want to set in the y parameter 
eg:- mean median,max etc
'''
# sns.barplot(x='day',y='tip',data=tips,hue='smoker',estimator = np.max,order=['Sat','Fri','Sun','Thur'])
# sns.barplot(x='day',y='tip',data=tips,hue='smoker',estimator = np.median,order=['Sat','Fri','Sun','Thur'])
 
'''
 including ci which is confidence interval 
 basically SHOWS  athe  percentile of the bars 
 '''
# sns.barplot(x='smoker',y='tip',data = tips,hue = 'sex',palette = 'hot',ci=12)
# sns.barplot(x='smoker',y='tip',data = tips,hue = 'sex',estimator = np.max,palette = 'BuPu',ci=12,saturation = .2 )
'''
bar graph using kwargs
'''
# kwargs = {'alpha':0.9,'linestyle':':','linewidth':5,'edgecolor':'k'}
# ax=sns.barplot(x = 'day',y = 'tip',hue = 'smoker',data = tips,**kwargs)

# ax.set(title = 'barplot of tips df')
# # 		xlabel = 'Days'
# # 		ylabel = 'tip')

'''
bar plot by giving specific size of fig
'''
plt.figure(figsize = (6,7))
sns.barplot(x='smoker',y='tip',data = tips,hue = 'sex',estimator = np.max,palette = 'BuPu',ci=12,saturation = .2 )


plt.show()
 
