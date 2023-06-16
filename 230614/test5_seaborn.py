import seaborn as sns
import matplotlib.pyplot as plt

#print(sns.__version__)

f = sns.load_dataset('flights')
print(f)

#sns.barplot(data=f, x='year', y='passengers')
#sns.barplot(data=f, x='month', y='passengers')

#sns.boxplot(data=f, x='year', y='passengers')
#sns.boxplot(data=f, x='month', y='passengers')

#sns.swarmplot(data=f, x='year', y='passengers')
#sns.swarmplot(data=f, x='month', y='passengers')

#sns.lineplot(data=f, x='year', y='passengers')
sns.lineplot(data=f, x='month', y='passengers')

#sns.distplot(f['passengers'])

plt.show()