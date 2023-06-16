import sklearn
#print(sklearn.__version__)

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

import pandas as pd

b_cancer = datasets.load_breast_cancer()
#print(b_cancer)
#print(type(b_cancer))
#print(b_cancer.feature_names)
#print(b_cancer.target_names)


# b_cancer_df = pd.DataFrame(data=b_cancer.data, columns=b_cancer.feature_names)
# print(b_cancer_df)
# b_cancer_df['target'] = b_cancer.target
# print(b_cancer_df)

data_train, data_test, target_train, target_test = train_test_split(b_cancer.data, b_cancer.target, test_size=0.3, random_state=615)

#plt.scatter(x,y)
#plt.plot(data_test)
print(data_train.shape)
data_train_r = data_train.reshape(11940, 1)

# plt.hist(data_test, bins = 30)
# plt.show()


from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()

data_train_scaled = mms.fit_transform(data_train)
data_test_scaled = mms.transform(data_test)

print(data_train_scaled)

data_train_scaled_r = data_train_scaled.reshape(11940, 1)

# plt.subplot(1,2,1)
# plt.hist(data_train_r, bins=30, color='blue')
# plt.subplot(1,2,2)
# plt.hist(data_train_scaled_r, bins=30, color='red')
# plt.show()

from sklearn.preprocessing import StandardScaler

stds = StandardScaler()
data_train_scaled_s = stds.fit_transform(data_train)
data_test_scaled_s = stds.fit_transform(data_test)

data_train_scaled_s_r = data_train_scaled_s.reshape(11940, 1)

# plt.subplot(1,2,1)
# plt.hist(data_train_r, bins=30, color='blue')
# plt.subplot(1,2,2)
# plt.hist(data_train_scaled_s_r, bins=30, color='red')
# plt.show()

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

Irs = LogisticRegression(random_state=42)
Irs.fit(data_train, target_train)

print(f'model perf : {round(Irs.score(data_test, target_test), 4)}')

Irs.fit(data_train_scaled_m, target_train)
print(f'model perf_MMS : {round(Irs.score(data_test_scaled_m, target_test), 4)}')

Irs.fit(data_train_scaled_s, target_train)
print(f'model perf_STDS : {round(Irs.score(data_test_scaled_s, target_test), 4)}')
