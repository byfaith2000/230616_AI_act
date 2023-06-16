import matplotlib.pyplot as plt
import numpy as np

#print(matplotlib.__version__)

#plt.plot([1, 2, 3, 4])
#plt.plot([1,2,3,4], [1 ,4, 9, 16], 'ro')
#plt.plot([1,2,3,4], [1 ,4, 9, 16], 'r-')
#plt.plot([1,2,3,4], [1 ,4, 9, 16], 'r--')
#plt.plot([1,2,3,4], [1 ,4, 9, 16], 'ro:')
#plt.plot([1,2,3,4], [1 ,4, 9, 16], ':', marker='o')
#plt.plot([1,2,3,4], [1 ,4, 9, 16], ':', color='g', marker='o', markerfacecolor='r')
#plt.plot([1,2,3,4], [1 ,4, 9, 16], ':', color='g', marker='o', markerfacecolor='r', label='line_1')
#temp = np.arange(0, 5, 0.2)
#plt.plot(temp, temp, 'bo')
#plt.plot(temp, temp**2, 'ro')

#plt.plot(temp, temp, 'ro', temp, temp**2, 'g*')
#print(temp)

#data_dict = {'d_x': [1,2,3,4,5], 'd_y':[2,3,5,10,8]}
#plt.plot('d_x', 'd_y', data=data_dict)

# x = np.arange(4)

# years = ['2020', '2021', '2022', '2023']
# values = [100, 400, 700, 1000]

# plt.bar(x, values)

# plt.xticks(x, years)

# weight = [10, 40, 20, 30, 60, 70, 55, 48, 44, 39]

# plt.hist(weight)

# ratio = [23, 44, 54, 33]
# labels = ['Apple', 'Banana', 'Melon', 'Peach']

# plt.pie(ratio, labels=labels, autopct='%.1f%%')

# arr = np.random.standard_normal((30, 40))
# plt.matshow(arr)

n = 50
x = np.random.rand(n)
y = np.random.rand(n)
area = (30 * np.random.rand(n))**2
colors = np.random.rand(n)

#plt.scatter(x,y)
plt.scatter(x,y, s=area, c=colors, alpha=0.5, cmap='Spectral')
plt.colorbar()

plt.legend()

plt.xlabel('x_label')
plt.ylabel('y_label')

#plt.axis([0, 6, 0, 20])
#plt.axis([0, 6, 0, 30])

plt.grid(True)

plt.title("230613")

plt.show()