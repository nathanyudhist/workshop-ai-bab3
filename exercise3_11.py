# Exercise 3.11
# Modify Example 3.15 (linear regression) to use more data points, and add
# an x label, y label, title, legend, and grid to the plot.
import matplotlib.pyplot as plt
from scipy import stats

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [3, 5, 5, 6, 7, 8, 8, 9, 11, 12]

slope, intercept, r, p, std_err = stats.linregress(x, y)
print("slope:", slope)
print("intercept:", intercept)

def myfunc(x):
    return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y, label='Data points')
plt.plot(x, mymodel, color='red', label='Best-fit line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
