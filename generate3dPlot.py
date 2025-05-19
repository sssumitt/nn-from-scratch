import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the CSV data
data = np.loadtxt('xor_grid.csv', delimiter=',', skiprows=1)
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# Determine grid dimensions
n = int(np.sqrt(x.size))
xx = x.reshape((n, n))
yy = y.reshape((n, n))
zz = z.reshape((n, n))

# Create 3D surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, zz, rstride=5, cstride=5, edgecolor='none', alpha=0.7)

# Label axes
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Probability of Class 1')
ax.set_title('3D Decision Surface from xor_grid.csv')

plt.show()
