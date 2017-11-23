import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D

# generate some sample data
import scipy.misc

pickle_in = open("heat_map.pickle","rb")
heat_map = pickle.load(pickle_in)
w, h = heat_map.shape

print heat_map.shape

# downscaling has a "smoothing" effect
heat_map = scipy.misc.imresize(heat_map, 0.50, interp='cubic')
print heat_map.shape
# create the x and y coordinate arrays (here we just use pixel indices)
xx, yy = np.mgrid[0:heat_map.shape[0], 0:heat_map.shape[1]]

# create the figure
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, heat_map ,rstride=1, cstride=1, cmap=plt.cm.jet, linewidth=0)

# show it
plt.show()