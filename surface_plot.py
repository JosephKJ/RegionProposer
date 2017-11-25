import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D

# generate some sample data
import scipy.misc

pickle_in = open("heat_map.pickle","rb")
heat_map = pickle.load(pickle_in)
# w, h = heat_map.shape

print heat_map.shape
heat_map = heat_map

# downscaling has a "smoothing" effect
heat_map = scipy.misc.imresize(heat_map, 0.50, interp='cubic')
print heat_map.shape
# create the x and y coordinate arrays (here we just use pixel indices)
xx, yy = np.mgrid[0:heat_map.shape[0], 0:heat_map.shape[1]]
# create the figure
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, heat_map ,rstride=1, cstride=1, cmap=plt.cm.jet, linewidth=0)

# ax.view_init(45, 45)
# rotate the axes and update
# for angle in range(0, 180):
#     ax.view_init(30, angle)
#     plt.draw()
#     plt.pause(.001)
# plt.gca().invert_xaxis()
# plt.gca().invert_yaxis()
# plt.gca().invert_zaxis()
ax.view_init(75, 45)
# show it
plt.show()
