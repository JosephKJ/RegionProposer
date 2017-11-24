import mayavi.mlab as mlab
import numpy as np
x,y = np.mgrid[-1:1:0.001, -1:1:0.001]
z = x**2+y**2
s = mlab.mesh(x, y, z)
alpha = 30  # degrees
mlab.view(azimuth=0, elevation=90, roll=-90+alpha)

mlab.show()