import camera
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# loading images
im1=np.array(Image.open('imaged/001.jpg'))
im2=np.array(Image.open('imaged/001.jpg'))

# loading 2D points for each images
points2D=[np.loadtxt('2D/00'+str(i+1)+'.corners').T for i in range (3)]

# loading 3D points
point3D=np.loadtxt('3D/p3d').T

# loading correspondences
corr=np.genfromtxt('2D/nviews-corners', dtype='int', missing_values='*')

# loading cameras
P=[camera.Camera(np.loadtxt('2D/00'+str(i+1)+'.P')) for i in range(3)]

X_s=np.vstack(point3D,np.ones(point3D.shape[1]))
x=P[0].project(X_s)

# plotting on view1
plt.figure()
plt.imshow(im1)
plt.plot(points2D[0][0],points2D[0][1],'*')
plt.axis('off')

plt.figure()
plt.imshow(im1)
plt.plot(x[0], x[1], 'r.')
plt.axis=('off')

plt.show()

# plotting in 3D
fig=plt.figure
ax=fig.gca(projection="3d")

X,Y,Z=axes3d.get_test_data(0.25)
ax.plot(X.flatten(), Y.flatten(), Z.flatten(), 'o')
ax.show()

