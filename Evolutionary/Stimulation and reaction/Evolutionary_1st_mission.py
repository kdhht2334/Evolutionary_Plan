"""I refer following web site:
    https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import time

import torch

# Our 2-dimensional distribution will be over variables X and Y
N = 60
X = np.linspace(-3, 3, N)
Y = np.linspace(-3, 4, N)
X, Y = np.meshgrid(X, Y)

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N


for i in range(100):
#    x, y = np.mgrid[-1.0:1.0:30j, -1.0:1.0:30j]
#    
#    # Need an (N, 2) array of (x, y) pairs.
#    xy = np.column_stack([x.flat, y.flat])
#    
#    mu = np.array([0.0, 0.0])
#    
#    sigma = np.array([.5 + 0.1*i, .5 + 0.1*i])
#    covariance = np.diag(sigma**2)
#    
#    z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
#    
#    # Reshape back to a (30, 30) grid.
#    z = z.reshape(x.shape)
#    
#    fig = plt.figure()
#    
#    ax = fig.add_subplot(111, projection='3d')
#    ax.plot_surface(x,y,z)
#    plt.savefig('d.png')
#    plt.show()
#    #ax.plot_wireframe(x,y,z)
    
    # Mean vector and covariance matrix
    mu = np.array([0., 1. + 0.1*i])
    Sigma = np.array([[ 1. + 0.5 * i , -0.5], [-0.5,  1.5 + 0.5*i]])
    
    # The distribution on the variables X, Y packed into pos.
    Z = multivariate_gaussian(pos, mu, Sigma)
    
    # Create a surface plot and projected filled contour plot under it.
    fig = plt.figure(figsize=(10,7))
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                    cmap=cm.viridis)
    
#    cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)
    
    # Adjust the limits, ticks and view angle
#    ax.set_zlim(-0.15,0.2)
    ax.set_zlim(0.0,0.1)
    ax.set_zticks(np.linspace(0,0.1,5))
    ax.view_init(27, -21)
    ax.set_title('{}-th experiments'.format(i))
    plt.axis('off')
    plt.show()
    
    time.sleep(1)

