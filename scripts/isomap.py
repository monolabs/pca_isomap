import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils.graph_shortest_path import graph_shortest_path
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


def construct_distance_matrix(X, points, dist_type='l2'):
    '''
    computes distance matrix of datapoints in X to multiple points
    Args:
        X: m x d array where m = # datapoints, d = dim
        points: n x d ndarray
        dist_type: type of distance, 'l2' or 'l1'
    Returns:
        distance matrix D: m x n ndarray where m = # datapoints in X, n = number of points to compute distances from
            
    D[i, j] = distance of point i in intialization array to point j in the argument points, where:
        i = {0, 1, ..., m}
        j = {0, 1, ..., n}
    '''
    # initialize empty distance matrix
    m = X.shape[0]
    d = X.shape[1]
    points = points.reshape((-1, d))
    n = points.shape[0]
    D = np.empty((m, n))
    
    if dist_type == 'l2':
        def dist_func(X, y):
            return np.linalg.norm(X-y, axis=1)
    
    if dist_type == 'l1':
        def dist_func(X, y):
            return (np.abs(X-y)).sum(axis=1)
        
    # compute distance and assign to D
    for j in range(n):
        D[:, j] = dist_func(X, points[j, :])    
        
    return D


class Isomap:
    
    def __init__(self, X):
        self.X = X
        self.m, self.d = X.shape
        
    def run_isomap(self, n_comp, epsilon, dist_type='l2'):
        dist_mat = construct_distance_matrix(self.X, self.X, dist_type=dist_type)
        dist_mat[dist_mat > epsilon] = 0
        D = graph_shortest_path(dist_mat, directed=False)
        H = np.identity(self.m) - 1/self.m * np.ones((self.m, self.m))
        G = -1/2 * np.matmul(H, np.matmul(D**2, H))
        
#         v, x = np.linalg.eig(G)
#         idx_sorted = np.argsort(-v)
#         v = v[idx_sorted]
#         x = x[:, idx_sorted]
#         v = v.real[:n_comp]
#         x = x.real[:, :n_comp]
#         print(f"eigenvalues:\n{v}\n\neigenvectors:\n{x}")
#         Z = np.matmul(x, np.diag(v))
        
        # with SVD
        u, s, vh = np.linalg.svd(G)
        lambda_ = np.diag(s[:n_comp])
        U = u[:, :n_comp]
        Z = np.matmul(U, lambda_)
        
        return Z
    
    
def annotate_images_random(ax, coord_arr, img_arr, img_full_size, zoom, n, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    indices = rng.integers(coord_arr.shape[0], size=n)
    
    # add image
    for i in indices:
        ax.plot(coord_arr[i, 0], coord_arr[i, 1], "or", markersize=5)
        img = img_arr[i].reshape((img_full_size)).T
        imagebox = OffsetImage(img, zoom=zoom, cmap='gray')
        xy = coord_arr[i]    # coordinates to position image

        ab = AnnotationBbox(imagebox, 
                            xy,
                            xybox=(50, -50.),
                            xycoords='data',
                            boxcoords="offset points")                                  
        ax.add_artist(ab)
        
    plt.draw()
    plt.show()
        
        
        