import numpy as np
from matplotlib.offsetbox import TextArea, AnnotationBbox
import matplotlib.pyplot as plt

class PCAnalyzer:
    
    def __init__(self, X):
        self.X = X
        self.m, self.d = X.shape
        
    def construct_covariance_matrix(self):
        cov = np.zeros((self.d, self.d))
        self.X_centered = self.X - self.X.mean(axis=0)
        for x in self.X_centered:
            out_prod = np.outer(x, x)
            cov = cov + out_prod
    
        self.cov = cov/self.m
        
    def reduce_dim(self, new_d):
        self.construct_covariance_matrix()
#         x, v, _ = np.linalg.svd(self.cov)
#         x = x[:, :new_d]
#         v = v[:new_d]
        
        v, x = np.linalg.eig(self.cov)
        idx_sorted = np.argsort(-v)
        v = v[idx_sorted]
        x = x[:, idx_sorted]
        
        x = x[:, :new_d].real
        Z = np.matmul(x.T, self.X_centered.T).T
        return Z

    
def annotate(ax, X, names_list):
    assert len(names_list) == X.shape[0]
    # add texts
    for i in range(len(names_list)):
        xy = X[i]    # coordinates to position image
        offsetbox = TextArea(names_list[i], minimumdescent=False)
        ab = AnnotationBbox(offsetbox,
                            xy,
                            xybox=(0, -20),
                            xycoords='data',
                            boxcoords='offset points')                              
        ax.add_artist(ab)
        
    plt.draw()
    plt.show()