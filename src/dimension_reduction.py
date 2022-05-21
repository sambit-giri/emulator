import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA 
from sklearn import metrics # .r2_score

def pca_reduce(X, nComp=2, return_score=True):
    # mins, maxs = X.min(axis=0), X.max(axis=0)
    # X   = (X-mins)/(maxs-mins) 
    pca = PCA()
    pca.fit(X) 
    mu = np.mean(X, axis=0)
    coeff = pca.transform(X)[:,:nComp]
    Comp = pca.components_[:nComp,:]
    pca_dict = {'coeff': coeff, 'pca': pca, 'mu': mu, 'Comp': Comp}#, 'min': mins, 'max': maxs}
    if return_score:
        Xreconst = pca_expand(coeff, pca_dict=pca_dict)
        r2 = metrics.r2_score(X, Xreconst)
        print('r2 score:', r2)
    return pca_dict

def pca_expand(coeff, pca_dict=None, Comp=None, mu=0):
    if coeff.ndim==1: coeff = coeff[None,:]
    if pca_dict is not None:
        mu = pca_dict['mu']
        Comp = pca_dict['pca'].components_
    nComp    = coeff.shape[1]
    Xreconst = np.dot(coeff, Comp[:nComp,:]) + mu
    # Xreconst = pca_dict['min'] + Xreconst*(pca_dict['max']-pca_dict['min'])
    return Xreconst