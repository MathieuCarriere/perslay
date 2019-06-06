"""Module :mod:`perskay.preprocessing` implement preprocessing for perslay compatibility."""

# Authors: Mathieu Carriere <mathieu.carriere3@gmail.com>
# License: MIT


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#############################################
# Preprocessing #############################
#############################################

class BirthPersistenceTransform(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.matmul(X, np.array([[1., -1.],[0., 1.]]))


class DiagramPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, use=False, scalers=[]):
        self.scalers = scalers
        self.use     = use

    def fit(self, X, y=None):
        if self.use:
            if len(X) == 1:
                P = X[0]
            else:
                P = np.concatenate(X,0)
            for (indices, scaler) in self.scalers:
                scaler.fit(P[:,indices])
        return self

    def transform(self, X):
        Xfit = [np.copy(d) for d in X]
        if self.use:
            for i in range(len(Xfit)):
                if Xfit[i].shape[0] > 0:
                    for (indices, scaler) in self.scalers:
                        Xfit[i][:,indices] = scaler.transform(Xfit[i][:,indices])
        return Xfit

class Padding(BaseEstimator, TransformerMixin):

    def __init__(self, use=False):
        self.use = use

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.use:
            Xfit, num_diag = [], len(X)
            max_card = max([len(diag) for diag in X])
            for diag in X:
                [num_pts, dim] = diag.shape
                diag_pad = np.zeros([max_card, dim+1])
                diag_pad[:num_pts,:dim] = diag
                diag_pad[:num_pts, dim] = np.ones(num_pts)
                Xfit.append(diag_pad)                    
        else:
            Xfit = X
        return Xfit


class ProminentPoints(BaseEstimator, TransformerMixin):

    def __init__(self, use=False, num_pts=10, threshold=-1, location="upper", point_type="finite"):
        self.num_pts = num_pts
        self.threshold = threshold
        self.use = use
        self.location = location
        self.point_type = point_type

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.use:
            Xfit, num_diag = [], len(X)
            for i in range(num_diag):
                diag = X[i]
                if self.point_type == "finite":
                    if diag.shape[0] > 0:
                        pers       = np.abs(np.matmul(diag[:,:2], [-1., 1.]))
                        idx_thresh = pers >= self.threshold
                        thresh_diag, thresh_pers  = diag[idx_thresh.flatten()], pers[idx_thresh.flatten()]
                        sort_index  = np.flip(np.argsort(thresh_pers, axis=None), 0)
                        if self.location == "upper":
                            new_diag = thresh_diag[sort_index[:min(self.num_pts, thresh_diag.shape[0])],:]
                        if self.location == "lower":
                            new_diag = np.concatenate( [ thresh_diag[sort_index[min(self.num_pts, thresh_diag.shape[0]):],:], diag[~idx_thresh.flatten()] ], axis=0)
                    else:
                        new_diag = diag

                else:
                    if diag.shape[0] > 0:
                        birth      = diag[:,:1]
                        idx_thresh = birth >= self.threshold
                        thresh_diag, thresh_birth  = diag[idx_thresh.flatten()], birth[idx_thresh.flatten()]
                        if self.location == "upper":
                            new_diag = thresh_diag[:min(self.num_pts, thresh_diag.shape[0]),:]
                        if self.location == "lower":
                            new_diag = np.concatenate( [ thresh_diag[min(self.num_pts, thresh_diag.shape[0]):,:], diag[~idx_thresh.flatten()] ], axis=0)
                    else:
                        new_diag = diag

                Xfit.append(new_diag)                    
        else:
            Xfit = X
        return Xfit


class DiagramSelector(BaseEstimator, TransformerMixin):

    def __init__(self, use=False, limit=np.inf, point_type="finite"):
        self.use, self.limit, self.point_type = use, limit, point_type

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.use:
            Xfit, num_diag = [], len(X)
            if self.point_type == "finite":
                for i in range(num_diag):
                    diag = X[i]
                    if diag.shape[0] != 0:
                        idx_fin = diag[:,1] != self.limit
                        Xfit.append(diag[idx_fin,:])
                    else:
                        Xfit.append(diag)
            if self.point_type == "essential":
                for i in range(num_diag):
                    diag = X[i]
                    if diag.shape[0] != 0:
                        idx_ess = diag[:,1] == self.limit
                        Xfit.append(np.delete(diag,1,1)[idx_ess,:])
                    else:
                        Xfit.append(np.delete(diag,1,1))
        else:
            Xfit = X
        return Xfit


# Preprocessing class used in "Deep learning for topological signatures"
class _nu_separator(BaseEstimator, TransformerMixin):
    def __init__(self, nu=0.1):
        self.nu = nu

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.matmul(X, (np.sqrt(2) / 2) * np.array([[1, -1], [1, 1]]))
        idx_up = X[:, 1] <= self.nu
        X[idx_up, 1] = self.nu * (1 + np.log(np.maximum(1e-10 * np.ones(X[idx_up, 1].shape), X[idx_up, 1]) / self.nu))
        return X


def preprocess(diag, thresh=500):
    filts = diag.keys()
    scaler = [([0, 1],  Pipeline([("1", BirthPersistenceTransform()), ("2", MinMaxScaler())]))]

    # Whole pipeline
    tmp = Pipeline([
        ("Selector",      DiagramSelector(use=True, point_type="finite")),
        ("ProminentPts",  ProminentPoints(use=True, num_pts=400, point_type="finite")),
        ("Scaler",        DiagramPreprocessor(use=True,  scalers=scaler)),
        ("NuSeparator",   DiagramPreprocessor(use=False, scalers=[([0, 1], _nu_separator(nu=.1))])),
        ("Padding",       Padding(use=True)),
                          ])
    prm = {filt: {"ProminentPts__num_pts": min(thresh, max([len(dgm) for dgm in diag[filt]]))} for filt in filts if
           max([len(dgm) for dgm in diag[filt]]) > 0}
    # Apply the previous pipeline on the different filtrations.
    D = []
    for dt in prm.keys():
        param = prm[dt]
        tmp.set_params(**param)
        D.append(tmp.fit_transform(diag[dt]))

    # For each filtration, concatenate all diagrams in a single array.
    D_pad = []
    for dt in range(len(prm.keys())):
        D_pad.append(np.concatenate([D[dt][i][np.newaxis, :] for i in range(len(D[dt]))], axis=0))
        # print(D_pad[dt].shape)

    return D_pad, filts
