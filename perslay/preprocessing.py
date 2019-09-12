"""Module :mod:`perslay.preprocessing` implement preprocessing for perslay compatibility."""

# Authors: Mathieu Carriere <mathieu.carriere3@gmail.com>
# License: MIT

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class BirthPersistenceTransform(BaseEstimator, TransformerMixin):

    def __init__(self):
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xfit = []
        for diag in X:
            new_diag = np.empty(diag.shape)
            np.copyto(new_diag, diag)
            new_diag[:,1] = new_diag[:,1] - new_diag[:,0]
            Xfit.append(new_diag)
        return Xfit

class Clamping(BaseEstimator, TransformerMixin):

    def __init__(self, limit=np.inf):
        self.limit = limit

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xfit = np.where(X >= self.limit, self.limit * np.ones(X.shape), X)
        return Xfit

class DiagramScaler(BaseEstimator, TransformerMixin):

    def __init__(self, use=False, scalers=[]):
        self.scalers  = scalers
        self.use      = use

    def fit(self, X, y=None):
        if self.use:
            if len(X) == 1:
                P = X[0]
            else:
                P = np.concatenate(X,0)
            for (indices, scaler) in self.scalers:
                scaler.fit(np.reshape(P[:,indices], [-1, 1]))
        return self

    def transform(self, X):
        Xfit = [np.copy(d) for d in X]
        if self.use:
            for i in range(len(Xfit)):
                if Xfit[i].shape[0] > 0:
                    for (indices, scaler) in self.scalers:
                        for I in indices:
                            Xfit[i][:,I] = np.squeeze(scaler.transform(np.reshape(Xfit[i][:,I], [-1,1])))
        return Xfit

class Padding(BaseEstimator, TransformerMixin):

    def __init__(self, use=False):
        self.use = use

    def fit(self, X, y=None):
        self.max_pts = max([len(diag) for diag in X])
        return self

    def transform(self, X):
        if self.use:
            Xfit, num_diag = [], len(X)
            for diag in X:
                diag_pad = np.pad(diag, ((0,max(0, self.max_pts - diag.shape[0])), (0,1)), "constant", constant_values=((0,0),(0,0)))
                diag_pad[:diag.shape[0],2] = np.ones(diag.shape[0])
                Xfit.append(diag_pad)                    
        else:
            Xfit = X
        return Xfit

class ProminentPoints(BaseEstimator, TransformerMixin):

    def __init__(self, use=False, num_pts=10, threshold=-1, location="upper"):
        self.num_pts    = num_pts
        self.threshold  = threshold
        self.use        = use
        self.location   = location

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.use:
            Xfit, num_diag = [], len(X)
            for i in range(num_diag):
                diag = X[i]
                if diag.shape[1] >= 2:
                    if diag.shape[0] > 0:
                        pers       = np.abs(diag[:,1] - diag[:,0])
                        idx_thresh = pers >= self.threshold
                        thresh_diag, thresh_pers  = diag[idx_thresh], pers[idx_thresh]
                        sort_index  = np.flip(np.argsort(thresh_pers, axis=None), 0)
                        if self.location == "upper":
                            new_diag = thresh_diag[sort_index[:min(self.num_pts, thresh_diag.shape[0])],:]
                        if self.location == "lower":
                            new_diag = np.concatenate( [ thresh_diag[sort_index[min(self.num_pts, thresh_diag.shape[0]):],:], diag[~idx_thresh] ], axis=0)
                    else:
                        new_diag = diag

                else:
                    if diag.shape[0] > 0:
                        birth      = diag[:,:1]
                        idx_thresh = birth >= self.threshold
                        thresh_diag, thresh_birth  = diag[idx_thresh], birth[idx_thresh]
                        if self.location == "upper":
                            new_diag = thresh_diag[:min(self.num_pts, thresh_diag.shape[0]),:]
                        if self.location == "lower":
                            new_diag = np.concatenate( [ thresh_diag[min(self.num_pts, thresh_diag.shape[0]):,:], diag[~idx_thresh] ], axis=0)
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
                Xfit = [ diag[diag[:,1] < self.limit] if diag.shape[0] != 0 else diag for diag in X]
            else:
                Xfit = [ diag[diag[:,1] == self.limit, 0:1] if diag.shape[0] != 0 else diag for diag in X]
        else:
            Xfit = X
        return Xfit


class nu_separator(BaseEstimator, TransformerMixin):
    """ Preprocessing class used in 'Learning Representations of Persistence Barcodes' """
    def __init__(self, nu=0.1):
        self.nu = nu

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xfit = np.where(X >= self.nu, X, np.maximum(2*self.nu - self.nu*self.nu/X, -1e-10 * np.ones(X.shape)))
        return Xfit


def preprocess(diag, thresh=500, nu_scale=False):

    filts = diag.keys()

    # Whole pipeline
    if nu_scale:
        tmp = Pipeline([
            ("Selector",      DiagramSelector(use=True, point_type="finite")),
            ("ProminentPts",  ProminentPoints(use=True, num_pts=thresh)),
            ("Scaler",        DiagramScaler(use=True,  scalers=[([0,1], MinMaxScaler())])),
            ("BPT",           BirthPersistenceTransform()),
            ("NuSeparator",   DiagramScaler(use=True, scalers=[([1], nu_separator(nu=.1))])),
            ("Padding",       Padding(use=True)),
                          ])
    else:
        tmp = Pipeline([
            ("Selector",      DiagramSelector(use=True, point_type="finite")),
            ("ProminentPts",  ProminentPoints(use=True, num_pts=thresh)),
            ("Scaler",        DiagramScaler(use=True,  scalers=[([0,1], MinMaxScaler())])),
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

    return D_pad, prm.keys()
