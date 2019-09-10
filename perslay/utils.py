"""Module :mod:`perslay.utils` provide utils functions."""

# Authors: Mathieu Carriere <mathieu.carriere3@gmail.com>
#          Theo Lacombe <theo.lacombe@inria.fr>
#          Martin Royer <martin.royer@inria.fr>
# License: MIT
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ast import literal_eval

import numpy as np
import gudhi as gd


def load_config(dataset):
    filepath = "./data/" + dataset + "/" + dataset + ".conf"
    with open(filepath) as fp:
        lines = fp.readlines()
        dataset_type = lines[0][:-1]
        list_filtrations = literal_eval(lines[1])
        thresh = int(lines[2])
        perslay_parameters = literal_eval(lines[3])
        optim_parameters = literal_eval(lines[4])
    return dataset_type, list_filtrations, thresh, perslay_parameters, optim_parameters


# Input utility functions for persistence diagrams
def diag_to_dict(diag_file, filts):
    out_dict = dict()
    if len(filts) == 0:
        filts = diag_file.keys()
    for filtration in filts:
        list_dgm, num_diag = [], len(diag_file[filtration].keys())
        for diag in range(num_diag):
            list_dgm.append(np.array(diag_file[filtration][str(diag)]))
        out_dict[filtration] = list_dgm
    return out_dict


# diagrams utils
def get_base_simplex(A):
    num_vertices = A.shape[0]
    st = gd.SimplexTree()
    for i in range(num_vertices):
        st.insert([i], filtration=-1e10)
        for j in range(i + 1, num_vertices):
            if A[i, j] > 0:
                st.insert([i, j], filtration=-1e10)
    return st.get_filtration()


# graph utils
def hks_signature(eigenvectors, eigenvals, time):
    return np.square(eigenvectors).dot(np.diag(np.exp(-time * eigenvals))).sum(axis=1)


def apply_graph_extended_persistence(A, filtration_val, basesimplex):
    num_vertices = A.shape[0]
    (xs, ys) = np.where(np.triu(A))
    num_edges = len(xs)

    if len(filtration_val.shape) == 1:
        min_val, max_val = filtration_val.min(), filtration_val.max()
    else:
        min_val = min([filtration_val[xs[i], ys[i]] for i in range(num_edges)])
        max_val = max([filtration_val[xs[i], ys[i]] for i in range(num_edges)])

    st = gd.SimplexTree()
    st.set_dimension(2)

    for simplex, filt in basesimplex:
        st.insert(simplex=simplex + [-2], filtration=-3)

    if len(filtration_val.shape) == 1:
        if max_val == min_val:
            fa = -.5 * np.ones(filtration_val.shape)
            fd = .5 * np.ones(filtration_val.shape)
        else:
            fa = -2 + (filtration_val - min_val) / (max_val - min_val)
            fd = 2 - (filtration_val - min_val) / (max_val - min_val)
        for vid in range(num_vertices):
            st.assign_filtration(simplex=[vid], filtration=fa[vid])
            st.assign_filtration(simplex=[vid, -2], filtration=fd[vid])
    else:
        if max_val == min_val:
            fa = -.5 * np.ones(filtration_val.shape)
            fd = .5 * np.ones(filtration_val.shape)
        else:
            fa = -2 + (filtration_val - min_val) / (max_val - min_val)
            fd = 2 - (filtration_val - min_val) / (max_val - min_val)
        for eid in range(num_edges):
            vidx, vidy = xs[eid], ys[eid]
            st.assign_filtration(simplex=[vidx, vidy], filtration=fa[vidx, vidy])
            st.assign_filtration(simplex=[vidx, vidy, -2], filtration=fd[vidx, vidy])
        for vid in range(num_vertices):
            if len(np.where(A[vid, :] > 0)[0]) > 0:
                st.assign_filtration(simplex=[vid], filtration=min(fa[vid, np.where(A[vid, :] > 0)[0]]))
                st.assign_filtration(simplex=[vid, -2], filtration=min(fd[vid, np.where(A[vid, :] > 0)[0]]))

    st.make_filtration_non_decreasing()
    distorted_dgm = st.persistence()
    normal_dgm = dict()
    normal_dgm["Ord0"], normal_dgm["Rel1"], normal_dgm["Ext0"], normal_dgm["Ext1"] = [], [], [], []
    for point in range(len(distorted_dgm)):
        dim, b, d = distorted_dgm[point][0], distorted_dgm[point][1][0], distorted_dgm[point][1][1]
        pt_type = "unknown"
        if (-2 <= b <= -1 and -2 <= d <= -1) or (b == -.5 and d == -.5):
            pt_type = "Ord" + str(dim)
        if (1 <= b <= 2 and 1 <= d <= 2) or (b == .5 and d == .5):
            pt_type = "Rel" + str(dim)
        if (-2 <= b <= -1 and 1 <= d <= 2) or (b == -.5 and d == .5):
            pt_type = "Ext" + str(dim)
        if np.isinf(d):
            continue
        else:
            b, d = min_val + (2 - abs(b)) * (max_val - min_val), min_val + (2 - abs(d)) * (max_val - min_val)
            if b <= d:
                normal_dgm[pt_type].append(tuple([distorted_dgm[point][0], tuple([b, d])]))
            else:
                normal_dgm[pt_type].append(tuple([distorted_dgm[point][0], tuple([d, b])]))

    dgmOrd0 = np.array([normal_dgm["Ord0"][point][1] for point in range(len(normal_dgm["Ord0"]))])
    dgmExt0 = np.array([normal_dgm["Ext0"][point][1] for point in range(len(normal_dgm["Ext0"]))])
    dgmRel1 = np.array([normal_dgm["Rel1"][point][1] for point in range(len(normal_dgm["Rel1"]))])
    dgmExt1 = np.array([normal_dgm["Ext1"][point][1] for point in range(len(normal_dgm["Ext1"]))])
    if dgmOrd0.shape[0] == 0:
        dgmOrd0 = np.zeros([0, 2])
    if dgmExt1.shape[0] == 0:
        dgmExt1 = np.zeros([0, 2])
    if dgmExt0.shape[0] == 0:
        dgmExt0 = np.zeros([0, 2])
    if dgmRel1.shape[0] == 0:
        dgmRel1 = np.zeros([0, 2])
    return dgmOrd0, dgmExt0, dgmRel1, dgmExt1
