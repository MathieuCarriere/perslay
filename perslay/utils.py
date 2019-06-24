"""Module :mod:`perslay.utils` provide utils functions."""

# Authors: Mathieu Carriere <mathieu.carriere3@gmail.com>
#          Theo Lacombe <theo.lacombe@inria.fr>
#          Martin Royer <martin.royer@inria.fr>
# License: MIT
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path


from ast import literal_eval
import itertools

import numpy as np

import matplotlib.pyplot as plt

import h5py

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from scipy.sparse import csgraph
from scipy.io import loadmat, savemat
from scipy.linalg import eigh
import pandas as pd

import gudhi as gd


# diagrams utils
def _get_base_simplex(A):
    num_vertices = A.shape[0]
    st = gd.SimplexTree()
    for i in range(num_vertices):
        st.insert([i], filtration=-1e10)
        for j in range(i + 1, num_vertices):
            if A[i, j] > 0:
                st.insert([i, j], filtration=-1e10)
    return st.get_filtration()


# Input utility functions for persistence diagrams
def _diag_to_dict(diag_file, filts):
    out_dict = dict()
    if len(filts) == 0:
        filts = diag_file.keys()
    for filtration in filts:
        list_dgm, num_diag = [], len(diag_file[filtration].keys())
        for diag in range(num_diag):
            list_dgm.append(np.array(diag_file[filtration][str(diag)]))
        out_dict[filtration] = list_dgm
    return out_dict


# notebook utils
def _load_config(dataset):
    filepath = "./data/" + dataset + "/" + dataset + ".conf"
    with open(filepath) as fp:
        lines = fp.readlines()
        dataset_type = lines[0][:-1]
        list_filtrations = literal_eval(lines[1])
        thresh = int(lines[2])
        perslay_parameters = literal_eval(lines[3])
        optim_parameters = literal_eval(lines[4])
    return dataset_type, list_filtrations, thresh, perslay_parameters, optim_parameters


def load(dataset, verbose=False):
    # dataset_type, list_filtrations, thresh, perslay_parameters, optim_parameters = _load_config(dataset=dataset)
    path_dataset = "./data/" + dataset + "/"
    diagfile = h5py.File(path_dataset + dataset + ".hdf5", "r")
    filts = list(diagfile.keys())
    feat = pd.read_csv(path_dataset + dataset + ".csv", index_col=0, header=0)
    diag = _diag_to_dict(diagfile, filts=filts)

    # Extract and encode labels with integers
    L = np.array(LabelEncoder().fit_transform(np.array(feat["label"])))
    L = OneHotEncoder(sparse=False, categories="auto").fit_transform(L[:, np.newaxis])

    # Extract features
    F = np.array(feat)[:, 1:]  # 1: removes the labels

    if verbose:
        print("Dataset:", dataset)
        print("Number of observations:", L.shape[0])
        print("Number of classes:", L.shape[1])

    return diag, F, L


def _hks_signature(eigenvectors, eigenvals, time):
    return np.square(eigenvectors).dot(np.diag(np.exp(-time * eigenvals))).sum(axis=1)


def _apply_graph_extended_persistence(A, filtration_val, basesimplex):
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


def _save_matrix(A, gid, label, path):
    mat_name = "nodes_%i_edges_%i_gid_%i_lb_%i_index_1_adj.mat" % (A.shape[0], int(np.sum(A > 0)), gid, label)
    mat_file = {
        '__header__': 'PYTHON mimick MAT-file format',
        '__version__': 'nc',
        '__globals__': [],
        'A': A
    }
    return savemat(file_name=path + mat_name, mdict=mat_file)


def generate(dataset):

    if "REDDIT" in dataset:
        print("Unfortunately, REDDIT data are not available yet for memory issues.\n")
        print("Moreover, the link we used to download the data,")
        print("http://www.mit.edu/~pinary/kdd/datasets.tar.gz")
        print("is down at the commit time (May 23rd).")
        print("We will update this repository when we figure out a workaround.")
        return

    dataset_type, list_filtrations, thresh, perslay_parameters, optim_parameters = _load_config(dataset=dataset)
    path_dataset = "data/" + dataset + "/"
    if os.path.isfile(path_dataset + dataset + ".hdf5"):
        os.remove(path_dataset + dataset + ".hdf5")
    diag_file = h5py.File(path_dataset + dataset + ".hdf5")
    # if "REDDIT" in dataset:
    #     _prepreprocess_reddit(dataset)
    if dataset_type == "graph":
        [diag_file.create_group(filtration_type + "_" + str(filtration))
         for filtration, filtration_type in itertools.product(list_filtrations, ["Ord0", "Rel1", "Ext0", "Ext1"])]

        # preprocessing
        pad_size = 1
        for graph_name in os.listdir(path_dataset + "mat/"):
            A = np.array(loadmat(path_dataset + "mat/" + graph_name)["A"], dtype=np.float32)
            pad_size = np.max((A.shape[0], pad_size))

        features = pd.DataFrame(index=range(len(os.listdir(path_dataset + "mat/"))),
                                columns=["label"] +
                                        ["eval" + str(i) for i in range(pad_size)] +
                                        [name + "-percent" + str(i) for name, i in
                                         itertools.product([f for f in list_filtrations if "hks" in f],
                                                           10 * np.arange(11))])

        for idx, graph_name in enumerate((os.listdir(path_dataset + "mat/"))):
            name = graph_name.split("_")
            gid = int(name[name.index("gid") + 1]) - 1
            A = np.array(loadmat(path_dataset + "mat/" + graph_name)["A"], dtype=np.float32)
            num_vertices = A.shape[0]
            label = int(name[name.index("lb") + 1])
            L = csgraph.laplacian(A, normed=True)
            egvals, egvectors = eigh(L)
            basesimplex = _get_base_simplex(A)

            eigenvectors = np.zeros([num_vertices, pad_size])
            eigenvals = np.zeros(pad_size)
            eigenvals[:min(pad_size, num_vertices)] = np.flipud(egvals)[:min(pad_size, num_vertices)]
            eigenvectors[:, :min(pad_size, num_vertices)] = np.fliplr(egvectors)[:, :min(pad_size, num_vertices)]
            graph_features = []
            graph_features.append(eigenvals)

            for filtration in list_filtrations:
                # persistence
                hks_time = float(filtration.split("-")[0])
                filtration_val = _hks_signature(egvectors, egvals, time=hks_time)
                dgmOrd0, dgmExt0, dgmRel1, dgmExt1 = _apply_graph_extended_persistence(A, filtration_val, basesimplex)
                diag_file["Ord0_" + filtration].create_dataset(name=str(gid), data=dgmOrd0)
                diag_file["Ext0_" + filtration].create_dataset(name=str(gid), data=dgmExt0)
                diag_file["Rel1_" + filtration].create_dataset(name=str(gid), data=dgmRel1)
                diag_file["Ext1_" + filtration].create_dataset(name=str(gid), data=dgmExt1)
                # features
                graph_features.append(np.percentile(_hks_signature(eigenvectors, eigenvals, time=hks_time),
                                                    10*np.arange(11)))
            features.loc[gid] = np.insert(np.concatenate(graph_features), 0, label)
        features['label'] = features['label'].astype(int)

    elif dataset_type == "orbit":
        [diag_file.create_group(_) for _ in ["Alpha0", "Alpha1"]]
        labs = []
        count = 0
        num_diag_per_param = 1000 if "5K" in dataset else 20000
        num_pts_per_orbit = 1000
        for lab, r in enumerate([2.5, 3.5, 4.0, 4.1, 4.3]):
            print("Generating", num_diag_per_param, "orbits and diagrams for r = ", r, "...")
            for dg in range(num_diag_per_param):
                x0, y0 = np.random.rand(), np.random.rand()
                xcur, ycur = x0, y0
                X = np.zeros([num_pts_per_orbit, 2])
                X[0, :] = [x0, y0]
                for idx in range(num_pts_per_orbit - 1):
                    xcur += r * ycur * (1. - ycur)
                    xcur -= int(xcur)
                    ycur += r * xcur * (1. - xcur)
                    ycur -= int(ycur)
                    X[idx, :] = [xcur, ycur]

                alpha_complex = gd.AlphaComplex(points=X)
                simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=1e50)
                simplex_tree.persistence()
                diag_file["Alpha0"].create_dataset(name=str(count),
                                                   data=np.array(simplex_tree.persistence_intervals_in_dimension(0)))
                diag_file["Alpha1"].create_dataset(name=str(count),
                                                   data=np.array(simplex_tree.persistence_intervals_in_dimension(1)))
                orbit_label = {"label": lab, "pcid": count}
                labs.append(orbit_label)
                count += 1
        labels = pd.DataFrame(labs)
        labels.set_index("pcid")
        features = labels[["label"]]
    features.to_csv(path_dataset + dataset + ".csv")
    return diag_file.close()


# Batches
def _create_batches(indices, feed_dict, num_tower, tower_size, random=False):
    batch_size = num_tower * tower_size
    data_num_pts = len(indices)
    residual = data_num_pts % batch_size
    nbsplit = int((data_num_pts - residual) / batch_size)
    split = np.split(np.arange(data_num_pts - residual), nbsplit) if nbsplit > 0 else []
    # number_of_batches = nbsplit + min(residual, 1)
    if random:
        perm = np.random.permutation(data_num_pts)
    batches = []
    for i in range(nbsplit):
        feed_sub = dict()
        for k in feed_dict.keys():
            feed_sub[k] = feed_dict[k][perm[split[i]]] if random else feed_dict[k][split[i]]
        batches.append(feed_sub)
    if residual > 0:
        st, sz = data_num_pts - residual, residual - (residual % num_tower)
        feed_sub = dict()
        for k in feed_dict.keys():
            feed_sub[k] = feed_dict[k][perm[np.arange(st, st + sz)]] if random else feed_dict[k][np.arange(st, st + sz)]
        batches.append(feed_sub)
    return batches


def visualization(diag, ilist=(0, 10, 20, 30, 40, 50)):
    # path_dataset = "./data/" + dataset + "/"
    # diagfile = h5py.File(path_dataset + dataset + ".hdf5", "r")
    # filts = list(diagfile.keys())
    # diag = _diag_to_dict(diagfile, filts=filts)

    filts = diag.keys()

    n, m = len(filts), len(ilist)

    fig, axs = plt.subplots(n, m, figsize=(m*n / 2, n*m / 2))

    for (i, filtration) in enumerate(filts):
        for (j, idx) in enumerate(ilist):
            xs, ys = diag[filtration][idx][:, 0], diag[filtration][idx][:, 1]
            axs[i, j].scatter(xs, ys)
            axs[i, j].axis([0, 1, 0, 1])
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    # axis plot
    cols = ["idx = " + str(i) for i in ilist]
    rows = filts

    for ax, col in zip(axs[0], cols):
        ax.set_title(col)

    for ax, row in zip(axs[:, 0], rows):
        ax.set_ylabel(row, rotation=90, size='large')
    plt.show()
    return
