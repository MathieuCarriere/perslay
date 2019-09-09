"""Module :mod:`perslay.utils` provide utils functions."""

# Authors: Mathieu Carriere <mathieu.carriere3@gmail.com>
#          Theo Lacombe <theo.lacombe@inria.fr>
#          Martin Royer <martin.royer@inria.fr>
# License: MIT
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

from scipy.sparse import csgraph
from scipy.io import loadmat
from scipy.linalg import eigh

import numpy as np
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt

from perslay.utils import get_base_simplex, hks_signature, apply_graph_extended_persistence


# visu helpers
def _Cantor_pairing(k1, k2):
    return int(k2 + (k1+k2)*(1+k1+k2)/2)


def _compute_mappings(D1, D2):
    try:
        from ot.bregman import sinkhorn
    except ModuleNotFoundError:
        print("POT not found")
        return

    n1, n2 = D1.shape[0], D2.shape[0]
    gamma = sinkhorn(a=(1/n1) * np.ones(n1), b=(1/n2) * np.ones(n2), M=pairwise_distances(D1, D2, metric="euclidean"), reg=1e-1 )
    mappings = [np.zeros(n1, dtype=np.int32), np.zeros(n2, dtype=np.int32)]
    for i in range(n1):
        mappings[0][i] = np.argmax(gamma[i,:])
    for i in range(n2):
        mappings[1][i] = np.argmax(gamma[:,i])
    return mappings


def generate_for_visu(dataset, list_times, dgm_type="Ord0", idx=0, path_out="./", path_in="./", bnds=[0.,1.,0.,1.]):
    path_dataset = path_in + dataset + "/"
    graph_name = os.listdir(path_dataset + "mat/")[idx]
    A = np.array(loadmat(path_dataset + "mat/" + graph_name)["A"], dtype=np.float32)
    L = csgraph.laplacian(A, normed=True)
    egvals, egvectors = eigh(L)
    basesimplex = get_base_simplex(A)

    LDgm, LOrd0, LExt1, LExt0, LRel1 = [], [], [], [], []
    for hks_time in list_times:
        filtration_val = hks_signature(egvectors, egvals, time=hks_time)
        dgmOrd0, dgmExt0, dgmRel1, dgmExt1 = apply_graph_extended_persistence(A, filtration_val, basesimplex)
        DD = np.array([[bnds[0], bnds[2]],[bnds[1], bnds[2]],[bnds[0], bnds[3]],[bnds[1], bnds[3]]])
        LOrd0.append(dgmOrd0)
        LRel1.append(dgmRel1)
        LExt1.append(dgmExt1)
        LExt0.append(dgmExt0)
        LDgm.append(np.vstack([dgmOrd0,dgmExt0,dgmExt1,dgmRel1,DD]))

    M = []
    for idx in range(len(list_times)-1):

        nOrd01, nOrd02 = len(LOrd0[idx]), len(LOrd0[idx+1])
        if nOrd01 > 0 and nOrd02 > 0:
            mappingsOrd0 = _compute_mappings(LOrd0[idx], LOrd0[idx + 1])
        nExt01, nExt02 = len(LExt0[idx]), len(LExt0[idx+1])
        if nExt01 > 0 and nExt02 > 0:
            mappingsExt0 = _compute_mappings(LExt0[idx], LExt0[idx + 1])
        nExt11, nExt12 = len(LExt1[idx]), len(LExt1[idx+1])
        if nExt11 > 0 and nExt12 > 0:
            mappingsExt1 = _compute_mappings(LExt1[idx], LExt1[idx + 1])
        nRel11, nRel12 = len(LRel1[idx]), len(LRel1[idx+1])
        if nRel11 > 0 and nRel12 > 0:
            mappingsRel1 = _compute_mappings(LRel1[idx], LRel1[idx + 1])

        M.append([[],[]])

        if nOrd01 > 0 and nOrd02 > 0:
            M[-1][0] += [(i, mappingsOrd0[0][i]) for i in range(nOrd01)]
            M[-1][1] += [(i, mappingsOrd0[1][i]) for i in range(nOrd02)]
        if nExt01 > 0 and nExt02 > 0:
            M[-1][0] += [(nOrd01 + i, nOrd02 + mappingsExt0[0][i]) for i in range(nExt01)]
            M[-1][1] += [(nOrd02 + i, nOrd01 + mappingsExt0[1][i]) for i in range(nExt02)]
        if nExt11 > 0 and nExt12 > 0:
            M[-1][0] += [(nOrd01 + nExt01 + i, nOrd02 + nExt02 + mappingsExt1[0][i]) for i in range(nExt11)]
            M[-1][1] += [(nOrd02 + nExt02 + i, nOrd01 + nExt01 + mappingsExt1[1][i]) for i in range(nExt12)]
        if nRel11 > 0 and nRel12 > 0:
            M[-1][0] += [(nOrd01 + nExt01 + nExt11 + i, nOrd02 + nExt02 + nExt12 + mappingsRel1[0][i]) for i in range(nRel11)]
            M[-1][1] += [(nOrd02 + nExt02 + nExt12 + i, nOrd01 + nExt01 + nExt11 + mappingsRel1[1][i]) for i in range(nRel12)]

        M[-1][0].append((nOrd01 + nExt01 + nExt11 + nRel11,       nOrd02 + nExt02 + nExt12 + nRel12))
        M[-1][0].append((nOrd01 + nExt01 + nExt11 + nRel11 + 1,   nOrd02 + nExt02 + nExt12 + nRel12 + 1))
        M[-1][0].append((nOrd01 + nExt01 + nExt11 + nRel11 + 2,   nOrd02 + nExt02 + nExt12 + nRel12 + 2))
        M[-1][0].append((nOrd01 + nExt01 + nExt11 + nRel11 + 3,   nOrd02 + nExt02 + nExt12 + nRel12 + 3))

    f_config  = open(path_out + "tower_config.txt",  "w")
    f_layers  = open(path_out + "tower_layers.txt",  "w")
    f_layout  = open(path_out + "tower_layout.txt",  "w")
    f_edges   = open(path_out + "tower_edges.txt",   "w")
    f_colors  = open(path_out + "tower_colors.txt",  "w")

    f_config.write("data/dgm_tower/tower_edges.txt;data/dgm_tower/tower_layers.txt;data/dgm_tower/tower_layout.txt;")
    f_layers.write("layerID layerLabel\n")
    f_layout.write("nodeID nodeLabel nodeX nodeY\n")
    f_colors.write("nodeID layerID color size\n")

    layer_names = [str(round(t,2)) for t in list_times]
    #layer_names = ["t" + str(idx) for idx in range(len(list_times))]
    node_name, layout, edges, colors = {}, {}, [], []
    node_ID = 1

    for idx, dgm in enumerate(LDgm):

        l_idx = idx + 1
        f_layers.write(str(l_idx) + " " + layer_names[idx] + "\n")

        for idx_pt in range(len(dgm)):

            nOrd0 = len(LOrd0[idx])
            nExt0 = len(LExt0[idx])
            nExt1 = len(LExt1[idx])
            nRel1 = len(LRel1[idx])

            Cantor_idx = _Cantor_pairing(idx_pt, idx)
            node_name[Cantor_idx] = node_ID

            if len(dgm) - idx_pt <= 4:
                if idx_pt == len(dgm)-1:
                    layout[node_ID] = ["n" + str(node_ID), str(2*dgm[idx_pt,0]), str(2*dgm[idx_pt,1])]
                if idx_pt == len(dgm)-2:
                    layout[node_ID] = ["n" + str(node_ID), str(dgm[idx_pt,0]), str(2*dgm[idx_pt,1])]
                if idx_pt == len(dgm)-3:
                    layout[node_ID] = ["n" + str(node_ID), str(2*dgm[idx_pt,0]),   str(dgm[idx_pt,1])]
                if idx_pt == len(dgm)-4:
                    layout[node_ID] = ["n" + str(node_ID), str(dgm[idx_pt,0]),   str(dgm[idx_pt,1])]
                cols = "\"#%02x%02x%02x\"" % (0,0,0)
                size = 0.01

            if idx_pt < nOrd0:
                layout[node_ID] = ["n" + str(node_ID), str(dgm[idx_pt,0]), str(dgm[idx_pt,1])]
                cols = "\"#%02x%02x%02x\"" % (128,0,0)
                size = 1

            if nOrd0 <= idx_pt < nOrd0 + nExt0:
                layout[node_ID] = ["n" + str(node_ID), str(bnds[1] + dgm[idx_pt,0]), str(dgm[idx_pt,1])]
                cols = "\"#%02x%02x%02x\"" % (0,128,0)
                size = 1

            if nOrd0 + nExt0 <= idx_pt < nOrd0 + nExt0 + nExt1:
                layout[node_ID] = ["n" + str(node_ID), str(dgm[idx_pt,0]), str(bnds[3] + dgm[idx_pt,1])]
                cols = "\"#%02x%02x%02x\"" % (0,0,128)
                size = 1

            if nOrd0 + nExt0 + nExt1 <= idx_pt < nOrd0 + nExt0 + nExt1 + nRel1:
                layout[node_ID] = ["n" + str(node_ID), str(bnds[1] + dgm[idx_pt,0]), str(bnds[3] + dgm[idx_pt,1])]
                cols = "\"#%02x%02x%02x\"" % (128,128,128)
                size = 1

            colors.append([str(node_ID), str(l_idx), str(cols), str(size)])
            node_ID += 1

    for idx in range(len(LDgm)-1):

        l_idx = idx + 1
        mapping = M[idx][0]
        for idxc, (p,q) in enumerate(mapping):
            edges.append([str(node_name[_Cantor_pairing(p, idx)]), str(l_idx), str(node_name[_Cantor_pairing(q, idx + 1)]), str(l_idx + 1), "1"])
        #mapping = M[idx][1]
        #for idxc, (q,p) in enumerate(mapping):
        #    edges.append([str(node_name[Cantor_pairing(p,idx)]), str(l_idx), str(node_name[Cantor_pairing(q,idx+1)]), str(l_idx+1), "1"])

    for key, line in iter(layout.items()):
        f_layout.write(" ".join([str(key)] + line) + "\n")

    for line in edges:
        f_edges.write(" ".join(line) + "\n")

    for line in colors:	
        f_colors.write(" ".join(line) + "\n")

    f_config.close()
    f_layers.close()
    f_layout.close()
    f_edges.close()
    f_colors.close()
    return


def visualise_diag(diag, ilist=(0, 10, 20, 30, 40, 50)):
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
