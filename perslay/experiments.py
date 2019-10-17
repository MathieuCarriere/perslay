"""Module :mod:`perslay.expe` provide experimental functions to run perslay."""

# Authors: Mathieu Carriere <mathieu.carriere3@gmail.com>
#          Theo Lacombe <theo.lacombe@inria.fr>
#          Martin Royer <martin.royer@inria.fr>
# License: MIT

import os.path
import itertools
import h5py

from ast import literal_eval

from scipy.sparse import csgraph
from scipy.io import loadmat, savemat
from scipy.linalg import eigh

import datetime

import numpy as np
import tensorflow as tf
import gudhi as gd

import matplotlib.pyplot as plt
import pandas as pd

from six.moves import xrange

from sklearn.preprocessing   import LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold, ShuffleSplit, GridSearchCV
from sklearn.pipeline        import Pipeline
from sklearn.svm             import SVC

from perslay.perslay       import perslay_channel
from perslay.preprocessing import preprocess
from perslay.utils         import diag_to_dict, hks_signature, get_base_simplex, apply_graph_extended_persistence

from tensorflow import random_uniform_initializer as rui





















class baseModel:

    def __init__(self, filt_parameters, perslay_parameters, labels, combination=False): 
        self.filt_parameters = filt_parameters
        self.perslay_parameters = perslay_parameters
        self.num_labels = labels.shape[1]
        self.num_filts = len(self.filt_parameters["names"])
        self.combination = combination
        
    def get_parameters(self):
        return [self.filt_parameters, self.perslay_parameters, self.combination]

    def instance(self, indxs, feats, diags):
        
        if self.filt_parameters["learn"]:

            lpd = tf.load_op_library("persistence_diagram.so")
            hks = tf.load_op_library("hks.so")
            import _persistence_diagram_grad
            import _hks_grad

            H, T = np.array(self.filt_parameters["homology"]), np.array(self.filt_parameters["thresholds"])
            N, I = np.array([[self.num_filts]]), np.array(self.filt_parameters["init"], dtype=np.float32)
            cumsum = np.cumsum(np.array([0] + [thr for thr in T[:,0]]))
            times = tf.get_variable("times", initializer=I)
            conn  = hks.heat_kernel_signature(indxs, times)
            pdiag_array, _ = lpd.persistence_diagram(H, T, indxs, N, conn)
            pds = tf.reshape(pdiag_array, [-1, cumsum[-1], 3])
            pdiags  = [pds[:,cumsum[i]:cumsum[i+1],:] for i in range(self.num_filts)] 

        else:
            pdiags = diags
            
        list_v = []
        
        if self.combination:

            n_pl = len(self.perslay_parameters)
            alpha = tf.get_variable("perslay_coeffs", initializer=np.array(np.ones(n_pl), dtype=np.float32))

            for i in range(self.num_filts):
            # A perslay channel must be defined for each type of persistence diagram. 
            # Here it is a linear combination of several pre-defined layers.

                list_dgm = []
                for prm in range(n_pl):
                    perslay_channel(output  =  list_dgm,              # list used to store all outputs
                                    name    =  "perslay-" + str(i),   # name of this layer
                                    diag    =  pdiags[i],             # i-th type of diagrams
                                    **self.perslay_parameters[prm])
            
                list_dgm = [tf.multiply(alpha[idx], tf.layers.batch_normalization(dgm)) 
                        for idx, dgm in enumerate(list_dgm)]
                list_v.append(tf.math.add_n(list_dgm))
        else:
            if type(self.perslay_parameters) is not list:
                for i in range(self.num_filts):
                # A perslay channel must be defined for each type of persistence diagram. 
                # Here they all have the same hyper-parameters.
                    perslay_channel(output  =  list_v,              # list used to store all outputs
                                    name    =  "perslay-" + str(i), # name of this layer
                                    diag    =  pdiags[i],           # i-th type of diagrams
                                    **self.perslay_parameters)
            else:
                for i in range(self.num_filts):
                # A perslay channel must be defined for each type of persistence diagram. 
                # Here they all have the same hyper-parameters.
                    perslay_channel(output  =  list_v,              # list used to store all outputs
                                    name    =  "perslay-" + str(i), # name of this layer
                                    diag    =  pdiags[i],           # i-th type of diagrams
                                    **self.perslay_parameters[i])

        # Concatenate all channels and add other features
        with tf.variable_scope("perslay"):
            representations = tf.concat(list_v, 1)
        with tf.variable_scope("norm_feat"):
            feat = tf.layers.batch_normalization(feats)

        final_representations = tf.concat([representations, feat], 1)

        #  Final layer to make predictions
        with tf.variable_scope("final-dense"):
            logits = tf.layers.dense(final_representations, self.num_labels)

        return representations, logits





















def load_config(filepath):
    with open(filepath, "r") as fp:
        lines = fp.readlines()
        dataset_type = lines[0][:-1]
        filt_parameters = literal_eval(lines[1])
        perslay_parameters = literal_eval(lines[2])
        combs = literal_eval(lines[3])
        optim_parameters = literal_eval(lines[4])
        for k in perslay_parameters.keys():
            if k[-4:] == "init":
                a, b = perslay_parameters[k][0], perslay_parameters[k][1]
                perslay_parameters[k] = rui(a, b) 
    return dataset_type, filt_parameters, perslay_parameters, combs, optim_parameters


# filtrations and features generation for datasets in the paper
def generate_diag_and_features(dataset, path_dataset=""):
    path_dataset = "./data/" + dataset + "/" if not len(path_dataset) else path_dataset
    filepath = path_dataset + dataset + ".conf"
    dataset_type, filt_parameters, thresh, perslay_parameters, optim_parameters = load_config(filepath=filepath)

    if "REDDIT" in dataset:
        print("Unfortunately, REDDIT data are not available yet for memory issues.\n")
        print("Moreover, the link we used to download the data,")
        print("http://www.mit.edu/~pinary/kdd/datasets.tar.gz")
        print("is down at the commit time (May 23rd).")
        print("We will update this repository when we figure out a workaround.")
        return
    # if "REDDIT" in dataset:
    #     _prepreprocess_reddit(dataset)
    if os.path.isfile(path_dataset + dataset + ".hdf5"):
        os.remove(path_dataset + dataset + ".hdf5")
    diag_file = h5py.File(path_dataset + dataset + ".hdf5")
    list_filtrations = filt_parameters["names"]
    [diag_file.create_group(str(filtration)) for filtration in filt_parameters["names"]]
    list_hks_times = np.unique([filtration.split("_")[1] for filtration in list_filtrations])
    if dataset_type == "graph":

        # preprocessing
        pad_size = 1
        for graph_name in os.listdir(path_dataset + "mat/"):
            A = np.array(loadmat(path_dataset + "mat/" + graph_name)["A"], dtype=np.float32)
            pad_size = np.max((A.shape[0], pad_size))

        features = pd.DataFrame(index=range(len(os.listdir(path_dataset + "mat/"))), columns=["label"] + ["eval" + str(i) for i in range(pad_size)] + [name + "-percent" + str(i) for name, i in itertools.product([f for f in list_hks_times if "hks" in f], 10 * np.arange(11))])

        for idx, graph_name in enumerate((os.listdir(path_dataset + "mat/"))):
            name = graph_name.split("_")
            gid = int(name[name.index("gid") + 1]) - 1
            A = np.array(loadmat(path_dataset + "mat/" + graph_name)["A"], dtype=np.float32)
            num_vertices = A.shape[0]
            label = int(name[name.index("lb") + 1])
            L = csgraph.laplacian(A, normed=True)
            egvals, egvectors = eigh(L)
            basesimplex = get_base_simplex(A)

            eigenvectors = np.zeros([num_vertices, pad_size])
            eigenvals = np.zeros(pad_size)
            eigenvals[:min(pad_size, num_vertices)] = np.flipud(egvals)[:min(pad_size, num_vertices)]
            eigenvectors[:, :min(pad_size, num_vertices)] = np.fliplr(egvectors)[:, :min(pad_size, num_vertices)]
            graph_features = []
            graph_features.append(eigenvals)

            for fhks in list_hks_times:
                hks_time = float(fhks.split("-")[0])
                # persistence
                filtration_val = hks_signature(egvectors, egvals, time=hks_time)
                dgmOrd0, dgmExt0, dgmRel1, dgmExt1 = apply_graph_extended_persistence(A, filtration_val, basesimplex)
                diag_file["Ord0_" + str(hks_time) + "-hks"].create_dataset(name=str(gid), data=dgmOrd0)
                diag_file["Ext0_" + str(hks_time) + "-hks"].create_dataset(name=str(gid), data=dgmExt0)
                diag_file["Rel1_" + str(hks_time) + "-hks"].create_dataset(name=str(gid), data=dgmRel1)
                diag_file["Ext1_" + str(hks_time) + "-hks"].create_dataset(name=str(gid), data=dgmExt1)
                # features
                graph_features.append(np.percentile(hks_signature(eigenvectors, eigenvals, time=hks_time), 10 * np.arange(11)))
            features.loc[gid] = np.insert(np.concatenate(graph_features), 0, label)
        features['label'] = features['label'].astype(int)

    elif dataset_type == "orbit":
        def _gen_orbit(num_pts_per_orbit, param):
            X = np.zeros([num_pts_per_orbit, 2])
            xcur, ycur = np.random.rand(), np.random.rand()
            for idx in range(num_pts_per_orbit):
                xcur = (xcur + param * ycur * (1. - ycur)) % 1
                ycur = (ycur + param * xcur * (1. - xcur)) % 1
                X[idx, :] = [xcur, ycur]
            return X

        labs = []
        count = 0
        num_diag_per_param = 1000 if "5K" in dataset else 20000
        for lab, r in enumerate([2.5, 3.5, 4.0, 4.1, 4.3]):
            print("Generating", num_diag_per_param, "orbits and diagrams for r = ", r, "...")
            for dg in range(num_diag_per_param):
                X = _gen_orbit(num_pts_per_orbit=1000, param=r)
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


# notebook utils
def load_diagfeatlabels(dataset, path_dataset="", verbose=False):
    path_dataset = "./data/" + dataset + "/" if not len(path_dataset) else path_dataset
    diagfile = h5py.File(path_dataset + dataset + ".hdf5", "r")
    filts = list(diagfile.keys())
    feat = pd.read_csv(path_dataset + dataset + ".csv", index_col=0, header=0)
    diag = diag_to_dict(diagfile, filts=filts)

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


# learning utils
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





















def _evaluate_nn_model(LB, FT, DG, train_sub, test_sub, model, optim_parameters, verbose=True):

    num_tower, tower_type, num_epochs, decay, learning_rate, tower_size, optimizer = optim_parameters["num_tower"], optim_parameters["tower_type"], optim_parameters["num_epochs"], optim_parameters["decay"], optim_parameters["learning_rate"], optim_parameters["tower_size"], optim_parameters["optimizer"]

    tf.reset_default_graph()

    with tf.device("/cpu:0"):

        num_pts, num_labels, num_features, num_filt = LB.shape[0], LB.shape[1], FT.shape[1], len(DG)

        # Neural network input
        indxs = tf.placeholder(shape=[None, 1], dtype=tf.int32)
        label = tf.placeholder(shape=[None, num_labels], dtype=tf.float32)
        feats = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
        diags = [tf.placeholder(shape=[None, DG[dt].shape[1], DG[dt].shape[2]], dtype=tf.float32) for dt in range(num_filt)]

        # Optimizer
        gs = tf.Variable(0, trainable=False)
        if decay > 0:
            decay_steps, decay_rate, staircase = optim_parameters["decay_steps"], optim_parameters["decay_rate"], optim_parameters["staircase"]
            lr = tf.train.exponential_decay(learning_rate=learning_rate, global_step=gs, decay_steps=decay_steps, decay_rate=decay_rate, staircase=staircase)
        else:
            lr = learning_rate
        if optimizer == "adam":
            epsilon = optim_parameters["epsilon"]
            opt = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)
        elif optimizer == "gradient_descent":
            opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
        elif optimizer == "rmsprop":
            opt = tf.train.RMSPropOptimizer(learning_rate=lr)

        sp_indxs = tf.split(indxs, num_or_size_splits=num_tower, axis=0)
        sp_label = tf.split(label, num_or_size_splits=num_tower, axis=0)
        sp_feats = tf.split(feats, num_or_size_splits=num_tower, axis=0)
        sp_diags = [tf.split(diags[dt], num_or_size_splits=num_tower, axis=0) for dt in range(num_filt)]

        # Neural network is built by placing a graph on each computing unit (tower)
        # Calculate the gradients for each model tower
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            accuracy = 0
            for i in xrange(num_tower):
                with tf.device("/" + tower_type + ":" + str(i)):
                    with tf.name_scope("tower_" + str(i)):  # as scope:
                        # Get split corresponding to tower
                        tow_indxs, tow_label, tow_feats, tow_diags = sp_indxs[i], sp_label[i], sp_feats[i], [
                            sp_diags[dt][i] for dt in range(num_filt)]

                        # Apply model
                        representations, tow_logit = model.instance(tow_indxs, tow_feats, tow_diags)

                        # Compute train loss and accuracy on this tower
                        tow_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tow_logit, 1), tf.argmax(tow_label, 1)), dtype=tf.float32))
                        tow_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tow_label, logits=tow_logit))

                        # for v in tf.trainable_variables():
                        #    tow_loss += tf.nn.l2_loss(v)

                        accuracy += tow_acc * (1 / num_tower)
                        tf.get_variable_scope().reuse_variables()

                        # Calculate the gradients for the batch of data on this tower
                        grads = opt.compute_gradients(tow_loss)

                        # Keep track of the gradients across all towers
                        tower_grads.append(grads)

        # Calculate the mean of each gradient, this is the synchronization point across all towers
        grads = []

        # Each grad_and_vars looks like the following: ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        for grad_and_vars in zip(*tower_grads):
            gr = []
            for g, _ in grad_and_vars:

                # Add 0 dimension to the gradients to represent the tower
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below
                gr.append(expanded_g)

            # Average over the 'tower' dimension
            grad = tf.reduce_mean(tf.concat(axis=0, values=gr), 0)

            # Keep in mind that the Variables are redundant because they are shared across towers,
            # so we just return the first tower's pointer to the Variable
            grads.append((grad, grad_and_vars[0][1]))

        # Apply the gradients to adjust the shared variables
        apply_gradient_op = opt.apply_gradients(grads, global_step=None)
        increase_global_step = gs.assign_add(1)

        model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        is_training = tf.get_variable("is_training", shape=(), dtype=tf.bool,
                                      initializer=tf.constant_initializer(True, dtype=tf.bool))

        # Create EMA object and update internal variables after optimization step
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        with tf.control_dependencies([apply_gradient_op]):
            train_op = ema.apply(model_vars)

        # Create backup for trainable variables
        with tf.variable_scope("BackupVariables"):
            backup_vars = [tf.get_variable(var.op.name, dtype=var.value().dtype, trainable=False,
                                           initializer=var.initialized_value()) for var in model_vars]

        def to_training():
            tf.assign(is_training, True)
            return tf.group(*(tf.assign(var, bck.read_value()) for var, bck in zip(model_vars, backup_vars)))

        def to_testing():
            tf.assign(is_training, False)
            tf.group(*(tf.assign(bck, var.read_value()) for var, bck in zip(model_vars, backup_vars)))
            return tf.group(*(tf.assign(var, ema.average(var).read_value()) for var in model_vars))

        switch_to_train_mode_op = tf.cond(is_training, true_fn=lambda: tf.group(), false_fn=to_training)
        switch_to_test_mode_op = tf.cond(is_training, true_fn=to_testing, false_fn=lambda: tf.group())

    # Create train and test indices
    train_sub = train_sub[:len(train_sub) - (len(train_sub) % num_tower)]
    test_sub = test_sub[:len(test_sub) - (len(test_sub) % num_tower)]
    train_num_pts, test_num_pts, = len(train_sub), len(test_sub)

    # Create train and test input dictionaries for Tensorflow
    feed_train, feed_test = dict(), dict()
    feed_train[indxs], feed_test[indxs] = train_sub[:, np.newaxis], test_sub[:, np.newaxis]
    feed_train[label], feed_test[label] = LB[train_sub, :], LB[test_sub, :]
    feed_train[feats], feed_test[feats] = FT[train_sub, :], FT[test_sub, :]

    for dt in range(num_filt):
        feed_train[diags[dt]], feed_test[diags[dt]] = DG[dt][train_sub, :], DG[dt][test_sub, :]

    # Create test batches
    test_batches = _create_batches(test_sub, feed_test, num_tower, tower_size, False)

    # Build an initialization operation to run below
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to True to build towers on GPU, since some of the ops do not have GPU implementations.
    # For GPU debugging, one may want to add in ConfigProto arguments: log_device_placement=True
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        # Initialize parameters
        sess.run(init)
        sess.run(switch_to_train_mode_op)

        weights, times = [[] for _ in range(model.num_filts)], []
        perslay_parameters = model.get_parameters()[1]
        
        if not model.get_parameters()[2]:
    
            for nf in range(model.num_filts):
                weight_fun = perslay_parameters["persistence_weight"] if type(perslay_parameters) == dict else perslay_parameters[nf]["persistence_weight"] 
                if weight_fun == "grid":
                    weights[nf].append(np.flip(sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "perslay-" + str(nf) + "-grid_pweight/W")[0]).T, 0))
                if weight_fun == "gmix":
                    means = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "perslay-" + str(nf) + "-gmix_pweight/M")[0])
                    varis = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "perslay-" + str(nf) + "-gmix_pweight/V")[0])
                    weights[nf].append((means,varis))                

        list_train_accs, list_test_accs = [], []

        # Training with optimization of parameters
        for epoch in xrange(num_epochs):

            # Create random train batches
            train_batches = _create_batches(train_sub, feed_train, num_tower, tower_size, True)

            # Apply gradient descent
            for feed_batch in train_batches:
                sess.run(train_op, feed_dict=feed_batch)
            sess.run(increase_global_step)

            if not model.get_parameters()[2]:
 
                # Retrieve weight matrices
                for nf in range(model.num_filts):
                    weight_fun = perslay_parameters["persistence_weight"] if type(perslay_parameters) == dict else perslay_parameters[nf]["persistence_weight"] 
                    if weight_fun == "grid":
                        weights[nf].append(np.flip(sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "perslay-" + str(nf) + "-grid_pweight/W")[0]).T, 0))
                    if weight_fun == "gmix":
                        means = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "perslay-" + str(nf) + "-gmix_pweight/M")[0])
                        varis = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "perslay-" + str(nf) + "-gmix_pweight/V")[0])
                        weights[nf].append((means,varis))

                # Retrieve times
                if model.get_parameters()[0]["learn"]:
                    times.append(np.array(sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "times")[0])))


            # Switch to test mode and evaluate train and test accuracy
            sess.run(switch_to_test_mode_op)
            train_acc, test_acc = 0, 0
            for feed_batch in train_batches:
                train_acc += 100 * accuracy.eval(feed_dict=feed_batch) * (feed_batch[label].shape[0] / train_num_pts)
            for feed_batch in test_batches:
                test_acc += 100 * accuracy.eval(feed_dict=feed_batch) * (feed_batch[label].shape[0] / test_num_pts)
            if (epoch+1) % 10 == 0 and verbose:
                print("Epoch: {:d}, train acc: {:04.1f}, test acc: {:04.1f}".format(epoch+1, train_acc, test_acc))
            list_train_accs.append(train_acc)
            list_test_accs.append(test_acc)

            # Go back to train mode
            sess.run(switch_to_train_mode_op)

            tr_repres = (representations.eval(feed_dict=feed_train), feed_train[label])
            te_repres = (representations.eval(feed_dict=feed_test),  feed_test[label])

    if model.get_parameters()[0]["learn"] and type(perslay_parameters) == dict:    
        times = np.concatenate(times, axis=1)
    
    return list_train_accs, list_test_accs, weights, times, [tr_repres, te_repres]





















def perform_expe(num_runs=1, path_dataset=None, dataset="custom",
    model=None, diags=[np.empty([0,0,0])], feats=np.empty([0,0]), labels=np.empty([0,0]),
    optim_parameters={}, perslay_cv=10, standard_model=False, standard_parameters=[], standard_cv=10, verbose=True):
    
    if path_dataset is not None:

        path_dataset = "./data/" + dataset + "/" if not len(path_dataset) else path_dataset
        filepath = path_dataset + dataset + ".conf"
        dataset_type, filt_parameters, perslay_parameters, combs, optim_parameters = load_config(filepath=filepath)
        thresh = filt_parameters["pad"]
        diag, feats, labels = load_diagfeatlabels(dataset, path_dataset=path_dataset, verbose=verbose)
        diags, _ = preprocess(diag, thresh=thresh)
        if type(filt_parameters) is not list and type(perslay_parameters) is not list:
            model = baseModel(filt_parameters, perslay_parameters, labels, combination=combs)
        else:
            model = []
            list_filt = filt_parameters if type(filt_parameters) == list else [filt_parameters]
            list_pers = perslay_parameters if type(perslay_parameters) == list else [perslay_parameters]
            list_comb = combs if type(perslay_parameters) == list else [combs]
            for fi in list_filt:
                for idx, pe in enumerate(list_pers):
                    model.append(baseModel(fi, pe, labels, combination=combs[idx]))

    mode, num_folds, num_epochs = optim_parameters["mode"], optim_parameters["folds"], optim_parameters["num_epochs"]

    # Train and test data.
    train_accs_res = np.zeros([num_runs, num_folds, num_epochs]) if not standard_model else np.zeros([num_runs, num_folds, num_epochs+1])
    test_accs_res = np.zeros([num_runs, num_folds, num_epochs]) if not standard_model else np.zeros([num_runs, num_folds, num_epochs+1])

    for idx_score in range(num_runs):

        print("Run number %i" % (idx_score+1))
        print("*************")
        if mode == "KF":  # Evaluation with k-fold on test set
            folds = KFold(n_splits=num_folds, random_state=idx_score, shuffle=True).split(np.empty([feats.shape[0]]))
        if mode == "RP":  # Evaluation with random test set
            test_size = optim_parameters["test_size"]
            folds = ShuffleSplit(n_splits=num_folds, test_size=test_size, random_state=idx_score).split(np.empty([feats.shape[0]]))

        for idx, (train_sub, test_sub) in enumerate(folds):

            print("Run number %i -- fold %i" % (idx_score+1, idx+1))
            print(str(len(train_sub)) + " train points and " + str(len(test_sub)) + " test points")

            # Evaluation of neural network
            if type(model) is not list and type(optim_parameters) is not list:
                best_model, best_optim = model, optim_parameters
            else:
                list_model = model if type(model) == list else [model]
                list_optim = optim_parameters if type(optim_parameters) == list else [optim_parameters]
                best_model, best_avg, best_optim = list_model[0], 0., list_optim[0]
                for mdl in list_model:
                    for opt in list_optim:
                        avg_acc = 0.
                        folds_inner = KFold(n_splits=perslay_cv, random_state=idx+1, shuffle=True).split(np.empty([len(train_sub)]))
                        for _, (train_param, valid_param) in enumerate(folds_inner):
                            _, te, _, _, _ = _evaluate_nn_model(labels, feats, diags, train_sub[train_param], train_sub[valid_param], mdl, opt, verbose=False)
                            avg_acc += te[-1] / perslay_cv
                        if avg_acc > best_avg:
                            best_model, best_avg, best_optim = mdl, avg_acc, opt

            ltrain, ltest, _, _, vecs = _evaluate_nn_model(labels, feats, diags, train_sub, test_sub, best_model, best_optim, verbose)
            
            if standard_model:
                tr_vectors, te_vectors = vecs[0][0], vecs[1][0]
                tr_labels,  te_labels  = np.array([np.where(vecs[0][1][i,:]==1)[0][0] for i in range(len(tr_vectors))]), np.array([np.where(vecs[1][1][i,:]==1)[0][0] for i in range(len(te_vectors))])
                pipe  = Pipeline([("Estimator", SVC())])
                std_model = GridSearchCV(pipe, standard_parameters, cv=standard_cv)
                std_model = std_model.fit(tr_vectors, tr_labels)
                ltrain.append(100 * std_model.score(tr_vectors, tr_labels))
                ltest.append(100 * std_model.score(te_vectors, te_labels))

            train_accs_res[idx_score, idx, :] = np.array(ltrain)
            test_accs_res[idx_score, idx, :] = np.array(ltest)

    filt_print = [m.get_parameters()[0] for m in model] if type(model) == list else model.get_parameters()[0]
    pers_print = [m.get_parameters()[1] for m in model] if type(model) == list else model.get_parameters()[1]
    comb_print = [m.get_parameters()[2] for m in model] if type(model) == list else model.get_parameters()[2]

    output = "./" if path_dataset is None else path_dataset
    with open(output + "summary.txt", "w") as text_file:
        text_file.write("DATASET: " + dataset + "\n")
        text_file.write(str(datetime.datetime.now()) + "\n\n")        
        text_file.write("****** " + str(num_runs) + " RUNS SUMMARY ******\n")
        text_file.write("Mode: " + mode + ", number of folds: " + str(num_folds) + "\n")
        text_file.write("Filtrations parameters: " + str(filt_print) + "\n")
        text_file.write("PersLay parameters: " + str(pers_print) + "\n")
        text_file.write("Linear combinations: " + str(comb_print) + "\n")
        text_file.write("Optimization parameters: " + str(optim_parameters) + "\n")
        if standard_model:
            text_file.write("Standard classifiers: " + str(standard_parameters) + "\n")

        folders_means = np.mean(test_accs_res, axis=1)
        overall_best_epoch = np.argmax(np.mean(folders_means, axis=0))
        final_means = folders_means[:, -1]
        best_means = folders_means[:, overall_best_epoch]

        text_file.write("Mean: " + str(np.round(np.mean(final_means), 2)) + "% +/- " + str(np.round(np.std(final_means), 2)) + "%\n")
        text_file.write("Best mean: " + str(np.round(np.mean(best_means), 2)) + "% +/- " + str(np.round(np.std(best_means), 2)) + "%, reached at epoch " + str(overall_best_epoch + 1))

        print("Mean: " + str(np.round(np.mean(final_means), 2)) + "% +/- " + str(np.round(np.std(final_means), 2)) + "%")
        print("Best mean: " + str(np.round(np.mean(best_means), 2)) + "% +/- " + str(np.round(np.std(best_means), 2)) + "%, reached at epoch " + str(overall_best_epoch + 1))

    np.save(output + "train_accs.npy", train_accs_res)
    np.save(output + "test_accs.npy", train_accs_res)

    return





















def single_run(test_size, path_dataset=None, dataset="custom",
               model=None, diags=[np.empty([0,0,0])], feats=np.empty([0,0]), labels=np.empty([0,0]),
               optim_parameters={},
               perslay_cv=None, standard_model=False, standard_parameters=[], standard_cv=10, 
               visualize_weights_times=False, verbose=True,
               **kwargs):

    if path_dataset is not None:

        path_dataset = "./data/" + dataset + "/" if not len(path_dataset) else path_dataset
        filepath = path_dataset + dataset + ".conf"
        dataset_type, filt_parameters, perslay_parameters, combs, optim_parameters = load_config(filepath=filepath)
        diag, feats, labels = load_diagfeatlabels(dataset, path_dataset=path_dataset, verbose=verbose)
        thresh = filt_parameters["pad"]
        diags, _ = preprocess(diag, thresh=thresh)
        if type(filt_parameters) is not list and type(perslay_parameters) is not list:
            model = baseModel(filt_parameters, perslay_parameters, labels, combination=combs)
        else:
            model = []
            list_filt = filt_parameters if type(filt_parameters) == list else [filt_parameters]
            list_pers = perslay_parameters if type(perslay_parameters) == list else [perslay_parameters]
            list_comb = combs if type(perslay_parameters) == list else [combs]
            for fi in list_filt:
                for idx, pe in enumerate(list_pers):
                    model.append(baseModel(fi, pe, labels, combination=combs[idx]))

    filt_print = [m.get_parameters()[0] for m in model] if type(model) == list else model.get_parameters()[0]
    pers_print = [m.get_parameters()[1] for m in model] if type(model) == list else model.get_parameters()[1]
    comb_print = [m.get_parameters()[2] for m in model] if type(model) == list else model.get_parameters()[2]
    print("Filtration parameters:", filt_print)
    print("PersLay parameters:", pers_print)
    print("Linear combinations:", comb_print)
    print("Optimization parameters:", optim_parameters)
    if standard_model:
        print("Standard classifiers:", standard_parameters)

    # Train and test data.
    folds = ShuffleSplit(n_splits=1, test_size=test_size).split(np.empty([feats.shape[0]]))

    for idx, (train_sub, test_sub) in enumerate(folds):

        print(str(len(train_sub)) + " train points and " + str(len(test_sub)) + " test points")

        # Evaluation of neural network
        if type(model) is not list and type(optim_parameters) is not list:
            best_model, best_optim = model, optim_parameters
        else:
            list_model = model if type(model) == list else [model]
            list_optim = optim_parameters if type(optim_parameters) == list else [optim_parameters]
            best_model, best_avg, best_optim = list_model[0], 0., list_optim[0]
            for mdl in list_model:
                for opt in list_optim:
                    avg_acc = 0.
                    folds_inner = KFold(n_splits=perslay_cv, random_state=42, shuffle=True).split(np.empty([len(train_sub)]))
                    for _, (train_param, valid_param) in enumerate(folds_inner):
                        _, te, _, _, _ = _evaluate_nn_model(labels, feats, diags, train_sub[train_param], train_sub[valid_param], mdl, opt, verbose=False)
                        avg_acc += te[-1] / perslay_cv
                    if avg_acc > best_avg:
                        best_model, best_avg, best_optim = mdl, avg_acc, opt

        if type(model) is list:
            print("Best model:", best_model)
        if type(optim_parameters) is list:
            print("Best optim:", best_optim)

        ltrain, ltest, weights, times, vecs = _evaluate_nn_model(labels, feats, diags, train_sub, test_sub, best_model, best_optim, verbose=True)
        if standard_model:
            tr_vectors, te_vectors = vecs[0][0], vecs[1][0]
            tr_labels,  te_labels  = np.array([np.where(vecs[0][1][i,:]==1)[0][0] for i in range(len(tr_vectors))]), np.array([np.where(vecs[1][1][i,:]==1)[0][0] for i in range(len(te_vectors))])
            pipe  = Pipeline([("Estimator", SVC())])
            model = GridSearchCV(pipe, standard_parameters, cv=standard_cv)
            model = model.fit(tr_vectors, tr_labels)
            print("Best standard classifier:", model.best_params_)
            tracc, teacc = 100 * model.score(tr_vectors, tr_labels), 100 * model.score(te_vectors, te_labels)
            ltrain.append(tracc)
            ltest.append(teacc)
            print("train acc: " + str(tracc) + ", test acc: " + str(teacc))
            

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.array(ltrain), color="blue", label="train acc")
        ax.plot(np.array(ltest), color="red", label="test acc")
        ax.set_ylim(top=100)
        ax.legend()
        ax.set_xlabel("epochs")
        ax.set_ylabel("classif. accuracy")
        ax.set_title("Evolution of train/test accuracy")
        plt.show()

        list_filtrations = best_model.get_parameters()[0]["names"]
        if visualize_weights_times and not best_model.get_parameters()[2]:
            
            fig = plt.figure(figsize=(10,20))

            for nf, filt in enumerate(list_filtrations):

                weight_fun = best_model.get_parameters()[1]["persistence_weight"] if type(best_model.get_parameters()[1]) is not list else best_model.get_parameters()[1][nf]["persistence_weight"]

                if weight_fun == "grid":
                
                    plt.subplot(best_model.num_filts, 2, 2*nf+1)
                    plt.imshow(weights[nf][0], cmap="Purples",  vmin=kwargs["xmin"], vmax=kwargs["xmax"])
                    plt.title(filt + " -- before training")
                    plt.colorbar()

                    plt.subplot(best_model.num_filts,2,2*(nf+1))
                    plt.imshow(weights[nf][-1], cmap="Purples", vmin=kwargs["xmin"], vmax=kwargs["xmax"])
                    plt.title(filt + " -- after training")
                    plt.colorbar()

                if weight_fun == "gmix":

                    x = np.arange(kwargs["xmin"], kwargs["xmax"], kwargs["xstep"])
                    y = np.arange(kwargs["ymin"], kwargs["ymax"], kwargs["ystep"])
                    xx, yy = np.meshgrid(x, y)

                    ax = fig.add_subplot(best_model.num_filts,2,2*nf+1)
                    means, varis = weights[nf][0][0], weights[nf][0][1]
                    z = np.zeros(xx.shape)
                    for idx_g in range(means.shape[3]):
                        z += np.exp(-((xx-means[0,0,0,idx_g])**2/(varis[0,0,0,idx_g]) + (yy-means[0,0,1,idx_g])**2/(varis[0,0,1,idx_g])))
                    ax.contourf(xx, yy, z)
                    ax.title.set_text(filt + " -- before training")

                    ax = fig.add_subplot(best_model.num_filts,2,2*(nf+1))
                    means, varis = weights[nf][-1][0], weights[nf][-1][1]
                    z = np.zeros(xx.shape)
                    for idx_g in range(means.shape[3]):
                        z += np.exp(-((xx-means[0,0,0,idx_g])**2/(varis[0,0,0,idx_g]) + (yy-means[0,0,1,idx_g])**2/(varis[0,0,1,idx_g])))
                    ax.contourf(xx, yy, z)
                    ax.title.set_text(filt + " -- after training")

            plt.show()

            if best_model.get_parameters()[0]["learn"]:
                fig = plt.figure()
                for nf, filt in enumerate(list_filtrations):
                    plt.subplot(1, len(list_filtrations), nf+1)
                    plt.plot(times[nf, :])
                plt.show()
            
    return weights, times
