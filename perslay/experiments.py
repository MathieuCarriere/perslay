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

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold, ShuffleSplit

from perslay.preprocessing import preprocess
from perslay.utils import diag_to_dict, hks_signature, get_base_simplex, apply_graph_extended_persistence


def load_config(filepath):
    with open(filepath, "r") as fp:
        lines = fp.readlines()
        dataset_type = lines[0][:-1]
        list_filtrations = literal_eval(lines[1])
        thresh = int(lines[2])
        perslay_parameters = literal_eval(lines[3])
        optim_parameters = literal_eval(lines[4])
    return dataset_type, list_filtrations, thresh, perslay_parameters, optim_parameters


# filtrations and features generation for datasets in the paper
def generate_diag_and_features(dataset, path_dataset=""):
    path_dataset = "./data/" + dataset + "/" if not len(path_dataset) else path_dataset
    filepath = path_dataset + dataset + ".conf"
    dataset_type, list_filtrations, thresh, perslay_parameters, optim_parameters = load_config(filepath=filepath)

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

    if dataset_type == "graph":
        persist_types = ["Ord0", "Rel1", "Ext0", "Ext1"]
        [diag_file.create_group(persist_type + "_" + str(filtration))
         for filtration, persist_type in itertools.product(list_filtrations, persist_types)]

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
            basesimplex = get_base_simplex(A)

            eigenvectors = np.zeros([num_vertices, pad_size])
            eigenvals = np.zeros(pad_size)
            eigenvals[:min(pad_size, num_vertices)] = np.flipud(egvals)[:min(pad_size, num_vertices)]
            eigenvectors[:, :min(pad_size, num_vertices)] = np.fliplr(egvectors)[:, :min(pad_size, num_vertices)]
            graph_features = []
            graph_features.append(eigenvals)

            for filtration in list_filtrations:
                # persistence
                hks_time = float(filtration.split("-")[0])
                filtration_val = hks_signature(egvectors, egvals, time=hks_time)
                dgmOrd0, dgmExt0, dgmRel1, dgmExt1 = apply_graph_extended_persistence(A, filtration_val, basesimplex)
                diag_file["Ord0_" + filtration].create_dataset(name=str(gid), data=dgmOrd0)
                diag_file["Ext0_" + filtration].create_dataset(name=str(gid), data=dgmExt0)
                diag_file["Rel1_" + filtration].create_dataset(name=str(gid), data=dgmRel1)
                diag_file["Ext1_" + filtration].create_dataset(name=str(gid), data=dgmExt1)
                # features
                graph_features.append(np.percentile(hks_signature(eigenvectors, eigenvals, time=hks_time),
                                                    10 * np.arange(11)))
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

        [diag_file.create_group(_) for _ in ["Alpha0", "Alpha1"]]
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


def _evaluate_nn_model(LB, FT, DG,
                       train_sub, valid_sub, test_sub,
                       model,
                       num_tower, tower_type,
                       num_epochs, decay, learning_rate, tower_size,
                       verbose=True):

    with tf.device("/cpu:0"):
        num_pts, num_labels, num_features, num_filt = LB.shape[0], LB.shape[1], FT.shape[1], len(DG)
        # Neural network input
        indxs = tf.placeholder(shape=[None, 1], dtype=tf.int32)
        label = tf.placeholder(shape=[None, num_labels], dtype=tf.float32)
        feats = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
        diags = [tf.placeholder(shape=[None, DG[dt].shape[1], DG[dt].shape[2]], dtype=tf.float32) for dt in
                 range(num_filt)]

        # Optimizer
        gs = tf.Variable(0, trainable=False)
        # lr = tf.train.exponential_decay(learning_rate=learning_rate, global_step=gs, decay_steps=20, decay_rate=.5,
        #                                staircase=True)

        # opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-4)
        # opt = tf.train.GradientDescentOptimizer(learning_rate=lr)

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
                        tow_logit = model.instance(tow_feats, tow_diags)

                        # Compute train loss and accuracy on this tower
                        tow_acc = tf.reduce_mean(
                            tf.cast(tf.equal(tf.argmax(tow_logit, 1), tf.argmax(tow_label, 1)), dtype=tf.float32))
                        tow_loss = tf.reduce_mean(
                            tf.nn.softmax_cross_entropy_with_logits_v2(labels=tow_label, logits=tow_logit))
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

    # Create train, validation and test indices
    train_sub = train_sub[:len(train_sub) - (len(train_sub) % num_tower)]
    valid_sub = valid_sub[:len(valid_sub) - (len(valid_sub) % num_tower)]
    test_sub = test_sub[:len(test_sub) - (len(test_sub) % num_tower)]
    train_num_pts, test_num_pts, valid_num_pts = len(train_sub), len(test_sub), len(valid_sub)

    # Create train, validation and test input dictionaries for Tensorflow
    feed_train, feed_valid, feed_test = dict(), dict(), dict()
    feed_train[indxs] = train_sub[:, np.newaxis]
    feed_valid[indxs] = valid_sub[:, np.newaxis]
    feed_test[indxs] = test_sub[:, np.newaxis]
    feed_train[label], feed_valid[label], feed_test[label] = LB[train_sub, :], LB[valid_sub, :], LB[test_sub, :]
    feed_train[feats], feed_valid[feats], feed_test[feats] = FT[train_sub, :], FT[valid_sub, :], FT[test_sub, :]
    for dt in range(num_filt):
        feed_train[diags[dt]] = DG[dt][train_sub, :]
        feed_valid[diags[dt]] = DG[dt][valid_sub, :]
        feed_test[diags[dt]] = DG[dt][test_sub, :]

    # Create validation and test batches
    valid_batches = _create_batches(valid_sub, feed_valid, num_tower, tower_size, False)
    test_batches = _create_batches(test_sub, feed_test, num_tower, tower_size, False)

    # Build an initialization operation to run below
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    # allow_soft_placement must be set to True to build towers on GPU,
    # since some of the ops do not have GPU implementations.
    # For GPU debugging, one may want to add in ConfigProto arguments: log_device_placement=True
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        # Initialize parameters
        sess.run(init)
        sess.run(switch_to_train_mode_op)

        perslay_parameters = model.get_parameters()
        weight_fun = perslay_parameters["persistence_weight"]

        weights = [[] for _ in range(model.num_filts)]
        for nf in range(model.num_filts):
            if weight_fun == "grid":
                weights[nf].append(np.flip(sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "perslay-" + str(nf) + "-grid_pweight/W")[0]).T, 0))
            if weight_fun == "gmix":
                means = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "perslay-" + str(nf) + "-gmix_pweight/M")[0])
                varis = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "perslay-" + str(nf) + "-gmix_pweight/V")[0])
                weights[nf].append((means,varis))                

        list_train_accs, list_valid_accs, list_test_accs = [], [], []
        # Training with optimization of parameters
        for epoch in xrange(num_epochs):

            # Create random train batches
            train_batches = _create_batches(train_sub, feed_train, num_tower, tower_size, True)

            # Apply gradient descent
            for feed_batch in train_batches:
                sess.run(train_op, feed_dict=feed_batch)
            sess.run(increase_global_step)

            # Retrieve weight matrices
            for nf in range(model.num_filts):
                if weight_fun == "grid":
                    weights[nf].append(np.flip(sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "perslay-" + str(nf) + "-grid_pweight/W")[0]).T, 0))
                if weight_fun == "gmix":
                    means = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "perslay-" + str(nf) + "-gmix_pweight/M")[0])
                    varis = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "perslay-" + str(nf) + "-gmix_pweight/V")[0])
                    weights[nf].append((means,varis))
           
            # Switch to test mode and evaluate train and test accuracy
            sess.run(switch_to_test_mode_op)
            train_acc, valid_acc, test_acc = 0, 0, 0
            for feed_batch in train_batches:
                train_acc += 100 * accuracy.eval(feed_dict=feed_batch) * (feed_batch[label].shape[0] / train_num_pts)
            for feed_batch in valid_batches:
                valid_acc += 100 * accuracy.eval(feed_dict=feed_batch) * (feed_batch[label].shape[0] / valid_num_pts)
            for feed_batch in test_batches:
                test_acc += 100 * accuracy.eval(feed_dict=feed_batch) * (feed_batch[label].shape[0] / test_num_pts)
            if (epoch+1) % 10 == 0 and verbose:
                print("Epoch: {:d}, train acc: {:04.1f}, test acc: {:04.1f}".format(epoch+1, train_acc, test_acc))
            list_train_accs.append(train_acc)
            list_valid_accs.append(valid_acc)
            list_test_accs.append(test_acc)

            # Go back to train mode
            sess.run(switch_to_train_mode_op)

    return list_train_accs, list_valid_accs, list_test_accs, weights


def perform_expe(dataset, model, num_run=1, path_dataset=""):
    path_dataset = "./data/" + dataset + "/" if not len(path_dataset) else path_dataset
    filepath = path_dataset + dataset + ".conf"
    dataset_type, list_filtrations, thresh, perslay_parameters, optim_parameters = load_config(filepath=filepath)
    # Train and test data
    # In this subsection, we finally train and test the network on the data.
    #
    # Specify here how you want to train data and test data: either with K-folds ("KF")
    # or with random permutations of test set ("RP").

    mode, num_folds, test_size = "RP", 100, 0.3
    if dataset_type == "graph":
        mode = "KF"  # Either "KF" or "RP"
        num_folds = 10  # Number of splits
        test_size = 0.1  # Size of test set in case of "RP"
    elif dataset_type == "orbit":
        mode = "RP"  # Either "KF" or "RP"
        num_folds = 100  # Number of splits
        test_size = 0.3  # Size of test set in case of "RP"

    # Specify here if you have one or several GPUs or CPUs,
    # as well as number of epochs, batch size and validation size.
    # If you do not want to use validation sets for early stopping, set valid_size to 0.
    num_tower = 1  # Number of computing units
    tower_type = "gpu"  # Type of computing units ("cpu" or "gpu")
    batch_size = 128  # Batch size for each tower
    num_epochs = optim_parameters["num_epoch"]  # Number of epochs
    valid_size = 0.  # Size of validation set
    opt_mode = "adam"  # WARNING ! option not in use as of now
    withdiag = True  # use diagrams or not

    # Specify here the decay of Exponential Moving Average, the learning rate of optimizer and the verbose for training.
    decay = optim_parameters["decay"]  # Decay of Exponential Moving Average
    learn_rate = optim_parameters["lr"]  # Learning rate of optimizer
    verbose = True

    # load precalculated feats and diagrams and labels
    diag, feats, labels = load_diagfeatlabels(dataset, path_dataset=path_dataset, verbose=verbose)
    diags, filts = preprocess(diag)

    instance_model = model
    # partial(_model, parameters=perslay_parameters, num_filts=num_filts, num_labels=labels.shape[1], withdiag=withdiag)

    # Train and test data.
    train_accs_res = np.zeros([num_run, num_folds, num_epochs])
    valid_accs_res = np.zeros([num_run, num_folds, num_epochs])
    test_accs_res = np.zeros([num_run, num_folds, num_epochs])
    folds = None
    for idx_score in range(num_run):
        print("Run number %i" % (idx_score+1))
        if mode == "KF":  # Evaluation with k-fold on test set
            folds = KFold(n_splits=num_folds, random_state=idx_score, shuffle=True).split(np.empty([feats.shape[0]]))
        if mode == "RP":  # Evaluation with random test set
            folds = ShuffleSplit(n_splits=num_folds, test_size=test_size, random_state=idx_score).split(
                np.empty([feats.shape[0]]))

        for idx, (train_sub, test_sub) in enumerate(folds):
            print("Run number %i -- fold %i" % (idx_score+1, idx+1))
            valid_sub = train_sub[:int(valid_size * len(train_sub))]
            train_sub = train_sub[int(valid_size * len(train_sub)):]

            print(str(len(train_sub)) + " train points and " + str(len(test_sub)) + " test points")

            # Create neural network
            tf.reset_default_graph()

            # Evaluation of neural network
            ltrain, lvalid, ltest, weights = _evaluate_nn_model(labels, feats, diags, train_sub, valid_sub, test_sub,
                                                                instance_model, num_tower, tower_type, num_epochs,
                                                                decay, learn_rate, batch_size, verbose)
            train_accs_res[idx_score, idx, :] = np.array(ltrain)
            valid_accs_res[idx_score, idx, :] = np.array(lvalid)
            test_accs_res[idx_score, idx, :] = np.array(ltest)

    with open(path_dataset + "summary.txt", "w") as text_file:
        text_file.write(str(datetime.datetime.now()) + "\n\n")
        text_file.write("withdiag is set to:" + str(withdiag) + "\n")
        text_file.write("****** " + str(num_run) + " RUN SUMMARY ******\n")
        text_file.write("DATASET:")
        text_file.write(dataset + "\n")
        text_file.write("num_run:" + str(num_run) + "num folds:" + str(folds) + "\n")
        text_file.write("Filtrations used:\n")
        text_file.write(str(filts) + "\n")
        text_file.write("Preprocessing mode:" + mode + ", threshold:" + str(thresh) + ".\n")
        text_file.write("perslay architecture:\n")
        text_file.write(str(perslay_parameters) + "\n")
        text_file.write("num epochs:" + str(num_epochs) + "\n")
        text_file.write(
            "optimizer:" + opt_mode + ", learning_rate:" + str(learn_rate) + ", decay_rate:" + str(decay)
            + ", tower size:" + str(num_tower) + "\n")

        folders_means = np.mean(test_accs_res, axis=1)
        overall_best_epoch = np.argmax(np.mean(folders_means, axis=0))
        final_means = folders_means[:, -1]
        best_means = folders_means[:, overall_best_epoch]

        text_file.write("mean:" + str(np.round(np.mean(final_means), 2)) + "% +/-" + str(
            np.round(np.std(final_means), 2)) + "%\n")
        print("mean:" + str(np.round(np.mean(final_means), 2)) + "% +/-" + str(
            np.round(np.std(final_means), 2)) + "%\n")
        text_file.write("best mean:" + str(np.round(np.mean(best_means), 2)) + "% +/-" + str(
            np.round(np.std(best_means), 2)) + "%, reached at epoch" + str(overall_best_epoch + 1))
        print("best mean:" + str(np.round(np.mean(best_means), 2)) + "% +/-" + str(
            np.round(np.std(best_means), 2)) + "%, reached at epoch" + str(overall_best_epoch + 1))

    return


def _print_info(filtrations, thresh, perslay_parameters, optim_parameters):
    print("Filtrations used:")
    print(filtrations)
    print("Thresholding in diagrams:", thresh)

    print(" ***** PersLay parameters: *****")
    print("Layer:", perslay_parameters["layer"])
    layer = perslay_parameters["layer"]
    if layer == "im":
        print("  image size:", perslay_parameters["image_size"])
        print("  image boundaries:", perslay_parameters["image_bnds"])
        print("  convolution operations:", perslay_parameters["cv_layers"])
    elif layer == "pm":
        print("  permutation equivariant operations:", perslay_parameters["peq"])
        print("  fully-connected operations:", perslay_parameters["fc_layers"])
    elif layer == "ls":
        print("  number of samples:", perslay_parameters["num_samples"])
        print("  fully-connected operations:", perslay_parameters["fc_layers"])
    elif layer == "gs":
        print("  number of Gaussians:", perslay_parameters["num_gaussians"])
        print("  fully-connected operations:", perslay_parameters["fc_layers"])

    print("Weight function:", perslay_parameters["persistence_weight"])
    weight = perslay_parameters["persistence_weight"]
    if weight == "grid":
        print("  grid size:", perslay_parameters["grid_size"])
        print("  grid boundaries:", perslay_parameters["grid_bnds"])

    print("Permutation invariant operation:", perslay_parameters["perm_op"])
    pop = perslay_parameters["perm_op"]
    if pop == "topk":
        print("  number of largest values", perslay_parameters["keep"])

    print("***** Optimization parameters *****")
    print("Optimizer:", "ADAM")
    print("Number of epochs:", optim_parameters["num_epoch"])
    print("Learning rate:", optim_parameters["lr"])
    print("Decay:", optim_parameters["decay"])
    print("*" * 20)
    return


def single_run(diags, feats, labels,
               list_filtrations, model,
               optim_parameters,
               test_size,
               thresh=500,
               visualize_weight=False,
               **kwargs):
    # dataset_type, list_filtrations, thresh, perslay_parameters, optim_parameters = _load_config(dataset=dataset)
    num_folds = 1  # Number of splits

    perslay_parameters = model.get_parameters()

    _print_info(list_filtrations, thresh, perslay_parameters, optim_parameters)

    # Specify here if you have one or several GPUs or CPUs,
    # as well as number of epochs, batch size and validation size.
    # If you do not want to use validation sets for early stopping, set valid_size to 0.
    num_tower = 1  # Number of computing units
    tower_type = "gpu"  # Type of computing units ("cpu" or "gpu")
    batch_size = 128  # Batch size for each tower
    num_epochs = optim_parameters["num_epoch"]  # Number of epochs
    valid_size = 0.  # Size of validation set

    # Specify here the decay of Exponential Moving Average, the learning rate of optimizer and the verbose for training.
    decay = optim_parameters["decay"]  # Decay of Exponential Moving Average
    learn_rate = optim_parameters["lr"]  # Learning rate of optimizer

    # Train and test data.
    folds = ShuffleSplit(n_splits=num_folds, test_size=test_size).split(np.empty([feats.shape[0]]))

    for idx, (train_sub, test_sub) in enumerate(folds):
        valid_sub = train_sub[:int(valid_size * len(train_sub))]
        train_sub = train_sub[int(valid_size * len(train_sub)):]

        print(str(len(train_sub)) + " train points and " + str(len(test_sub)) + " test points")

        # Create neural network
        tf.reset_default_graph()

        # Evaluation of neural network
        ltrain, lvalid, ltest, weights = _evaluate_nn_model(labels, feats, diags, train_sub, valid_sub, test_sub,
                                                   model, num_tower, tower_type, num_epochs,
                                                   decay, learn_rate, batch_size, verbose=True)
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

        if visualize_weight:
            if model.get_parameters()["persistence_weight"] == "grid":
                fig = plt.figure(figsize=(10,20))
                filts = list(list_filtrations)
                for nf in range(model.num_filts):
                    plt.subplot(model.num_filts,2,2*nf+1)
                    plt.imshow(weights[nf][0], cmap="Purples",  vmin=kwargs["xmin"], vmax=kwargs["xmax"])
                    plt.title(filts[nf] + " -- before training")
                    plt.colorbar()

                    plt.subplot(model.num_filts,2,2*(nf+1))
                    plt.imshow(weights[nf][-1], cmap="Purples", vmin=kwargs["xmin"], vmax=kwargs["xmax"])
                    plt.title(filts[nf] + " -- after training")
                    plt.colorbar()

                plt.show()

            if model.get_parameters()["persistence_weight"] == "gmix":
                x = np.arange(kwargs["xmin"], kwargs["xmax"], kwargs["xstep"])
                y = np.arange(kwargs["ymin"], kwargs["ymax"], kwargs["ystep"])
                xx, yy = np.meshgrid(x, y)

                fig = plt.figure(figsize=(10,20))

                for nf in range(model.num_filts):

                    ax = fig.add_subplot(model.num_filts,2,2*nf+1)
                    means, varis = weights[nf][0][0], weights[nf][0][1]
                    z = np.zeros(xx.shape)
                    for idx_g in range(means.shape[3]):
                        z += np.exp(-((xx-means[0,0,0,idx_g])**2/(varis[0,0,0,idx_g]) + (yy-means[0,0,1,idx_g])**2/(varis[0,0,1,idx_g])))
                    ax.contourf(xx, yy, z)
                    ax.title.set_text(filts[nf] + " -- before training")

                    ax = fig.add_subplot(model.num_filts,2,2*(nf+1))
                    means, varis = weights[nf][-1][0], weights[nf][-1][1]
                    z = np.zeros(xx.shape)
                    for idx_g in range(means.shape[3]):
                        z += np.exp(-((xx-means[0,0,0,idx_g])**2/(varis[0,0,0,idx_g]) + (yy-means[0,0,1,idx_g])**2/(varis[0,0,1,idx_g])))
                    ax.contourf(xx, yy, z)
                    ax.title.set_text(filts[nf] + " -- after training")
                plt.show()
            
            return weights

    return
