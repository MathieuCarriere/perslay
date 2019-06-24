"""Module :mod:`perskay.archi` implement the persistence layer."""

# Authors: Mathieu Carriere <mathieu.carriere3@gmail.com>
#          Theo Lacombe <theo.lacombe@inria.fr>
#          Martin Royer <martin.royer@inria.fr>
# License: MIT
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow import random_uniform_initializer as rui

from layers import permutation_equivariant_layer, gaussian_layer, image_layer, landscape_layer


# Post-processing operation with combination of batch normalization, dropout and relu
def _post_processing(vector, pro, dropout_value=.9):
    for c in pro:
        if c == "b":
            vector = tf.layers.batch_normalization(vector)
        if c == "d":
            vector = tf.nn.dropout(vector, dropout_value)
        if c == "r":
            vector = tf.nn.relu(vector)
    return vector


# PersLay channel for persistence diagrams
def perslay(output, name, diag,
            layer="pm", perm_op="sum", fc_layers=(), cv_layers=(),
            peq=(50, None), keep=50,
            num_gaussians=150, num_samples=150,
            persistence_weight="linear",
            coeff_init=rui(1., 1.), coeff_const=False,
            grid_size=(50, 50), grid_bnds=((-0.01, 1.01), (-0.01, 1.01)), grid_init=rui(1.0, 1.0), grid_const=False,
            image_size=(30, 30), image_bnds=((0., 1.), (0., 1.)),
            weight_init=rui(0.0, 1.0), weight_const=False, bias_init=rui(0.0, 1.0), bias_const=False,
            mean_init=rui(0.0, 1.0), mean_const=False, variance_init=rui(3.0, 3.0), variance_const=False,
            sample_init=rui(0.0, 1.0), sample_const=False,
            tensor=True):
    """
        output : list containing the output of all channels
        name : string to refer to the variables
        diag : big matrix of shape [N_diag, N_pts_per_diag, dimension_diag (coordinates of points) + 1 (mask--0 or 1)]
        layer: accept "pm", "gs", "ls", "im". Cf code for details. Could write other types of layer.
        perm_op : string, operation invariante par permutation. Accepte "sum", "max", "mean", "topk".
        fc_layers : following fully_connected layers in the current channel. (int, string), nb inter dim,
                        and subsequent activation and regularization function (rbd = ReLu Batchnorm Dropout)
        tensor : token specifying if inputs are tensorflow tensors (True) or numpy arrays (False)
    """

    if not tensor:
        tensor_diag = tf.convert_to_tensor(diag, dtype=tf.float32)
    else:
        tensor_diag = diag
    N, dimension_diag = tensor_diag.get_shape()[1], tensor_diag.get_shape()[2]
    tensor_mask = tensor_diag[:, :, dimension_diag - 1]
    tensor_diag = tensor_diag[:, :, :dimension_diag - 1]

    if persistence_weight == "linear":
        with tf.variable_scope(name + "-linear_pweight"):
            C = tf.get_variable("C", shape=[1], initializer=coeff_init) if not coeff_const \
                else tf.get_variable("C", initializer=coeff_init)
            weight = C * tf.abs(tensor_diag[:, :, 1:2])

    if persistence_weight == "grid":
        with tf.variable_scope(name + "-grid_pweight"):

            W = tf.get_variable("W", shape=grid_size, initializer=grid_init) if not grid_const \
                else tf.get_variable("W", initializer=grid_init)
            indices = []
            for dim in range(dimension_diag-1):
                [m, M] = grid_bnds[dim]
                coords = tf.slice(tensor_diag, [0, 0, dim], [-1, -1, 1])
                ids = grid_size[dim] * (coords - m)/(M - m)
                indices.append(tf.cast(ids, tf.int32))
            weight = tf.expand_dims(tf.gather_nd(params=W, indices=tf.concat(indices, axis=2)), -1)

    # First layer of channel: processing of the persistence diagrams by vectorization of diagram points
    if layer == "pm":  # Channel with permutation equivariant layers
        for idx, (dim, pop) in enumerate(peq):
            with tf.variable_scope(name + "-perm_eq-" + str(idx)):
                tensor_diag = permutation_equivariant_layer(inp=tensor_diag, dimension=dim, perm_op=pop,
                                                            L_init=weight_init, G_init=weight_init, b_init=bias_init,
                                                            L_const=weight_const, G_const=weight_const,
                                                            b_const=bias_const)
    elif layer == "gs":  # Channel with gaussian layer
        with tf.variable_scope(name + "-gaussians"):
            tensor_diag = gaussian_layer(inp=tensor_diag, num_gaussians=num_gaussians, m_init=mean_init,
                                         s_init=variance_init, m_const=mean_const, s_const=variance_const)
    elif layer == "ls":  # Channel with landscape layer
        with tf.variable_scope(name + "-samples"):
            tensor_diag = landscape_layer(inp=tensor_diag, num_samples=num_samples, s_init=sample_init,
                                          s_const=sample_const)
    elif layer == "im":  # Channel with image layer
        with tf.variable_scope(name + "-bandwidth"):
            tensor_diag = image_layer(inp=tensor_diag, im_size=image_size, im_bnds=image_bnds, s_init=variance_init,
                                      s_const=variance_const)

    output_dim = len(tensor_diag.shape) - 2

    vector = None  # to avoid warning

    if output_dim == 1:
        # Apply weight and mask
        if persistence_weight is not None:
            tiled_weight = tf.tile(weight, [1, 1, tensor_diag.shape[2].value])
            tensor_diag = tf.multiply(tensor_diag, tiled_weight)
        tiled_mask = tf.tile(tf.expand_dims(tensor_mask, -1), [1, 1, tensor_diag.shape[2].value])
        masked_layer = tf.multiply(tensor_diag, tiled_mask)

        # Permutation invariant operation
        if perm_op == "topk":  # k first values
            masked_layer_t = tf.transpose(masked_layer, perm=[0, 2, 1])
            values, indices = tf.nn.top_k(masked_layer_t, k=keep)
            vector = tf.reshape(values, [-1, keep * tensor_diag.shape[2].value])
        elif perm_op == "sum":  # sum
            vector = tf.reduce_sum(masked_layer, axis=1)
        elif perm_op == "max":  # maximum
            vector = tf.reduce_max(masked_layer, axis=1)
        elif perm_op == "mean":  # minimum
            vector = tf.reduce_mean(masked_layer, axis=1)

        # Second layer of channel: fully-connected (None if fc_layers is set to [], default value)
        for idx, tup in enumerate(fc_layers):
            # tup is a tuple whose element are
            # 1. dim of fully-connected,
            # 2. string for processing,
            # 3. (optional) dropout value
            with tf.variable_scope(name + "-fc-" + str(idx)):
                vector = tf.layers.dense(vector, tup[0])
            with tf.variable_scope(name + "-bn-" + str(idx)):
                if len(tup) == 2:
                    vector = _post_processing(vector, tup[1])
                else:
                    vector = _post_processing(vector, tup[1], tup[2])

    elif output_dim == 2:

        # Apply weight and mask
        if persistence_weight is not None:
            weight = tf.expand_dims(weight, -1)
            tiled_weight = tf.tile(weight, [1, 1, tensor_diag.shape[2].value, tensor_diag.shape[3].value])
            tensor_diag = tf.multiply(tensor_diag, tiled_weight)
        tiled_mask = tf.tile(tf.reshape(tensor_mask, [-1, N, 1, 1]),
                             [1, 1, tensor_diag.shape[2].value, tensor_diag.shape[3].value])
        masked_layer = tf.multiply(tensor_diag, tiled_mask)

        # Permutation invariant operation
        if perm_op == "sum":  # sum
            vector = tf.reduce_sum(masked_layer, axis=1)
        elif perm_op == "max":  # maximum
            vector = tf.reduce_max(masked_layer, axis=1)
        elif perm_op == "mean":  # minimum
            vector = tf.reduce_mean(masked_layer, axis=1)

        # Second layer of channel: convolution
        vector = tf.expand_dims(vector, -1)
        for idx, tup in enumerate(cv_layers):
            # tup is a tuple whose element are
            # 1. num of filters,
            # 2. kernel size,
            # 3. string for postprocessing,
            # 4. (optional) dropout value
            with tf.variable_scope(name + "-cv-" + str(idx)):
                vector = tf.layers.conv2d(vector, filters=tup[0], kernel_size=tup[1])
            with tf.variable_scope(name + "-bn-" + str(idx)):
                if len(tup) == 3:
                    vector = _post_processing(vector, tup[2])
                else:
                    vector = _post_processing(vector, tup[2], tup[3])
        vector = tf.layers.flatten(vector)

    output.append(vector)
    return vector


class baseModel:

    def __init__(self, parameters, filts, labels):
        self.num_filts = len(filts)
        self.parameters = parameters
        self.num_labels = labels.shape[1]

    def get_parameters(self):
        return self.parameters

    def instance(self, feats, diags):
        list_v = []
        for i in range(self.num_filts):
            # A perslay channel must be defined for each type of diagram
            # Here, they all have the same hyper-parameters.
            perslay(output=list_v,  # the vector use to store the output of all perslay
                    name="perslay-" + str(i),  # name of this layer
                    diag=diags[i],  # this layer handle the i-th type of diagrams
                    layer=self.parameters["layer_type"],
                    perm_op=self.parameters["perm_op"],
                    keep=self.parameters["keep"],
                    persistence_weight=self.parameters["weight"],
                    grid_size=self.parameters["grid_size"],
                    image_size=self.parameters["image_size"],
                    num_gaussians=self.parameters["num_gaussians"],
                    num_samples=self.parameters["num_samples"],
                    peq=[(self.parameters["pm_dimension"], None)]
                    )

        # Concatenate all channels and add other features
        vector = tf.concat(list_v, 1)
        with tf.variable_scope("norm_feat"):
            feat = tf.layers.batch_normalization(feats)

        vector = tf.concat([vector, feat], 1)

        #  Final layer to make predictions
        with tf.variable_scope("final-dense-3"):
            vector = _post_processing(tf.layers.dense(vector, self.num_labels), "")

        return vector
