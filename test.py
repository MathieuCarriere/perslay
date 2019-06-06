"""Module :mod:`perslay.test` Test script."""

# Authors: Theo Lacombe <theo.lacombe@inria.fr>
# License: MIT
from utils import generate, load
from archi import baseModel
from preprocessing import preprocess
from expe import single_run
print("...imports done")

dataset = "MUTAG"

generate(dataset)

diags_tmp, feats, labels = load(dataset)
diags, filts = preprocess(diags_tmp)

layer_type = "im"

perm_op = "sum"
keep = 5  # only useful if perm_op = "topk"

weight = "grid"
grid_size = (10, 10)
# Parameter specific to layer_type="im"
image_size = (10, 10)
# Parameter specific to layer_type="gs"
num_gaussians = 50
# Parameter specific to layer_type="pm"
d = 50  # Output dimension
# Parameter specific to layer_type="ls"
num_samples = 50

perslayParameters = {"layer_type":layer_type,
                     "perm_op": perm_op, "keep":keep,
                     "weight":weight, "grid_size":grid_size,
                     "image_size": image_size,
                     "num_gaussians": num_gaussians,
                     "pm_dimension": d,
                     "num_samples": num_samples}

model = baseModel(perslayParameters, filts, labels)

decay = 0.9
lr = 0.01
num_epoch = 100
optim_parameters = {"decay":decay,"lr":lr, "num_epoch":num_epoch}

single_run(diags, feats, labels, filts, model, optim_parameters, test_size=0.1)