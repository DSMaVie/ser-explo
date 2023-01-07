# Instructions

## Preprocessing

Preprocessing is controlled by `.yaml` instructions. These are structured with the following keys:

* `src`: the source corpus
* `steps`: the description of thge individual steps to alter the manifest.
  * `name`: the name of the step. must match a step from the `PreproFunc` enum.
  * `args`: arguments for this step
  *
* `label`: information for the label encoder-decoder (encodec).
  * `classes`: a list of possible classes
  * `NofN`: weather labels are exclusive or not. If true the encoder produces MHE labels
* `features`: information for the feature extractor
  * arguments for this depend on the feature extractor class and are given at instantiation time.

## Architecture

The architecture folder contains default hyperparams for the architectures we test. The keys are the same as the parameter
names of the `__init__` of the the model of the same name as the `yaml` file they are defined in.

## train

contains training parameters for the different trainings. there are a few required ones:

* `model` is the architectural `yml` file that gets pulled.
* `loss_fn` contains the name of the Loss fn used. see `erinyes/train/other.py` for the different options.
  sometimes we decide between multi_binary decisisons and multiclass decisisons this will be resolved depeding on the
  data used. I.e. if the loss_fn is `ce` then it gets resolved to either `mc_ce` (multiclass) or `binary_ce`,
  depeding on the data.
* `optimizer` contains the name of the optimizer used. see `erinyes/train/other.py` for the different options.