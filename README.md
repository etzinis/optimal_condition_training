# Optimal Condition Training (OCT) for Target Source Separation

Code and data recipes for the paper: Optimal Condition Training for Target Source Separation by Efthymios Tzinis, Gordon Wichern, Paris Smaragdis and Jonathan Le Roux 

TLDR; The main contribution of this paper is to introduce a novel way of training, namely, optimal condition training (OCT) for single-channel target source separation, based on greedy parameter updates using the highest performing condition among equivalent conditions associated with a given target source. OCT improves upon single-conditioned models and oracle permutation invariant training. We also propose a variation of OCT with condition refinement, in which an initial conditional vector is adapted to the given mixture and transformed to a more amenable representation for target source extraction.

## Table of contents

- [Paths Configurations](#paths-configurations)
- [How to use the dataset loaders](#how-to-use-the-dataset-loaders)
- [References](#references)

## Paths Configurations

Change the dataset paths to the ones stored locally for all the root directories (the metadata paths are going to be created after runnign the scripts presented below):
```shell
git clone https://github.com/etzinis/optimal_condition_training.git
export PYTHONPATH={the path that you stored the github repo}:$PYTHONPATH
cd optimal_condition_training
vim __config__.py
```

You should change the following:
```shell
ROOT_DIRPATH = "the path that you stored the github repo"
FSD50K_ROOT_PATH = "the path that you stored FSD50K"
```

## How to use the dataset loaders
Now that everything is in place, one might use the combined dataset loader which enables to create the three different versions of FSD50K [[1]](#1) dataset.

- *Random super-classes*: We first randomly sample two distinct sound classes (out of the available 200), then sample a representative source waveform for each class and mix them together. ```is_hierarchical=False, intra_class_prior=0.```
- *Different super-classes*: We select a subset of classes from the FSD50K ontology corresponding to six diverse and more challenging to separate super-classes of sounds, namely: Animal (21 subclasses), Musical Instrument (35 subclasses), Vehicle (15 subclasses), Domestic & Home Sounds (26 subclasses), Speech (5 subclasses) and Water Sounds (6 subclasses). Each mixture contains two sound waveforms that belong to distinct super-classes. ```is_hierarchical=True, intra_class_prior=0.```
- *Same super-class*: Following the super-class definition from above, we force each mixture to consist of sources that belong to the same abstract category of sounds to test the ability of text-conditioned models in extremely challenging scenarios. ```is_hierarchical=True, intra_class_prior=1.```

By channging the following variables we can also control the samplign probability of each heterogeneous condition [[2]](#2):
```python
valid_conditions = ["harmonicity", "energy", "source_order",
                    "one_hot_class", "one_hot_super_class", "multi_hot_class"]
queries_priors = [0.4] + [0.3] + [0.3] + [0.] * 3
```

The dataset uses online mixing. You can test the generator as shown next:

```shell
cd optimal_condition_training/dataset_loader 
➜  dataset_loader git:(main) ✗ python heterogen_fsd50k_2mix.py
For randomly sampling between 200 classes of sounds in FSD50K.
Sampled the classes: ('Dishes', 'Crushing', 'Cat', 'Telephone', 'Doorbell', 'Sigh', 'Singing', 'Chirp')
Sampled the following conditional vectors: tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 1., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 1., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]])
It took me: 1.7711923122406006 secs to fetch the batch

For a hierarchical FSD50K version with probability of sampling from the same super class: 0.0
Sampled the classes: ('Stream', 'Bathtub', 'Child speech', 'Writing', 'Toilet flush', 'Frog', 'Child speech', 'Livestock')
Sampled the following conditional vectors: tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [1., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 1.,  ..., 0., 0., 0.]])
It took me: 0.8072104454040527 secs to fetch the batch

For a hierarchical FSD50K version with probability of sampling from the same super class: 1.0
Sampled the classes: ('Child speech', 'Tambourine', 'Boat', 'Drum kit', 'Truck', 'Waves', 'Wind chime', 'Stream')
Sampled the following conditional vectors: tensor([[1., 0., 0.,  ..., 0., 0., 0.],
        [1., 0., 0.,  ..., 0., 0., 0.],
        [1., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [1., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]])
It took me: 1.0530891418457031 secs to fetch the batch
```   

## References

<a id="1">[1]</a> E. Fonseca, X. Favory, J. Pons, F. Font, and X. Serra, “Fsd50k: an open dataset of human-labeled sound events,” arXiv preprint arXiv:2010.00475, 2020.

<a id="2">[2]</a> Tzinis, E., Wichern G., Subramanian, A., Smaragdis, P., and Le Roux, J., “Heterogeneous target speech separation.” In Proceedings of Interspeech, 2022, pp. 1796-1800.

<a id="3">[3]</a> Tzinis, E., Wang, Z., Jiang, X., and Smaragdis, P., “Compute and memory efficient universal sound source separation.” In Journal of Signal Processing Systems, vol. 9, no. 2, pp. 245–259, 2022, Springer
