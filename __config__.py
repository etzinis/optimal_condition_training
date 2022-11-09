# © MERL 2022
# Created by Efthymios Tzinis
"""Declare path variables used in this project."""
import os

"""Figure out the paths """
ROOT_DIRPATH = "/home/thymios/MERL/code/optimal_condition_training/"

"""Datasets paths"""
FSD50K_ROOT_PATH = "/mnt/data/FSD50K/FSD50K/"

"""File for FSD50K synonyms class, do not change this path"""
FSD50K_SYNONYMS_P = os.path.join(ROOT_DIRPATH, "optimal_condition_training/dataset_loader/fsd50k_synonyms.txt")

"""Dirpaths for storing pre-trained language models used for text encoding."""
TEXT_EMB_PRETRAINED_DIR = os.path.join(ROOT_DIRPATH, "text_transformers")
