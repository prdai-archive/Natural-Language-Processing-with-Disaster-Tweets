import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.nn import *
from torch.optim import *
from torchvision.models import *
from sklearn.model_selection import *
from sklearn.metrics import f1_score,accuracy_score,precision_score
import wandb
import nltk
from nltk.stem.porter import *
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn import svm
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import warnings
import os
warnings.filterwarnings("ignore")
PROJECT_NAME = "Natural-Language-Processing-with-Disaster-Tweets"
np.random.seed(55)
stemmer = PorterStemmer()
device = "cuda"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")
from Modelling.dataset import *
from Modelling.metrics import *
from Modelling.modelling import *
from Modelling.help_funcs import *
from Modelling.parameter_tunning import *
from Modelling.preproccessing import *
