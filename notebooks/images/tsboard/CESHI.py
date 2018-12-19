"""
 Deeper Multi-Layer Pecptron with XAVIER Init
 Xavier init from {Project: https://github.com/aymericdamien/TensorFlow-Examples/}
 @Sungjoon Choi (sungjoon.choi@cpslab.snu.ac.kr)
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
%matplotlib inline
import gc
gc.collect()