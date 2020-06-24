# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 03:22:16 2020
This code can be used to fetch the weights of our pretrained self-supervised model.
@author: pritam
"""

import os
import tensorflow as tf
import numpy as np

## specify the path where model is saved
model_path = os.path.abspath("enter_the_path_model_saved\\saved_model\\") 
## initialize the dictionary to save the weights and bias
weights = {}
with tf.compat.v1.Session() as sess:
    saver       = tf.compat.v1.train.import_meta_graph(model_path + "\\SSL_model.ckpt.meta")
    new_saver   = saver.restore(sess, tf.train.latest_checkpoint(model_path))

    graph       = tf.compat.v1.get_default_graph() # get the default graph
    tvars       = tf.trainable_variables() # get all trainable variables
    tvars_vals  = sess.run(tvars) 

    for var, val in zip(tvars, tvars_vals):
        weights[var.name] = val # save weights and bias in the dictionary

np.save('weights_bias.npy', weights)        
