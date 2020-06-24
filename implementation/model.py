# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 06:58:35 2019

@author: Pritam
"""
import tensorflow as tf
import utils
import keras
import os
import numpy as np
window_size = 2560
transform_task = [0, 1, 2, 3, 4, 5, 6]


def conv_block(input_tensor,  filter_size, kernel_size, stride, batch_norm, dropout, dropout_rate, isTraining, name):
  
    reuse = tf.compat.v1.AUTO_REUSE
    conv  = tf.layers.conv1d(inputs = input_tensor, filters = filter_size, kernel_size = kernel_size, strides = stride, padding='same', name=name, reuse=reuse)
    if batch_norm:
        conv    = tf.layers.batch_normalization(conv, training=isTraining, name = name, reuse=reuse)
    conv     = tf.nn.leaky_relu(conv, name = name)
    if dropout:
        conv    = tf.layers.dropout(inputs=conv, rate=dropout_rate, training=isTraining, name = name)
    
    return conv

def dense_block(input_tensor, hidden_nodes, drop_rate, isTraining, name):
    
    reuse = tf.compat.v1.AUTO_REUSE
    dense       = tf.layers.dense  (inputs=input_tensor, units=hidden_nodes, reuse=reuse,  name= name )
    dense       = tf.nn.leaky_relu(dense)
    dense       = tf.layers.dropout(inputs=dense, rate=drop_rate, training=isTraining, name= name)
    
    return dense


def self_supervised_model(input_tensor, isTraining, drop_rate, hidden_nodes = 128, stride_mp = 4):

    reuse = tf.compat.v1.AUTO_REUSE
    
    main_branch = conv_block(input_tensor,  filter_size = 32, kernel_size = 32, stride = 1, batch_norm = False, dropout = False, dropout_rate = drop_rate * 0.5, isTraining = isTraining, name = 'conv_layer_1')
    main_branch = conv_block(main_branch,  filter_size = 32, kernel_size = 32, stride = 1, batch_norm = False, dropout = False, dropout_rate = drop_rate * 0.5, isTraining = isTraining, name = 'conv_layer_2')

    ## conv block 1
    conv1     = main_branch 
    conv1     = tf.layers.max_pooling1d(conv1, pool_size = conv1.get_shape()[1].value, strides=stride_mp, padding='valid', name = 'GAP1')    
    conv1     = tf.layers.flatten(conv1, name = 'flat_layer1')

    main_branch     = tf.layers.max_pooling1d(main_branch, pool_size = 8, strides=2, padding='valid', name = 'mp1') 
    main_branch = conv_block(main_branch,  filter_size = 64, kernel_size = 16, stride = 1, batch_norm = False, dropout = False, dropout_rate = drop_rate * 0.5, isTraining = isTraining, name = 'conv_layer_3')
    main_branch = conv_block(main_branch,  filter_size = 64, kernel_size = 16, stride = 1, batch_norm = False, dropout = False, dropout_rate = drop_rate * 0.5, isTraining = isTraining, name = 'conv_layer_4')

    ## conv block 2
    conv2     = main_branch 
    conv2     = tf.layers.max_pooling1d(conv2, pool_size = conv2.get_shape()[1].value, strides=stride_mp, padding='valid', name = 'GAP2') 
    conv2     = tf.layers.flatten(conv2, name = 'flat_layer2')

    main_branch     = tf.layers.max_pooling1d(main_branch, pool_size = 8, strides=2, padding='valid', name = 'mp2')        
    main_branch = conv_block(main_branch,  filter_size = 128, kernel_size = 8, stride = 1, batch_norm = False, dropout = False, dropout_rate = drop_rate * 0.5, isTraining = isTraining, name = 'conv_layer_5')
    main_branch = conv_block(main_branch,  filter_size = 128, kernel_size = 8, stride = 1, batch_norm = False, dropout = False, dropout_rate = drop_rate * 0.5, isTraining = isTraining, name = 'conv_layer_6')

    ## conv block 3
    conv3     = main_branch 
    conv3     = tf.layers.max_pooling1d(conv3, pool_size = conv3.get_shape()[1].value, strides=stride_mp, padding='valid', name = 'GAP3') 
    conv3     = tf.layers.flatten(conv3, name = 'flat_layer3')
    
    gap_pool_size   = main_branch.get_shape()[1].value
    main_branch     = tf.layers.max_pooling1d(main_branch, pool_size = gap_pool_size, strides=1, padding='valid', name = 'GAP')
    main_branch     = tf.layers.flatten(main_branch, name = 'flat_layer') ## final conv block output

    ## dense layer branches
    task_0      = dense_block(input_tensor = main_branch, hidden_nodes= hidden_nodes, drop_rate= drop_rate, isTraining= isTraining, name = 'task_0_dense_1')
    task_0      = dense_block(input_tensor = task_0,      hidden_nodes= hidden_nodes, drop_rate= drop_rate, isTraining= isTraining, name = 'task_0_dense_2')
    task_0      = tf.layers.dense(inputs=task_0, units=1, name='task_0', reuse=reuse)

    task_1      = dense_block(input_tensor = main_branch, hidden_nodes= hidden_nodes, drop_rate= drop_rate, isTraining= isTraining, name = 'task_1_dense_1')
    task_1      = dense_block(input_tensor = task_1,      hidden_nodes= hidden_nodes, drop_rate= drop_rate, isTraining= isTraining, name = 'task_1_dense_2')
    task_1      = tf.layers.dense(inputs=task_1, units=1, name='task_1', reuse=reuse)
    
    task_2      = dense_block(input_tensor = main_branch, hidden_nodes= hidden_nodes, drop_rate= drop_rate, isTraining= isTraining, name = 'task_2_dense_1')
    task_2      = dense_block(input_tensor = task_2,      hidden_nodes= hidden_nodes, drop_rate= drop_rate, isTraining= isTraining, name = 'task_2_dense_2')
    task_2      = tf.layers.dense(inputs=task_2, units=1, name='task_2', reuse=reuse)

    task_3      = dense_block(input_tensor = main_branch, hidden_nodes= hidden_nodes, drop_rate= drop_rate, isTraining= isTraining, name = 'task_3_dense_1')
    task_3      = dense_block(input_tensor = task_3,      hidden_nodes= hidden_nodes, drop_rate= drop_rate, isTraining= isTraining, name = 'task_3_dense_2')
    task_3      = tf.layers.dense(inputs=task_3, units=1, name='task_3', reuse=reuse)

    task_4      = dense_block(input_tensor = main_branch, hidden_nodes= hidden_nodes, drop_rate= drop_rate, isTraining= isTraining, name = 'task_4_dense_1')
    task_4      = dense_block(input_tensor = task_4,      hidden_nodes= hidden_nodes, drop_rate= drop_rate, isTraining= isTraining, name = 'task_4_dense_2')
    task_4      = tf.layers.dense(inputs=task_4, units=1, name='task_4', reuse=reuse)

    task_5      = dense_block(input_tensor = main_branch, hidden_nodes= hidden_nodes, drop_rate= drop_rate, isTraining= isTraining, name = 'task_5_dense_1')
    task_5      = dense_block(input_tensor = task_5,      hidden_nodes= hidden_nodes, drop_rate= drop_rate, isTraining= isTraining, name = 'task_5_dense_2')
    task_5      = tf.layers.dense(inputs=task_5, units=1, name='task_5', reuse=reuse)
    
    task_6      = dense_block(input_tensor = main_branch, hidden_nodes= hidden_nodes, drop_rate= drop_rate, isTraining= isTraining, name = 'task_6_dense_1')
    task_6      = dense_block(input_tensor = task_6,      hidden_nodes= hidden_nodes, drop_rate= drop_rate, isTraining= isTraining, name = 'task_6_dense_2')
    task_6      = tf.layers.dense(inputs=task_6, units=1, name='task_6', reuse=reuse)
    
    return conv1, conv2, conv3, main_branch, task_0, task_1, task_2, task_3, task_4, task_5, task_6




def supervised_model_swell(x_tr_feature, 
                                  y_tr, 
                                  x_te_feature, 
                                  y_te, 
                                  identifier, 
                                  kfold, 
                                  result,
                                  summaries,
                                  current_time,
                                  epoch_super=200, 
                                  batch_super=128,
                                  lr_super=0.001,
                                  hidden_nodes=512,
                                  dropout=0,
                                  L2=0):
    

    input_dimension     = x_tr_feature.shape[1]
    output_dimension    = y_tr.shape[1]
    log_dir = os.path.join(summaries, 'ER')
    result  = os.path.join(result, 'ER')
    tb      = keras.callbacks.TensorBoard(log_dir= log_dir)
    
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(hidden_nodes, input_dim=input_dimension, activation='relu', kernel_regularizer = keras.regularizers.l2(L2)))
    model.add(keras.layers.Dense(hidden_nodes, activation='relu', kernel_regularizer = keras.regularizers.l2(L2)))

    if output_dimension == 2:
        model.add(keras.layers.Dense(output_dimension, activation='sigmoid'))
        op = keras.optimizers.Adam(lr=lr_super)
        model.compile(loss='binary_crossentropy', optimizer=op, metrics=['accuracy'])
    else :
        model.add(keras.layers.Dense(output_dimension))
        model.add(keras.layers.Activation('softmax'))
        op = keras.optimizers.Adam(lr=lr_super)
        model.compile(loss='categorical_crossentropy', optimizer=op, metrics=['accuracy'])

    model.fit(x_tr_feature, y_tr, epochs=epoch_super, batch_size=batch_super, callbacks=[tb], verbose=0, validation_data = (x_te_feature, y_te), shuffle = True) 
    y_tr_pred = model.predict(x_tr_feature, batch_size=batch_super)
    y_te_pred = model.predict(x_te_feature, batch_size=batch_super)
    
    y_tr        = np.argmax(y_tr, axis = 1)
    y_te        = np.argmax(y_te, axis = 1)
    
    y_tr_pred = np.argmax(y_tr_pred, axis = 1)
    y_te_pred = np.argmax(y_te_pred, axis = 1)

    utils.model_result_store(y_tr, y_tr_pred, os.path.join(result, str("tr_" + identifier + ".csv")), kfold)
    utils.model_result_store(y_te, y_te_pred, os.path.join(result, str("te_" + identifier + ".csv")), kfold)
       
    return  


def supervised_model_wesad(x_tr_feature, 
                                  y_tr, 
                                  x_te_feature, 
                                  y_te, 
                                  identifier, 
                                  kfold, 
                                  result,
                                  summaries,
                                  current_time,
                                  epoch_super=200, 
                                  batch_super=128,
                                  lr_super=0.001,
                                  hidden_nodes=512,
                                  dropout=0.2,
                                  L2=0):
    

    input_dimension     = x_tr_feature.shape[1]
    output_dimension    = y_tr.shape[1]
    log_dir = os.path.join(summaries, 'ER')
    result  = os.path.join(result, 'ER')
    tb      = keras.callbacks.TensorBoard(log_dir= log_dir)
    
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(hidden_nodes, input_dim=input_dimension, activation='relu', kernel_regularizer = keras.regularizers.l2(L2)))
    model.add(keras.layers.Dense(hidden_nodes, activation='relu', kernel_regularizer = keras.regularizers.l2(L2)))

    if output_dimension == 2:
        model.add(keras.layers.Dense(output_dimension, activation='sigmoid'))
        op = keras.optimizers.Adam(lr=lr_super)
        model.compile(loss='binary_crossentropy', optimizer=op, metrics=['accuracy'])
    
    else :
        model.add(keras.layers.Dense(output_dimension))
        model.add(keras.layers.Activation('softmax'))
        op = keras.optimizers.Adam(lr=lr_super)
        model.compile(loss='categorical_crossentropy', optimizer=op, metrics=['accuracy'])

    model.fit(x_tr_feature, y_tr, epochs=epoch_super, batch_size=batch_super, callbacks=[tb], verbose=0, validation_data = (x_te_feature, y_te), shuffle = True) 
    y_tr_pred = model.predict(x_tr_feature, batch_size=batch_super)
    y_te_pred = model.predict(x_te_feature, batch_size=batch_super)
    
    y_tr        = np.argmax(y_tr, axis = 1)
    y_te        = np.argmax(y_te, axis = 1)
    
    y_tr_pred = np.argmax(y_tr_pred, axis = 1)
    y_te_pred = np.argmax(y_te_pred, axis = 1)
    
    utils.model_result_store(y_tr, y_tr_pred, os.path.join(result, str("tr_" + identifier + ".csv")), kfold)
    utils.model_result_store(y_te, y_te_pred, os.path.join(result, str("te_" + identifier + ".csv")), kfold)
  
    return  




def supervised_model_dreamer(x_tr_feature, 
                                  y_tr, 
                                  x_te_feature, 
                                  y_te, 
                                  identifier, 
                                  kfold, 
                                  result,
                                  summaries,
                                  current_time,
                                  epoch_super=200, 
                                  batch_super=128,
                                  lr_super=0.001,
                                  hidden_nodes=512,
                                  dropout=0.2,
                                  L2=0.0001):   

     
    input_dimension     = x_tr_feature.shape[1]
    output_dimension    = y_tr.shape[1]
    log_dir = os.path.join(summaries, 'ER')
    result  = os.path.join(result, 'ER')
    tb      = keras.callbacks.TensorBoard(log_dir= log_dir)
    
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(hidden_nodes, input_dim=input_dimension, activation='relu', kernel_regularizer = keras.regularizers.l2(L2)))
    model.add(keras.layers.Dropout(rate = dropout))
    model.add(keras.layers.Dense(hidden_nodes, activation='relu', kernel_regularizer = keras.regularizers.l2(L2)))
    model.add(keras.layers.Dropout(rate = dropout))
    model.add(keras.layers.Dense(hidden_nodes, activation='relu', kernel_regularizer = keras.regularizers.l2(L2)))
    model.add(keras.layers.Dropout(rate = dropout))
    
    if output_dimension == 2:
        model.add(keras.layers.Dense(output_dimension, activation='sigmoid'))
        op = keras.optimizers.Adam(lr=lr_super)
        model.compile(loss='binary_crossentropy', optimizer=op, metrics=['accuracy'])
    
    else :
        model.add(keras.layers.Dense(output_dimension))
        model.add(keras.layers.Activation('softmax'))
        op = keras.optimizers.Adam(lr=lr_super)
        model.compile(loss='categorical_crossentropy', optimizer=op, metrics=['accuracy'])

    model.fit(x_tr_feature, y_tr, epochs=epoch_super, batch_size=batch_super, callbacks=[tb], verbose=0, validation_data = (x_te_feature, y_te), shuffle = True) 
    y_tr_pred = model.predict(x_tr_feature, batch_size=batch_super)
    y_te_pred = model.predict(x_te_feature, batch_size=batch_super)
    
    y_tr        = np.argmax(y_tr, axis = 1)
    y_te        = np.argmax(y_te, axis = 1)
    
    y_tr_pred = np.argmax(y_tr_pred, axis = 1)
    y_te_pred = np.argmax(y_te_pred, axis = 1)
    
    utils.model_result_store(y_tr, y_tr_pred, os.path.join(result, str("tr_" + identifier + ".csv")), kfold)
    utils.model_result_store(y_te, y_te_pred, os.path.join(result, str("tr_" + identifier + ".csv")), kfold)

   
    return 

def supervised_model_amigos(x_tr_feature, 
                                  y_tr, 
                                  x_te_feature, 
                                  y_te, 
                                  identifier, 
                                  kfold, 
                                  result,
                                  summaries,
                                  current_time,
                                  epoch_super=200, 
                                  batch_super=128,
                                  lr_super=0.001,
                                  hidden_nodes=512,
                                  dropout=0.4,
                                  L2=0):           

    
    input_dimension     = x_tr_feature.shape[1]
    output_dimension    = y_tr.shape[1]
    log_dir = os.path.join(summaries, 'ER')
    result  = os.path.join(result, 'ER')
    tb      = keras.callbacks.TensorBoard(log_dir= log_dir)
    
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(hidden_nodes, input_dim=input_dimension, activation='relu', kernel_regularizer = keras.regularizers.l2(L2)))
    model.add(keras.layers.Dropout(rate = dropout))

    model.add(keras.layers.Dense(hidden_nodes, activation='relu', kernel_regularizer = keras.regularizers.l2(L2)))
    model.add(keras.layers.Dropout(rate = dropout))
    model.add(keras.layers.Dense(hidden_nodes, activation='relu', kernel_regularizer = keras.regularizers.l2(L2)))
    model.add(keras.layers.Dropout(rate = dropout))
    
    if output_dimension == 2:
        model.add(keras.layers.Dense(output_dimension, activation='sigmoid'))
        op = keras.optimizers.Adam(lr=lr_super)
        model.compile(loss='binary_crossentropy', optimizer=op, metrics=['accuracy'])
    
    else :
        model.add(keras.layers.Dense(output_dimension))
        model.add(keras.layers.Activation('softmax'))
        op = keras.optimizers.Adam(lr=lr_super)
        model.compile(loss='categorical_crossentropy', optimizer=op, metrics=['accuracy'])
        
    model.fit(x_tr_feature, y_tr, epochs=epoch_super, batch_size=batch_super, verbose=0, callbacks=[tb], validation_data = (x_te_feature, y_te), shuffle = True) 
   
    y_tr_pred = model.predict(x_tr_feature, batch_size=batch_super)
    y_te_pred = model.predict(x_te_feature, batch_size=batch_super)
    
    y_tr        = np.argmax(y_tr, axis = 1)
    y_te        = np.argmax(y_te, axis = 1)
    
    y_tr_pred = np.argmax(y_tr_pred, axis = 1)
    y_te_pred = np.argmax(y_te_pred, axis = 1)

    utils.model_result_store(y_tr, y_tr_pred, os.path.join(result, str("tr_" + identifier + ".csv")), kfold)
    utils.model_result_store(y_te, y_te_pred, os.path.join(result, str("tr_" + identifier + ".csv")), kfold)

    return 
