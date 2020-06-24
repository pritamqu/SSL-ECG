# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 23:57:24 2019

@author: Pritam
"""


import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

tf.logging.set_verbosity(tf.logging.ERROR)

import model
import utils
import data_preprocessing

## mention paths
data_folder     = os.path.join(os.path.dirname(dirname), 'data_folder')
summaries       = os.path.join(os.path.dirname(dirname), 'summaries')
output          = os.path.join(os.path.dirname(dirname), 'output')
model_dir       = os.path.join(os.path.dirname(dirname), 'models')

## transformation task params
noise_param = 15 #noise_amount
scale_param = 1.1 #scaling_factor
permu_param = 20 #permutation_pieces
tw_piece_param = 9 #time_warping_pieces
twsf_param = 1.05 #time_warping_stretch_factor
no_of_task = ['original_signal', 'noised_signal', 'scaled_signal', 'negated_signal', 'flipped_signal', 'permuted_signal', 'time_warped_signal'] 
transform_task = [0, 1, 2, 3, 4, 5, 6] #transformation labels
single_batch_size = len(transform_task)

## hyper parameters
batchsize = 128  
actual_batch_size =  batchsize * single_batch_size
log_step = 100
epoch = 100
initial_learning_rate = 0.001
drop_rate = 0.6
regularizer = 1
L2 = 0.0001
lr_decay_steps = 10000
lr_decay_rate = 0.9
loss_coeff = [0.195, 0.195, 0.195, 0.0125, 0.0125, 0.195, 0.195]
window_size = 2560
extract_data = 0
current_time    = utils.current_time()

""" for the first time run this """ 
if extract_data == 1:
    _       = data_preprocessing.extract_swell_dataset(overlap_pct= 0, window_size_sec= 10, data_save_path= data_folder, save= 1)
    _       = data_preprocessing.extract_dreamer_dataset(overlap_pct= 0, window_size_sec= 10, data_save_path= data_folder, save= 1)
    _       = data_preprocessing.extract_amigos_dataset(overlap_pct= 0, window_size_sec= 10, data_save_path= data_folder, save=1)
    _       = data_preprocessing.extract_wesad_dataset(overlap_pct=0, window_size_sec=10, data_save_path= data_folder, save=1)

## load datasets
swell_data              = data_preprocessing.load_data(os.path.join(data_folder, 'swell_dict.npy'))
dreamer_data            = data_preprocessing.load_data(os.path.join(data_folder, 'dreamer_dict.npy'))   
amigos_data             = data_preprocessing.load_data(os.path.join(data_folder, 'amigos_dict.npy'))
wesad_data              = data_preprocessing.load_data(os.path.join(data_folder, 'wesad_dict.npy'))  

## prepared as 10 fold cv
swell_data              = data_preprocessing.swell_prepare_for_10fold(swell_data)  #person, y_input_stress, y_arousal, y_valence, 
wesad_data              = data_preprocessing.wesad_prepare_for_10fold(wesad_data) # person, y_stress
amigos_data             = data_preprocessing.amigos_prepare_for_10fold(amigos_data) # person, y_arousal, y_valence, y_dominance
dreamer_data            = data_preprocessing.dreamer_prepare_for_10fold(dreamer_data) # person, y_arousal, y_valence, y_dominance

total_fold = 10
kf = KFold(n_splits=total_fold, shuffle=True, random_state=True)
swell_train_index, swell_test_index     = utils.get_train_test_index(swell_data, kf)
wesad_train_index, wesad_test_index     = utils.get_train_test_index(wesad_data, kf)
amigos_train_index, amigos_test_index   = utils.get_train_test_index(amigos_data, kf)
dreamer_train_index, dreamer_test_index = utils.get_train_test_index(dreamer_data, kf)

""" self supervised task start """

graph = tf.Graph()
print('creating graph...')
with graph.as_default():
    
    ## initialize tensor
    
    input_tensor        = tf.compat.v1.placeholder(tf.float32, shape = (None, window_size, 1), name = "input")
    y                   = tf.compat.v1.placeholder(tf.float32, shape = (None, np.shape(transform_task)[0]), name = "output") 
    drop_out            = tf.compat.v1.placeholder_with_default(1.0, shape=(), name="Drop_out")
    isTrain             = tf.placeholder(tf.bool, name = 'isTrain')
    global_step         = tf.Variable(0, dtype=np.float32, trainable=False, name="steps")

    conv1, conv2, conv3, main_branch, task_0, task_1, task_2, task_3, task_4, task_5, task_6 = model.self_supervised_model(input_tensor, isTraining= isTrain, drop_rate= drop_out)
    logits = [task_0, task_1, task_2, task_3, task_4, task_5, task_6]
    ## main branch is the output after all conv layers
    featureset_size = main_branch.get_shape()[1].value
    y_label = utils.get_label(y= y, actual_batch_size= actual_batch_size)
    all_loss = utils.calculate_loss(y_label, logits)
    output_loss = utils.get_weighted_loss(loss_coeff, all_loss)  
    
    if regularizer:
        l2_loss = 0
        weights = []
        for v in tf.trainable_variables():
            weights.append(v)
            if 'kernel' in v.name:
                l2_loss += tf.nn.l2_loss(v)
        output_loss = output_loss + l2_loss * L2
        
    y_pred                = utils.get_prediction(logits = logits)
    learning_rate         = tf.compat.v1.train.exponential_decay(initial_learning_rate, global_step, decay_steps=lr_decay_steps, decay_rate=lr_decay_rate, staircase=True)

    optimizer             = tf.compat.v1.train.AdamOptimizer(learning_rate) 
    
    with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
        train_op    = optimizer.minimize(output_loss, global_step, colocate_gradients_with_ops=True)
        
    with tf.variable_scope('Session_saver'):
        saver       = tf.compat.v1.train.Saver(max_to_keep=10)

    tf.compat.v1.summary.scalar('learning_rate/lr', learning_rate)
    tf.compat.v1.summary.scalar('loss/training_batch_loss', output_loss)
    
    summary_op      = tf.compat.v1.summary.merge_all()    
        
print('graph creation finished')

""" Training """
for k in range(total_fold):
    
    flag                    = k
    ## save STR results
    tr_ssl_result_filename  =  os.path.join(output, "STR_result"   , str("tr_" + str(k) +"_"  + current_time + ".npy"))
    te_ssl_result_filename  =  os.path.join(output, "STR_result"   , str("te_" + str(k) +"_"  + current_time + ".npy"))
    tr_ssl_loss_filename    =  os.path.join(output, "STR_loss"     , str("tr_" + str(k) +"_"  + current_time + ".npy"))
    te_ssl_loss_filename    =  os.path.join(output, "STR_loss"     , str("te_" + str(k) +"_"  + current_time + ".npy"))
            
    str_logs        = os.path.join(summaries, "STR", current_time)
    er_logs         = os.path.join(summaries, "ER", current_time)
    utils.makedirs(str_logs)
    
    ## combine all ECG data
    train_ECG   = np.vstack((swell_data[swell_train_index[k], 4:], amigos_data[amigos_train_index[k], 3:], dreamer_data[dreamer_train_index[k], 3:], wesad_data[wesad_train_index[k], 2:])) 
    test_ECG    = np.vstack((swell_data[swell_test_index[k], 4:],  amigos_data[amigos_test_index[k], 3:],  dreamer_data[dreamer_test_index[k], 3:],  wesad_data[wesad_test_index[k], 2:])) 
    train_ECG   = shuffle(train_ECG)
    
    ## fetch emotion recognition labels
    train_swell_input_stress, test_swell_input_stress = utils.one_hot_encoding(arr = swell_data[:, 1], tr_index = swell_train_index[k], te_index = swell_test_index[k])
    train_swell_arousal, test_swell_arousal           = utils.one_hot_encoding(arr = swell_data[:, 2], tr_index = swell_train_index[k], te_index = swell_test_index[k])
    train_swell_valence, test_swell_valence           = utils.one_hot_encoding(arr = swell_data[:, 3], tr_index = swell_train_index[k], te_index = swell_test_index[k])
    train_dreamer_arousal, test_dreamer_arousal       = utils.one_hot_encoding(arr = dreamer_data[:, 1], tr_index = dreamer_train_index[k], te_index = dreamer_test_index[k])
    train_dreamer_valence, test_dreamer_valence       = utils.one_hot_encoding(arr = dreamer_data[:, 2], tr_index = dreamer_train_index[k], te_index = dreamer_test_index[k])
    train_amigos_arousal, test_amigos_arousal         = utils.one_hot_encoding(arr = amigos_data[:, 1],  tr_index = amigos_train_index[k], te_index = amigos_test_index[k])
    train_amigos_valence, test_amigos_valence         = utils.one_hot_encoding(arr = amigos_data[:, 2],  tr_index = amigos_train_index[k], te_index = amigos_test_index[k])
    train_wesad_stress, test_wesad_stress             = utils.one_hot_encoding(arr = wesad_data[:, 1],  tr_index = wesad_train_index[k], te_index = wesad_test_index[k])
    
    training_length = train_ECG.shape[0]
    testing_length  = test_ECG.shape[0]
    
    print('Initializing all parameters.')
    tf.reset_default_graph()
    with tf.Session(graph=graph) as sess:   
        summary_writer = tf.compat.v1.summary.FileWriter(str_logs, sess.graph)
    
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        print('self supervised training started')
        
        train_loss_dict = {}
        test_loss_dict = {}
    
        tr_ssl_result = {}
        te_ssl_result = {}    
        
        ## epoch loop
        for epoch_counter in tqdm(range(epoch)):
            
            tr_loss_task = np.zeros((len(transform_task), 1), dtype  = np.float32)
            train_pred_task = np.zeros((len(transform_task), actual_batch_size), dtype  = np.float32) -1
            train_true_task = np.zeros((len(transform_task), actual_batch_size), dtype  = np.float32) -1
            tr_output_loss = 0
    
           
            tr_total_gen_op = utils.make_total_batch(data = train_ECG, length = training_length, batchsize = batchsize, 
                                               noise_amount=noise_param, 
                                               scaling_factor=scale_param, 
                                               permutation_pieces=permu_param, 
                                               time_warping_pieces=tw_piece_param, 
                                               time_warping_stretch_factor= twsf_param, 
                                               time_warping_squeeze_factor= 1/twsf_param)
    
            for training_batch, training_labels, tr_counter, tr_steps in tr_total_gen_op:
                
                ## run the model here 
                training_batch, training_labels = utils.unison_shuffled_copies(training_batch, training_labels)
                training_batch = training_batch.reshape(training_batch.shape[0], training_batch.shape[1], 1)
                fetches = [all_loss, output_loss, y_pred, train_op]
                if tr_counter % log_step == 0:
                    fetches.append(summary_op)
                    
                fetched = sess.run(fetches, {input_tensor: training_batch, y: training_labels, drop_out: drop_rate, isTrain: True})
                
                if tr_counter % log_step == 0: # 
                    summary_writer.add_summary(fetched[-1], tr_counter)
                    summary_writer.flush()
    
                tr_loss_task = utils.fetch_all_loss(fetched[0], tr_loss_task) 
                tr_output_loss += fetched[1]
                
                train_pred_task = utils.fetch_pred_labels(fetched[2], train_pred_task)
                train_true_task = utils.fetch_true_labels(training_labels, train_true_task)

            ## loss after epoch
            tr_epoch_loss = np.true_divide(tr_loss_task, tr_steps)
            train_loss_dict.update({epoch_counter: tr_epoch_loss})
            tr_output_loss = np.true_divide(tr_output_loss, tr_steps)
            
            ## performance matrix after each epoch
            tr_epoch_accuracy, tr_epoch_f1_score = utils.get_results_ssl(train_true_task, np.asarray(train_pred_task, int))
            tr_ssl_result = utils.write_result(tr_epoch_accuracy, tr_epoch_f1_score, epoch_counter, tr_ssl_result)
            utils.write_summary(loss = tr_epoch_loss, total_loss = tr_output_loss, f1_score = tr_epoch_f1_score, epoch_counter = epoch_counter, isTraining = True, summary_writer = summary_writer)
            utils.write_result_csv(k, epoch_counter, os.path.join(output, "STR_result", "tr_str_f1_Score.csv"), tr_epoch_f1_score)
    
            model_path = os.path.join(model_dir , "epoch_" + str(epoch_counter))
            utils.makedirs(model_path)
            save_path = saver.save(sess, os.path.join(model_path, "SSL_model.ckpt"))
            print("Self-supervised trained model is saved in path: %s" % save_path) 
            
            ## initialize array
            te_loss_task    = np.zeros((len(transform_task), 1), dtype  = np.float32)
            test_pred_task  = np.zeros((len(transform_task), actual_batch_size), dtype  = np.float32)-1
            test_true_task  = np.zeros((len(transform_task), actual_batch_size), dtype  = np.float32)-1
            te_output_loss  = 0
           
            te_total_gen_op = utils.make_total_batch(data = test_ECG, 
                                                     length = testing_length, 
                                                     batchsize = batchsize, 
                                                     noise_amount=noise_param, 
                                                     scaling_factor=scale_param, 
                                                     permutation_pieces=permu_param, 
                                                     time_warping_pieces=tw_piece_param, 
                                                     time_warping_stretch_factor= twsf_param, 
                                                     time_warping_squeeze_factor= 1/twsf_param)
    
            for testing_batch, testing_labels, te_counter, te_steps in te_total_gen_op:
                
                ## run the model here 
                fetches = [all_loss, output_loss, y_pred]
                    
                fetched = sess.run(fetches, {input_tensor: testing_batch, y: testing_labels, drop_out: 0.0, isTrain: False})
    
                te_loss_task = utils.fetch_all_loss(fetched[0], te_loss_task)
                te_output_loss += fetched[1]
                test_pred_task = utils.fetch_pred_labels(fetched[2], test_pred_task)
                test_true_task = utils.fetch_true_labels(testing_labels, test_true_task)
    
            ## loss after epoch
            te_epoch_loss = np.true_divide(te_loss_task, te_steps)
            test_loss_dict.update({epoch_counter: te_epoch_loss})
            te_output_loss = np.true_divide(te_output_loss, te_steps)
    
            ## performance matrix after each epoch
            te_epoch_accuracy, te_epoch_f1_score = utils.get_results_ssl(test_true_task, test_pred_task)            
            te_ssl_result = utils.write_result(te_epoch_accuracy, te_epoch_f1_score, epoch_counter, te_ssl_result)    
            utils.write_summary(loss = te_epoch_loss, total_loss = te_output_loss, f1_score = te_epoch_f1_score, epoch_counter = epoch_counter, isTraining = False, summary_writer = summary_writer)
            utils.write_result_csv(k, epoch_counter, os.path.join(output, "STR_result", "te_str_f1_score.csv"), te_epoch_f1_score)
            
    
            if 1==1:
                """
                supervised task of self supervised learning
                """
                """  swell """
               
                ## training - testing ECG
                x_tr = swell_data[swell_train_index[k], 4:]
                x_te = swell_data[swell_test_index[k], 4:]
                
                ## features extracted from conv layers
                x_tr_feature = utils.extract_feature(x_original = x_tr, featureset_size = featureset_size, batch_super = batchsize, input_tensor = input_tensor, isTrain = isTrain, drop_out = drop_out, extract_layer = main_branch, sess = sess)
                x_te_feature = utils.extract_feature(x_original = x_te, featureset_size = featureset_size, batch_super = batchsize, input_tensor = input_tensor, isTrain = isTrain, drop_out = drop_out, extract_layer = main_branch, sess = sess)
                
                ## supervised emotion recognition
                model.supervised_model_swell(x_tr_feature = x_tr_feature, y_tr = train_swell_input_stress, x_te_feature = x_te_feature, y_te = test_swell_input_stress, identifier = 'swell_input_stress', kfold = flag, result = output, summaries = er_logs, current_time = current_time)        
                model.supervised_model_swell(x_tr_feature = x_tr_feature, y_tr = train_swell_arousal, x_te_feature = x_te_feature, y_te = test_swell_arousal, identifier = 'swell_arousal', kfold = flag, result = output, summaries = er_logs, current_time = current_time)  
                model.supervised_model_swell(x_tr_feature = x_tr_feature, y_tr = train_swell_valence, x_te_feature = x_te_feature, y_te = test_swell_valence, identifier = 'swell_valence', kfold = flag, result = output, summaries = er_logs, current_time = current_time)  
        
                
                """
                supervised task of self supervised learning
                """
                """  wesad """  

                ## training - testing ECG                
                x_tr = wesad_data[wesad_train_index[k], 2:]
                x_te = wesad_data[wesad_test_index[k], 2:]
                
                ## features extracted from conv layers
                x_tr_feature = utils.extract_feature(x_original = x_tr, featureset_size = featureset_size, batch_super = batchsize, input_tensor = input_tensor, isTrain = isTrain, drop_out = drop_out, extract_layer = main_branch, sess = sess)
                x_te_feature = utils.extract_feature(x_original = x_te, featureset_size = featureset_size, batch_super = batchsize, input_tensor = input_tensor, isTrain = isTrain, drop_out = drop_out, extract_layer = main_branch, sess = sess)
                
                ## supervised emotion recognition
                model.supervised_model_wesad(x_tr_feature = x_tr_feature, y_tr = train_wesad_stress, x_te_feature = x_te_feature, y_te = test_wesad_stress, identifier = 'wesad_affect', kfold = flag, result = output, summaries = er_logs, current_time = current_time)  
#        
                
                """
                supervised task of self supervised learning
                """
                """  dreamer """  
                
                ## training - testing ECG                
                x_tr = dreamer_data[dreamer_train_index[k], 3:]
                x_te = dreamer_data[dreamer_test_index[k], 3:]
                    
                ## features extracted from conv layers
                x_tr_feature = utils.extract_feature(x_original = x_tr, featureset_size = featureset_size, batch_super = batchsize, input_tensor = input_tensor, isTrain = isTrain, drop_out = drop_out, extract_layer = main_branch, sess = sess)
                x_te_feature = utils.extract_feature(x_original = x_te, featureset_size = featureset_size, batch_super = batchsize, input_tensor = input_tensor, isTrain = isTrain, drop_out = drop_out, extract_layer = main_branch, sess = sess)
                
                ## supervised emotion recognition
                model.supervised_model_dreamer(x_tr_feature = x_tr_feature, y_tr = train_dreamer_arousal, x_te_feature = x_te_feature, y_te = test_dreamer_arousal, identifier = 'dreamer_arousal', kfold = flag, result = output, summaries = er_logs, current_time = current_time)  
                model.supervised_model_dreamer(x_tr_feature = x_tr_feature, y_tr = train_dreamer_valence, x_te_feature = x_te_feature, y_te = test_dreamer_valence, identifier = 'dreamer_valence', kfold = flag, result = output, summaries = er_logs, current_time = current_time)  
                
        
                """
                supervised task of self supervised learning
                """
                """  amigos """  

                ## training - testing ECG                
                x_tr = amigos_data[amigos_train_index[k], 3:]
                x_te = amigos_data[amigos_test_index[k], 3:]
                    
                ## features extracted from conv layers
                x_tr_feature = utils.extract_feature(x_original = x_tr, featureset_size = featureset_size, batch_super = batchsize, input_tensor = input_tensor, isTrain = isTrain, drop_out = drop_out, extract_layer = main_branch, sess = sess)
                x_te_feature = utils.extract_feature(x_original = x_te, featureset_size = featureset_size, batch_super = batchsize, input_tensor = input_tensor, isTrain = isTrain, drop_out = drop_out, extract_layer = main_branch, sess = sess)
                
                ## supervised emotion recognition
                model.supervised_model_amigos(x_tr_feature = x_tr_feature, y_tr = train_amigos_arousal, x_te_feature = x_te_feature, y_te = test_amigos_arousal, identifier = 'amigos_arousal', kfold = flag, result = output, summaries = er_logs, current_time = current_time)  
                model.supervised_model_amigos(x_tr_feature = x_tr_feature, y_tr = train_amigos_valence, x_te_feature = x_te_feature, y_te = test_amigos_valence, identifier = 'amigos_valence', kfold = flag, result = output, summaries = er_logs, current_time = current_time)  

        ## save str loss, acc and f1 score    
        np.save(tr_ssl_loss_filename, train_loss_dict)
        np.save(te_ssl_loss_filename, test_loss_dict)
    
        np.save(tr_ssl_result_filename, tr_ssl_result)
        np.save(te_ssl_result_filename, te_ssl_result)


