# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 20:49:02 2020

@author: pritam
"""

import os
import tensorflow as tf
import numpy as np
import csv
from sklearn import metrics
import signal_transformation_task as stt
from mlxtend.evaluate import confusion_matrix
import time

window_size = 2560
transform_task = [0, 1, 2, 3, 4, 5, 6]
      
    
def get_label(y, actual_batch_size):
    """ get the label or y true """
    
    y_label = []
    for i in range(len(transform_task)):
        label = tf.reshape(y[:,i], [actual_batch_size,1])
        y_label.append(label)
    return y_label

def calculate_loss(y_label, logits):
    """ calculate loss of each transformtion task """
    all_loss = []
    for i in range(len(transform_task)):
        loss = tf.reduce_mean(tf.compat.v1.losses.sigmoid_cross_entropy(multi_class_labels=y_label[i], logits=logits[i]))
        all_loss.append(loss)
    return all_loss


def get_prediction(logits):
    """ get the prediction of the model"""
    y_pred = []
    for i in range(len(transform_task)):
        pred = tf.greater(tf.nn.sigmoid(logits[i]), 0.5)
        y_pred.append(pred)
    return y_pred

def make_batch(signal_batch, noise_amount, scaling_factor, permutation_pieces, time_warping_pieces, time_warping_stretch_factor, time_warping_squeeze_factor):
    """
    genrator to do signal transformation and then make a batch of 7, which is a batch contains all transformed signal with original signal"""
    
    for i in range(len(signal_batch)):
        
        signal              = signal_batch[i]
        signal              = np.trim_zeros(signal, 'b')
        sampling_freq       = len(signal)//10
        noised_signal       = stt.add_noise_with_SNR(signal, noise_amount = noise_amount) #round(np.random.uniform(0.005,0.05),2)) # 0.005 - 0.05
        scaled_signal       = stt.scaled(signal, factor = scaling_factor) #round(np.random.uniform(0.2,2),2)) # 0.2 - 2
        negated_signal      = stt.negate(signal)
        flipped_signal      = stt.hor_filp(signal)
        permuted_signal     = stt.permute(signal, pieces = permutation_pieces) # 2-20
        time_warped_signal  = stt.time_warp(signal, sampling_freq, pieces = time_warping_pieces, stretch_factor = time_warping_stretch_factor, squeeze_factor = time_warping_squeeze_factor)
        
        ## making signals of same size.. 
        tw_start_index      = np.int(np.random.randint(0, (len(time_warped_signal)-len(signal))))
        tw_stop_index       = np.int(tw_start_index + len(signal))
        time_warped_signal  = time_warped_signal[tw_start_index:tw_stop_index]

        
        signal                  = signal.reshape(len(signal), 1)
        noised_signal           = noised_signal.reshape(len(noised_signal), 1)
        scaled_signal           = scaled_signal.reshape(len(scaled_signal), 1)
        negated_signal          = negated_signal.reshape(len(negated_signal), 1)
        flipped_signal          = flipped_signal.reshape(len(flipped_signal), 1)
        permuted_signal         = permuted_signal.reshape(len(permuted_signal), 1)
        time_warped_signal      = time_warped_signal.reshape(len(time_warped_signal), 1)
                    
        
        batch = [signal, noised_signal, scaled_signal, negated_signal, flipped_signal, permuted_signal, time_warped_signal]
        labels = transform_task
        labels = tf.keras.utils.to_categorical(labels) 

        ## padding the transformed signal batch 
        batch = tf.keras.preprocessing.sequence.pad_sequences(batch, dtype='float32', padding='post')
    
        yield batch, labels


def make_total_batch(data, length, batchsize, noise_amount, scaling_factor, permutation_pieces, time_warping_pieces, time_warping_stretch_factor, time_warping_squeeze_factor):
    """ calling make_batch from here, when batch size is more than one, like 64 or 32, it will make actual batch size = batch_size * len(transformed signal)
    """
    steps = length // batchsize +1
    for counter in range(steps):

        signal_batch = data[np.mod(np.arange(counter*batchsize,(counter+1)*batchsize), length)]
        
        gen_op =  make_batch(signal_batch, noise_amount, scaling_factor, permutation_pieces, time_warping_pieces, time_warping_stretch_factor, time_warping_squeeze_factor)
        total_batch = np.array([])
        total_labels = np.array([])
        for batch, labels in gen_op:

            total_batch = np.vstack((total_batch, batch)) if total_batch.size else batch
            total_labels = np.vstack((total_labels, labels)) if total_labels.size else labels

        yield total_batch, total_labels, counter, steps

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def get_weighted_loss(loss_coeff, all_loss):
    """ calculate the weighted loss
    """
    
    output_loss = 0
    for i in range(len(loss_coeff)):
        temp = loss_coeff[i]*all_loss[i]
        output_loss = output_loss + temp
    
    return output_loss
        
def fetch_all_loss(all_losses, loss_task):
    """
    fetch individual signal transformation losses"""
    
    for i in range(len(transform_task)):
        loss_task[i] = np.add(loss_task[i], all_losses[i])
    return loss_task

def fetch_pred_labels(y_preds, pred_task):
    y_preds = np.squeeze(np.asarray(y_preds, dtype=np.int32)).T
    if np.all(pred_task==-1):
        pred_task = y_preds
    else:
        pred_task = np.vstack((pred_task, y_preds))    

    return pred_task

def fetch_true_labels(labels, true_task):
    if np.all(true_task==-1):
        true_task = labels
    else:
        true_task = np.vstack((true_task, labels))     
    
    return true_task

def get_results_ssl(y_true, y_pred):
    accuracy = np.full((1, 7), np.nan)
    f1_score = np.full((1, 7), np.nan)
    
    if y_true.shape == y_pred.shape:
        for i in range(len(transform_task)):
            accuracy[:, i]    = np.round(metrics.accuracy_score        (y_true[:, i], y_pred[:, i]), 2)
            f1_score[:, i]    = np.round(metrics.f1_score              (y_true[:, i], y_pred[:, i], labels = [0, 1]), 2)
    else:
        print("error in self supervised result calculation")
        
    return accuracy, f1_score

def write_result(accuracy, f1_score, epoch_number, result_dict):
    result = [accuracy, f1_score]
    result_dict.update({epoch_number: result})
    return result_dict
    

def write_summary(loss, total_loss, f1_score, epoch_counter, isTraining, summary_writer):

    task_name = ['original', 'noised', 'scaled', 'negated', 'flipped', 'permuted', 'timewarp']
    for i in range(len(task_name)):
        t = task_name[i]
        if isTraining:
            summary = tf.Summary(value=[tf.Summary.Value(tag="train loss/"+ t,          simple_value        =   loss[i][0])])
            summary_writer.add_summary(summary, epoch_counter)
            summary_writer.flush()

            summary = tf.Summary(value=[tf.Summary.Value(tag="train F1 score/"+ t,      simple_value        =   f1_score[0][i])])
            summary_writer.add_summary(summary, epoch_counter)
            summary_writer.flush()
            
        else:
            summary = tf.Summary(value=[tf.Summary.Value(tag="test loss/"+ t,           simple_value        =   loss[i][0])])
            summary_writer.add_summary(summary, epoch_counter)
            summary_writer.flush()

            summary = tf.Summary(value=[tf.Summary.Value(tag="test F1 score/"+ t,       simple_value        =   f1_score[0][i])])
            summary_writer.add_summary(summary, epoch_counter)
            summary_writer.flush()
            
    if isTraining:
        summary = tf.Summary(value=[tf.Summary.Value(tag="train loss/total_loss",          simple_value        =   total_loss)])
        summary_writer.add_summary(summary, epoch_counter)
        summary_writer.flush()        
    else:
        summary = tf.Summary(value=[tf.Summary.Value(tag="test loss/total_loss",           simple_value        =   total_loss)])
        summary_writer.add_summary(summary, epoch_counter)
        summary_writer.flush()

    return


def write_result_csv(kfold, epoch_number, result_store, f1_score):
    f1_score = f1_score[0]
    with open(result_store, 'a', newline='') as csvfile:
        fieldnames = ['fold', 'epoch', 'org', 'noised', 'scaled', 'neg', 'flip', 'perm', 'time_warp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'fold': kfold, 'epoch': epoch_number,  'org': f1_score[0], 'noised': f1_score[1], 'scaled': f1_score[2], 'neg': f1_score[3], 'flip': f1_score[4], 'perm': f1_score[5], 'time_warp': f1_score[6]})
                
    return 

def model_result_store(y, y_pred, result_store, kfold):

    accuracy    = np.round(metrics.accuracy_score        (y, y_pred), 4)
    conf_mat    = confusion_matrix(y_target=y, y_predicted=y_pred, binary=False)
    precision   = np.round(np.mean(np.diag(conf_mat) / np.sum(conf_mat, axis = 0)), 4)
    recall      = np.round(np.mean(np.diag(conf_mat) / np.sum(conf_mat, axis = 1)), 4)
    f1_score    = np.round(2*precision*recall / (precision + recall), 4)
    
    with open(result_store, 'a', newline='') as csvfile:
        fieldnames = ['fold', 'accuracy','precision', 'recall', 'f1_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'fold': kfold, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score' : f1_score})
                
    return 


def current_time():
    """ taking the current system time"""
    cur_time = time.strftime("%Y_%m_%d_%H_%M", time.gmtime())
    return cur_time

def one_hot_encoding(arr, tr_index, te_index):
    
    num_of_class = len(np.unique(arr))
    min_val      = np.min(arr)
    arr          = arr - min_val
    tr_encoded_array = tf.keras.utils.to_categorical(arr[tr_index], num_classes = num_of_class) 
    te_encoded_array = tf.keras.utils.to_categorical(arr[te_index], num_classes = num_of_class) 

    return tr_encoded_array, te_encoded_array

def makedirs(path):
    """ 
    create directory on the "path name" """
    
    if not os.path.exists(path):
        os.makedirs(path)
        
def get_train_test_index(data, kf):
    train_index = []
    test_index = []
    for train_i, test_i in kf.split(data):
        train_index.append(train_i)
        test_index.append(test_i)

    return train_index, test_index

def extract_feature(x_original, featureset_size, batch_super, input_tensor, isTrain, drop_out, extract_layer, sess):
    feature_set = np.zeros((1, featureset_size), dtype = int) 
    length = np.shape(x_original)[0]
    steps = length //batch_super +1  
    for j in range(steps):
        signal_batch = x_original[np.mod(np.arange(j*batch_super,(j+1)*batch_super), length)]
        signal_batch = signal_batch.reshape(np.shape(signal_batch)[0], np.shape(signal_batch)[1], 1)
        fetched = sess.run(extract_layer, {input_tensor: signal_batch, isTrain: False, drop_out: 0.0})
        feature_set = np.vstack((feature_set, fetched))
         
    x_feature = feature_set[1:length+1] ## resizing to the original signal
    
    return x_feature
