# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:29:28 2019

@author: Pritam
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
import utils

def import_filenames(directory_path):
    """ 
    import all file names of a directory """
    filename_list = []
    dir_list      = []
    for root, dirs, files in os.walk(directory_path, topdown=False):
        filename_list   = files     
        dir_list        = dirs
    return filename_list, dir_list
   
def normalize(x, x_mean, x_std):
    """ 
    perform z-score normalization of a signal """
    x_scaled = (x-x_mean)/x_std
    return x_scaled

def make_window(signal, fs, overlap, window_size_sec):
    """ 
    perform cropped signals of window_size seconds for the whole signal
    overlap input is in percentage of window_size
    window_size is in seconds """
    
    window_size = fs * window_size_sec
    overlap     = int(window_size * (overlap / 100))
    start       = 0   
    segmented   = np.zeros((1, window_size), dtype = int)
    while(start+window_size <= len(signal)):
        segment     = signal[start:start+window_size]
        segment     = segment.reshape(1, len(segment))
        segmented   = np.append(segmented, segment, axis =0)
        start       = start + window_size - overlap
    return segmented[1:]


        
        
def extract_swell_dataset(overlap_pct, window_size_sec, data_save_path, save):

    print("SWELL")
    
    swell_path = "set_your_path\\final_SWELL\\filtered_ecg\\"
    swell_labels_path = "set_your_path\\final_SWELL\\label\\behavioral-labels.xlsx"
    utils.makedirs(data_save_path)
    freq = 256
    window_size = window_size_sec * freq
    swell_file_names, _ = import_filenames(swell_path)
    person_name = []
    for i in swell_file_names:
        person_name.append(i[:i.find('_')])
        
    person = np.unique(person_name)
    k = 0
    swell_norm = np.empty((person.shape[0], 3))
    for i in tqdm(person):
        counter =0
        print(i)
        for j in swell_file_names:
            if j[:j.find('_')] == i:
                signal = np.loadtxt(swell_path + j)
                print(j)
                if counter == 0:
                    data = signal
                else:
                    data = np.vstack((data, signal))
                    counter = 1
        
        data = np.sort(data)
        std = np.std(data[np.int(0.025*data.shape[0]) : np.int(0.975*data.shape[0])])
        mean = np.mean(data)
        swell_norm[k, :] = [np.int(i[2:]), mean, std]
        k = k+1
        
        
    swell_dict = {}
    
    for i in tqdm(swell_file_names):
        name = np.int(i[2:i.find('_')])
        x_mean = swell_norm[np.where(swell_norm[:,0] == name)][:, 1][0]
        x_std  = swell_norm[np.where(swell_norm[:,0] == name)][:, 2][0]
        data = np.loadtxt(swell_path + i)
        data = normalize(data, x_mean, x_std)
        data_windowed = make_window (data, freq, overlap_pct, window_size_sec)
        swell_dict.update({i: data_windowed})
        
    
    counter = 0;
    label = pd.ExcelFile(swell_labels_path)
    label_sheet_names = label.sheet_names
    participant_labellings = pd.DataFrame
    
    print('getting labels...')
    for i in tqdm(range(len(label_sheet_names))):
        participant_labellings = label.parse(label_sheet_names[i])
        if counter == 0:
            labels = participant_labellings
        else:
            labels = labels.append(participant_labellings, ignore_index = True, sort=False)
        counter = counter + 1;

    swell_labels = labels.drop_duplicates(subset = ['PP','Blok'], keep = 'last')
    swell_labels = swell_labels.reset_index(drop = True)
    counter = 0
    
    #adding csv names into labels
    swell_labels['filename'] = 'default'
    
    for i in swell_file_names:
        start = i.find('_')
        end = i.rfind('c')
        condition = (swell_labels['PP'] == i[:start].upper()) & (swell_labels['Blok'] == int(i[end+1:-4]))
        index = np.where(condition)[0]
        if len(index) != 0:
            swell_labels['filename'].iloc[index[0]] = i

    
    print('dict unpacking...')
    
    final_set = np.zeros((1, window_size+12), dtype = int)
    key_list = swell_dict.keys()
    for i in tqdm(key_list):
        new_key = np.float(i[i.find('pp')+2:i.find('_')] + "." + i[i.find('c')+1:-4])
        values = swell_dict[i]
        key = np.repeat(new_key, len(values))
        key = key.reshape(len(key), 1)
        label_set = swell_labels[(swell_labels['PP'] == i[:i.find('_')].upper()) & (swell_labels['Blok'] == np.int(i[i.find('c')+1:-4]))]
        label_set = label_set[['Valence_rc', 'Arousal_rc', 'Dominance', 'Stress', 'MentalEffort', 'MentalDemand', 'PhysicalDemand', 'TemporalDemand', 'Effort','Performance_rc', 'Frustration']]
        label_set = pd.concat([label_set]*len(values), ignore_index=True)
        label_set = np.asarray(label_set)
        signal_set = np.hstack((key, label_set, values))
        final_set = np.vstack((final_set, signal_set))
        
    final_set = final_set[1:]
        
    if save:
        np.save(data_save_path / 'swell_dict.npy', final_set)          
        
    print('swell files importing finished...')
    return final_set



def extract_dreamer_dataset(overlap_pct, window_size_sec, data_save_path, save):

    print("DREAMER")

    dreamer_path = "set_your_path\\final_DREAMER\\filtered_ecg\\"
    dreamer_labels_path = "set_your_path\\final_DREAMER\\labels\\"
    utils.makedirs(data_save_path)
    freq = 256
    window_size = window_size_sec * freq # sampling freq is always 256

    dreamer_file_names, _ = import_filenames(dreamer_path)
    person_name = []
    for i in dreamer_file_names:
        person_name.append(i[:i.find('_')])
        
    person = np.unique(person_name)
    k = 0
    dreamer_norm = np.empty((person.shape[0], 3))
    for i in tqdm(person):
        counter =0
        print(i)
        for j in dreamer_file_names:
            if j[:j.find('_')] == i:
                signal = np.loadtxt(dreamer_path + j)
                print(j)
                if counter == 0:
                    data = signal
                else:
                    data = np.vstack((data, signal))
                    counter = 1
        
        data = np.sort(data)
        std = np.std(data[np.int(0.025*data.shape[0]) : np.int(0.975*data.shape[0])])
        mean = np.mean(data)
        dreamer_norm[k, :] = [np.int(i[2:]), mean, std]
        k = k+1
        
        
    dreamer_dict = {}


    for i in tqdm(dreamer_file_names):
        
        name = np.int(i[2:i.find('_')])
        x_mean = dreamer_norm[np.where(dreamer_norm[:,0] == name)][:, 1][0]
        x_std  = dreamer_norm[np.where(dreamer_norm[:,0] == name)][:, 2][0]
        
        data = np.loadtxt(dreamer_path + i)
        data = normalize(data, x_mean, x_std)
        data_windowed = make_window (data, freq, overlap_pct, window_size_sec)
        dreamer_dict.update({i: data_windowed})

 
      
    ## dreamer label information
    dreamer_labels_dict = {}
    dreamer_label_names, _ = import_filenames(dreamer_labels_path)
    for i in dreamer_label_names:
        dreamer_label = pd.read_csv(dreamer_labels_path + i, sep = ',')
        for j in range(len(dreamer_label)):
            label_key = i[:-4] + '_clips' + str(j+1) + '.txt'
            dreamer_labels_dict.update({label_key:dreamer_label.loc[j,:]})
            
    keys = dreamer_labels_dict.keys()
    
    ## load in a dataframe
    label_df = pd.DataFrame(columns = ['filename', 'Arousal', 'Dominance', 'Valence'])
    counter = 0
    for i in keys:
        index = i.find('_')
        key = i[2:index] + '.' + i[index+6: -4]
        label_df.loc[counter, 'filename'] = key
        label_df.loc[counter, 'Arousal'] = dreamer_labels_dict[i].values[0]
        label_df.loc[counter, 'Dominance'] = dreamer_labels_dict[i].values[1]
        label_df.loc[counter, 'Valence'] = dreamer_labels_dict[i].values[2]
        counter = counter + 1
    
    print('dict unpacking...')

    ## data loading with file name
    final_set = np.zeros((1, window_size+2), dtype = int)
    for i in tqdm(dreamer_dict.keys()):
        values = dreamer_dict[i]
        index = i.find('_')
        person_id = np.int(i[2:index])
        clip = np.int(i[index+6: -4])
        key = np.repeat(np.array([[person_id, clip]]), len(values), axis=0)
        signal_set = np.hstack((key, values))
    #    final_training_set = np.append(final_training_set, training_set, axis = 0)
        final_set = np.vstack((final_set, signal_set))
    
    ## first column stands for labels: XX.CC == XX person id, and CC clips
    final_set = final_set[1:]
    
    file_id = final_set[:, :2]

    y = np.zeros((1, 4)) ## ['person_id', 'Arousal', 'Dominance', 'Valence']

    print('labels are getting matched with signals...')
    for i in tqdm(range(len(final_set))):
        temp = [[file_id[i, 0], int(label_df[label_df.filename == str(np.int(file_id[i,0])) + '.' + str(np.int(file_id[i,1]))].Arousal.values[0]), int(label_df[label_df.filename == str(np.int(file_id[i,0])) + '.' + str(np.int(file_id[i,1]))].Valence.values[0]), int(label_df[label_df.filename == str(np.int(file_id[i,0])) + '.' + str(np.int(file_id[i,1]))].Dominance.values[0])]]
        temp = np.array(temp)
        y = np.append(y, temp, axis = 0)
        
    y = y[1:]

    temp = final_set
    final_set = np.hstack((y, temp[:, 2:]))

    if save:

        np.save(data_save_path / 'dreamer_dict.npy', final_set)
    
    print('dreamer files importing finished')
    return final_set

def extract_amigos_dataset(overlap_pct, window_size_sec, data_save_path, save):
    print("AMIGOS")
    amigos_path = "set_your_path\\final_AMIGOS\\filtered_ecg\\"
    amigos_labels_path = "set_your_path\\final_AMIGOS\\labels\\amigos_labels.xlsx"
    freq = 256
    utils.makedirs(data_save_path)
    window_size = window_size_sec * freq
    
    amigos_file_names, _ = import_filenames(amigos_path)
    person_name = []
    for i in amigos_file_names:
        person_name.append(i[:i.find('_')])
        
    person = np.unique(person_name)
    k = 0
    amigos_norm = np.empty((person.shape[0], 3))
    for i in tqdm(person):
        counter =0
        print(i)
        for j in amigos_file_names:
            if j[:j.find('_')] == i:
                signal = np.loadtxt(amigos_path + j)
                print(j)
                if counter == 0:
                    data = signal
                else:
                    data = np.vstack((data, signal))
                    counter = 1
        
        data = np.sort(data)
        std = np.std(data[np.int(0.025*data.shape[0]) : np.int(0.975*data.shape[0])])
        mean = np.mean(data)
        amigos_norm[k, :] = [np.int(i[2:]), mean, std]
        k = k+1
        
    amigos_dict = {}
    
    for i in tqdm(amigos_file_names):
        
        name = np.int(i[2:i.find('_')])
        x_mean = amigos_norm[np.where(amigos_norm[:,0] == name)][:, 1][0]
        x_std  = amigos_norm[np.where(amigos_norm[:,0] == name)][:, 2][0]
        
        data = np.loadtxt(amigos_path + i)
        data = normalize(data, x_mean, x_std)
        data_windowed = make_window (data, freq, overlap_pct, window_size_sec)
        amigos_dict.update({i: data_windowed})
    
    labels = pd.read_excel(amigos_labels_path, index_col=0)
    labels.reset_index(drop = True)
    final_set = np.zeros((1, window_size+4), dtype = int)
    labels['VideoID'] = labels['VideoID'].map(lambda x: x.lstrip("'").rstrip("'"))
    
    ## data loading with file name
    final_set = np.zeros((1, window_size+4), dtype = int)
    for i in tqdm(amigos_dict.keys()):
        values = amigos_dict[i]
        index = i.find('_')
        person_id = np.int(i[1:index])
        clip = i[index+1: -4]
        cond = labels[(labels.UserID == person_id) & (labels.VideoID == clip)]
        if not cond.empty:
            arousal = labels[(labels.UserID == person_id) & (labels.VideoID == clip)].arousal.values[0]
            valence = labels[(labels.UserID == person_id) & (labels.VideoID == clip)].valence.values[0]
            dominance = labels[(labels.UserID == person_id) & (labels.VideoID == clip)].dominance.values[0]
            key = np.repeat(np.array([[person_id, arousal, valence, dominance]]), len(values), axis=0)
            signal_set = np.hstack((key, values))
        #    final_training_set = np.append(final_training_set, training_set, axis = 0)
            final_set = np.vstack((final_set, signal_set))
    
    ## first column stands for labels: XX.CC == XX person id, and CC clips
    final_set = final_set[1:]
        
    if save:
        np.save(data_save_path / 'amigos_dict.npy', final_set)  

    print('amigos files importing finished')
    return final_set   


def extract_wesad_dataset(overlap_pct, window_size_sec, data_save_path, save):

    print('WESAD')
    
    wesad_path = "set_your_path\\final_WESAD\\filtered_ecg\\"
    wesad_labels_path= "set_your_path\\final_WESAD\\labels\\"
    freq = 256
    utils.makedirs(data_save_path)
    window_size = window_size_sec * freq
    
    wesad_file_names, _ = import_filenames(wesad_path)
    
    wesad_dict = {}
    wesad_labels = {}
     
        
    for i in tqdm(wesad_file_names):
        x_mean = np
        data = np.loadtxt(wesad_path + i)
        sort_data = np.sort(data)
        x_std = np.std(sort_data[np.int(0.025*sort_data.shape[0]) : np.int(0.975*sort_data.shape[0])])
        x_mean = np.mean(sort_data)
        data = normalize(data, x_mean, x_std)
        labels = np.loadtxt(wesad_labels_path + i)
        data_windowed   =    make_window (data, freq, overlap_pct, window_size_sec)
        labels_windowed =    make_window (labels, freq, overlap_pct, window_size_sec)

        wesad_dict.update({i: data_windowed})
        wesad_labels.update({i: labels_windowed})
        
    print('dict unpacking...')
    final_set = np.zeros((1, window_size+2), dtype = int)
    for i in tqdm(wesad_dict.keys()):
        values = wesad_dict[i]
        labels = wesad_labels[i]
        index = i.find('.')
        key = i[1:index]
        key = np.repeat(key, len(values))
        key = key.astype(float)
        key = key.reshape(len(key), 1)
        labels_max = np.amax(labels, axis = 1)
        labels_max = labels_max.reshape(len(labels_max), 1)

        signal_set = np.hstack((key, labels_max, values))
    #    final_training_set = np.append(final_training_set, training_set, axis = 0)
        final_set = np.vstack((final_set, signal_set))
    
    ## first column stands for labels: XX.CC == XX person id, and CC clips
    final_set = final_set[1:]

    
    if save:
        np.save(data_save_path / 'wesad_dict.npy', final_set)

    print('wesad files importing finished')
    return final_set
       
def load_data(path):
    dataset = np.load(path, allow_pickle=True)     
    return dataset   

def swell_prepare_for_10fold(swell_data):
    
    ecg = swell_data[:, 12:]
    
    """ 'person.blok', 'Valence_rc', 'Arousal_rc', 'Dominance' """
    """ 'person.blok', 'Valence_rc', 'Arousal_rc', 'Dominance', 'Stress', 'MentalEffort', 'MentalDemand', 'PhysicalDemand', 'TemporalDemand', 'Effort','Performance_rc', 'Frustration' """

    person               = np.floor(swell_data[:,0])
    y_input_stress       = (swell_data[:, 0]*10 - np.round(swell_data[:, 0])*10).astype(int)
    y_arousal            = swell_data[:, 2]
    y_valence            = swell_data[:, 1]
    person               = person.reshape(-1, 1)
    y_input_stress       = y_input_stress.reshape(-1, 1)
    y_arousal            = y_arousal.astype(int).reshape(-1, 1)
    y_valence            = y_valence.astype(int).reshape(-1, 1)
    swell_data  = np.hstack((person, y_input_stress, y_arousal, y_valence, ecg))
    return swell_data 


def wesad_prepare_for_10fold(wesad_data, numb_class=4):
    
    person = wesad_data[:, 0]
    y_stress = wesad_data[:, 1]    
    ecg = wesad_data[:, 2:]

    ecg             = ecg[(y_stress != 0) & (y_stress != 5) & (y_stress != 6) & (y_stress != 7)]
    person          = person[(y_stress != 0) & (y_stress != 5) & (y_stress != 6) & (y_stress != 7)].reshape(-1, 1)
    y_stress        = y_stress[(y_stress != 0) & (y_stress != 5) & (y_stress != 6) & (y_stress != 7)] - 1
    y_stress        = y_stress.reshape(-1, 1)
    
    wesad_data      = np.hstack((person, y_stress, ecg))
    
    return wesad_data # 4 class

def dreamer_prepare_for_10fold(dreamer_data):
    
    ecg = dreamer_data[:, 4:]
    
    """ 'person', 'Arousal', 'Dominance',  'Valence' """
    
    person    = dreamer_data[:, 0]
    y_arousal = dreamer_data[:, 1]
    y_valence = dreamer_data[:, 3]
    person = person.reshape(-1, 1)
    y_arousal = y_arousal.astype(int).reshape(-1, 1)
    y_valence = y_valence.astype(int).reshape(-1, 1)    
    dreamer_data = np.hstack((person, y_arousal, y_valence, ecg))
    return dreamer_data # binary

def amigos_prepare_for_10fold(amigos_data):
    
    ecg = amigos_data[:, 4:]
    
    """ 'Arousal', 'Dominance',  'Valence' """
    
    person          = amigos_data[:, 0]
    y_arousal       = np.round(amigos_data[:, 1],0)
    y_valence       = np.round(amigos_data[:, 3],0)
    person      = person.reshape(-1, 1)
    y_arousal   = y_arousal.astype(int).reshape(-1, 1)
    y_valence   = y_valence.astype(int).reshape(-1, 1)    
    amigos_data = np.hstack((person, y_arousal, y_valence, ecg))
    
    return amigos_data


def save_list(mylist, filename):
    for i in range(len(mylist)):
        temp = mylist[i]
        with open(filename, 'a', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(temp)
    return               
