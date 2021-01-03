# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:09:52 2020

@author: emmis
"""

from mido import MidiFile
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, pickle, sys, json, copy
import numpy as np

#%%

def MAESTRO_midi_graph(file_name, plot_type='jointplot', axes_=False, 
                       palette='icefire', gridsize=88, figwidth=20, 
                       figheight=10):
    
    #Import and parse MIDI file using MidiFile from the mido package
    mid = MidiFile(file_name) 
    
    #Filter the out the meta data in Track 0
    message_list = []
    for i in mid.tracks[1][1:-1]: 
        message_list.append(i)   
    
    #Transform the MIDI messages to strings
    message_strings = []
    for x in message_list:
        message_strings.append(str(x))
    
    #Split the message strings into attributes. The first attribute is the 
    #message type. Only the value of the message type is provided. The other 
    #attributes are listed as keys and values separated by '='.
    message_strings_split = []
    for message in message_strings:  
        split_str = message.split(" ")
        message_strings_split.append(split_str)
    
    #Slice the first attribute (message type) and transform it into a dataframe.
    message_type = []
    for item in message_strings_split:
        message_type.append(item[0])
    df1 = pd.DataFrame(message_type)
    df1.columns = ['message_type']
   
    #Slice the other attirubtes and store them in a list, 
    #one list for each message.
    attributes = []
    for item in message_strings_split:
        attributes.append(item[1:])
    
    #Transform the attribute lists above into dictionaries. 
    #The elements in the attribute list are split into key-value pairs
    #by the = sign.
    attributes_dict = [{}]    
    for item in attributes:
        for i in item:
            key, val = i.split("=")
            if key in attributes_dict[-1]:
                attributes_dict.append({})
            attributes_dict[-1][key] = val
    
    #Transform the list of dictionaries into a dataframe.
    df2 = pd.DataFrame.from_dict(attributes_dict)
    
    #Concatenate the two dataframes.
    df_complete = pd.concat([df1, df2], axis=1)
    
    
    #Transform the time and note attributes from strings to floats
    df_complete.time = df_complete.time.astype(float)
    try:
        df_complete.note = df_complete.note.astype(int)
    except:
        pass
    
    #Engineer a time elapsed attribute equal to the cumulative sum of time.
    df_complete['time_elapsed'] = df_complete.time.cumsum()
    
    #Filter rows to include only note_on messages 
    #with a velocity greater than zero
    df_filtered = df_complete[df_complete['message_type']=='note_on']
    df_filtered.note = df_filtered.note.astype(int)
    df_filtered = df_filtered.loc[df_filtered['velocity'] != '0']
    
    #Drop empty and unnecessary attributes
    
    df_filtered.drop(['channel', 'value', 'control', 'time'], 
                     axis=1, inplace=True)
    try:
        df_filtered.drop('program', axis=1, inplace=True)
    except:
        pass
    
    # Add a first and last row. This data is used to improve the plot
    add_first_row = []
    add_first_row.insert(0, {'message_type': 'note_on', 'note': 0, 'time': 0, 
                             'velocity': 0, 
                             'time_elapsed':-df_filtered.iloc[-1]['time_elapsed']*0.05})
    df_final = pd.concat([pd.DataFrame(add_first_row), df_filtered], 
                          ignore_index=True)
    last_time_elapsed = df_final.iloc[-1]['time_elapsed']*1.05
    add_last_row = []
    add_last_row.insert(0, {'message_type': 'note_on', 'note': 127, 'time': 0, 
                            'velocity': 0, 'time_elapsed':last_time_elapsed})
    df_final = pd.concat([df_final, pd.DataFrame(add_last_row)], 
                          ignore_index=True)
    
    
    return df_final

def GetOnNotes(midi, time):
    notes = midi.note
    times = midi.time_elapsed
    onNotes = np.zeros(max(notes))
    i = 1
    currentTime = 0
    while currentTime < time:
        currentTime = int(times[i]) 
        if onNotes[int(notes[i])] == 0:
            onNotes[int(notes[i])] = 1
        else:
            onNotes[int(notes[i])] = 0
        i += 1
        
    return onNotes  

def appendNotes(notes, times, onNotes, dt, current_time):
    event_times = times[times>current_time]
    event_notes = notes[times>current_time]
    event_times = event_times[event_times<current_time+dt]
    event_notes = event_notes[0:len(event_times)]
    event_notes = event_notes.array
    event_times = event_times.array
    out = onNotes
    for i in range(len(event_times)):
        if int(event_notes[i]) != 127:
            if out[int(event_notes[i])] == 0:
                out[int(event_notes[i])] = 1
            else:
                out[int(event_notes[i])] = 0

    return onNotes

def noteMat(midi, dt):
    times = midi.time_elapsed
    notes = midi.note
    end_index = int(np.floor(max(times)/dt))
    out = np.zeros((max(notes),end_index))
    current_time = 0
    worker = GetOnNotes(midi, 0)
    for i in range(end_index):
        worker = appendNotes(notes,times,worker,i * dt,current_time)
        out[:,i] = worker
        current_time = i * dt;
        
    return out

def json_to_dict(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
        print("Type: ", type(data))
        
    return data

def split_data(data):
    
    train = []
    test = []
    validation = []
    
    for dic in data:
        
        if dic['split'] == 'train':
            train.append(dic['midi_filename'])
        elif dic['split'] == 'test':
            test.append(dic['midi_filename'])
        elif dic['split'] == 'validation':
            validation.append(dic['midi_filename'])
            
    return train, test, validation


def make_zeropad(train, test, validation):
    
    train_new = []
    test_new = []
    validation_new = []
    
    train_max = 0
    test_max = 0
    validation_max = 0
    
    for tn in train:
        if tn.shape[1] > train_max:
            train_max = copy.deepcopy(tn.shape[1])
    
    for ts in test:
        if ts.shape[1] > test_max:
            test_max = copy.deepcopy(ts.shape[1])
    
    for val in validation:
        if val.shape[1] > validation_max:
            validation_max = copy.deepcopy(val.shape[1])
            
    max_value = np.amax([train_max, test_max, validation_max])
    
    for tn in train:
        diff = max_value - tn.shape[1]
        pad = np.zeros((tn.shape[0],diff))
        train_new.append(np.concatenate((tn,pad),axis=1))
        print("Train")
        
    for ts in test:
        diff = max_value - ts.shape[1]
        pad = np.zeros((ts.shape[0],diff))
        test_new.append(np.concatenate((ts,pad),axis=1))
        print("Test")
    
    for val in validation:
        diff = max_value - val.shape[1]
        pad = np.zeros((val.shape[0],diff))
        validation_new.append(np.concatenate((val,pad),axis=1))
        print("Validation")
        
    return train_new, test_new, validation_new
        

def midi_to_mat(train, test, validation, timestep):
    
    train_new = []
    test_new = []
    validation_new = []
    
    for idx,tn in enumerate(train):
        print("Train ",idx, "/",len(train))
        try:
            df_final = MAESTRO_midi_graph(tn)
        except:
            pass
        if max(df_final.time_elapsed) < 4*10**5:
            train_new.append(noteMat(df_final,timestep))
        else:
            pass
        
    for idx,ts in enumerate(test):
        print("Test ",idx, "/",len(test))
        try:
            df_final = MAESTRO_midi_graph(ts)
        except:
            pass
        if max(df_final.time_elapsed) < 4*10**5:
            test_new.append(noteMat(df_final,timestep))
        else:
            pass
        
    for idx,val in enumerate(validation):
        print("Validation ",idx, "/",len(validation))
        try:
            df_final = MAESTRO_midi_graph(val)
        except:
            pass
        if max(df_final.time_elapsed) < 4*10**5:
            validation_new.append(noteMat(df_final,timestep))
        else:
            pass
        
    return train_new, test_new, validation_new

#%%

os.chdir("C:/Users/emmis/OneDrive/Skrivebord/02456-Deep Learning/Projekt/maestro-v2.0.0")

filename = "maestro-v2.0.0.json"

data = json_to_dict(filename)
train, test, validation = split_data(data)
tn, ts, val = midi_to_mat(train, test, validation, timestep=50)

#%%
tn, ts, val = make_zeropad(tn, ts, val)

with open('split_data.pickle', 'wb') as f:
    pickle.dump([tn, ts, val], f)

