#!/usr/bin/env python

import numpy as np
from keras.models import load_model,model_from_json
def getsamples(ecg, window, stride):
    length = ecg.shape[1]
    ecg = ecg[:,:int(length//4*4)].reshape(ecg.shape[0],-1,4)
    ecg = np.average(ecg,axis=2)
    if len(ecg[0])<window:
        new = np.concatenate((ecg,np.zeros((12,window-len(ecg[0])))))
    else:
        i = 0
        new = []
        while i+window<len(ecg[0]):
            new.append(ecg[:,i:i+window])
            i += stride
        new.append(ecg[:,-window:])
    samples = np.array(new)
    return samples.reshape(-1,12,window,1)
from get_12ECG_features import get_12ECG_features

def run_12ECG_classifier(data,header_data,classes,model):

    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    ecg = data/1000
    ecg0 = ecg.copy()

    samples0 = getsamples(ecg0, 625, 312)
    samples = getsamples(ecg, 625, 156)

    l = samples0.shape[0]
    if l<30:
        z = np.zeros((30-l,12,625,1))
        samples0 = np.vstack((z,samples0))
    else:
        samples0 = samples0[:30,:]
    
    l = samples.shape[0]
    if l<55:
        z = np.zeros((55-l,12,625,1))
        samples = np.vstack((z,samples))
    else:
        samples = samples[:55,:]

    result_first = model[0].predict([samples0,samples])
    result = model[1].predict(np.concatenate((result_first[0].reshape(1, 30*9), 
                  result_first[1].reshape(1, 30*4)), axis=1))[0]

    fixed = False
    for ii in range(num_classes):
        if result[ii] > 0.5:
            current_label[ii] = 1;
            fixed = True
    if fixed == False:
        current_label[0] = 1;
    
    current_score = result;
    return current_label, current_score

def load_12ECG_model():
    # load the model from disk 
    model_first = model_from_json(open('model_first.json').read())
    model_first.load_weights('model_first.h5')
    model_second = model_from_json(open('model_second.json').read())
    model_second.load_weights('model_second.h5')

    loaded_model = {0: model_first, 1: model_second}
    return loaded_model
