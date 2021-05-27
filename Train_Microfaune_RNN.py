#!/usr/bin/env python
# coding: utf-8

# # Prerequisites
# 
# To run this notebook, `tensorflow` and `microfaune` need to be installed.
# 
# To train or check prediction results, the datasets *freefield* and *warblr* must be unzipped in a folder (its path is specified in the next cell).
# The folder tree should look like this:
# * [data_dir_path]/
#   * ff1010bird_wav/
#     * labels.csv
#     * wav/
#   * warblrb10k_public_wav/
#     * labels.csv
#     * wav/
# 
import IPython.display as ipd

from sklearn.metrics import roc_curve, auc

from microfaune.audio import load_wav
import librosa.display

# In[25]:


train = True

#Change the directory below to your local directory, refer to above folder tree
datasets_dir = "Enter directory path here"


# # imports and function definitions

# In[26]:


#get_ipython().run_line_magic('matplotlib', 'notebook')
import os
import csv
import pickle
from collections import Counter
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.contrib.layers import group_norm
from tensorflow.math import reduce_max

from microfaune.audio import wav2spc

#Data Augmentation added libraries
import matplotlib.pyplot as plt
import librosa
import scipy
from scipy.io import wavfile
import sox
import colorednoise as cn
from multipledispatch import dispatch 
from glob import glob


#DATA AUG BEGIN#####################


#check if GPU is being used
print("Checking if GPU is being used:")
import tensorflow as tf
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print (sess.run(c))
print("End of GPU check")


def make_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
        
#def save(signal, sample_rate, new_dir):
def save(signal, sample_rate, data_path):
    wavfile.write(data_path + "/wav/" + "temp.wav", sample_rate, signal)
        
@dispatch(str, object, object, object) 
def augment_and_save(feature_name, augment_function, signal, sample_rate):
    save(augment_function(signal, sample_rate), sample_rate, feature_name)
        
@dispatch(str, object, object, object, object) 
def augment_and_save(feature_name, augment_function, signal, sample_rate, factor):
    save(augment_function(signal, sample_rate, factor), sample_rate, feature_name)


# Pitch factor should be between 0.9 and 1.1
def augment_pitch(signal, sample_rate, factor):
    #print("Pitch Modulation Factor: ", factor)
    pitch_modulated_signal = librosa.effects.pitch_shift(signal, sample_rate, factor)
    return pitch_modulated_signal

# Noise factor should be between 0.001 and 0.02
def augment_noise(signal, sample_rate, factor):
    #print("Noise Modulation Factor: ", factor)
    noise = np.random.randn(len(signal)) 
    noise_modulated_signal = signal + factor * noise
    noise_modulated_signal = noise_modulated_signal.astype(type(signal[0]))
    return noise_modulated_signal

# Speed factor should be between 0.9 and 1.1
def augment_speed(signal, sample_rate, factor):
    #print("Speed Modulation Factor: ", factor)
    speed_modulated_signal = librosa.effects.time_stretch(signal, factor)
    return speed_modulated_signal
        
# Exponent factor should be 1 for pink noise
def add_colored_noise(signal, sample_rate, factor):
    #print("Gaussian distributed noise with exponent: ", factor)
    noise = cn.powerlaw_psd_gaussian(factor, sample_rate)
    noise = np.tile(noise, int(len(signal) / len(noise)) + 1)
    noise = noise[:len(signal)]
    noise_modulated_signal = signal + noise
    noise_modulated_signal = noise_modulated_signal.astype(type(signal[0]))
    return noise_modulated_signal

def add_gaussian_noise(signal, sample_rate):
    #print("Gaussian noise")
    noise_modulated_signal = signal + np.random.normal(0, 0.1, signal.shape)
    noise_modulated_signal = noise_modulated_signal.astype(type(signal[0]))
    return noise_modulated_signal


#DATA AUG END####################


# Set desired column length of data here (431 for exactly 10 seconds)
max_col_len = 431

# Pad a short clip with a specific value
def pad(data, wanted_shape, pad_value=0):
    row_padding = np.full((wanted_shape[0]-data.shape[0], data.shape[1]), pad_value)
    data = np.vstack((data, row_padding))
    col_padding = np.full((data.shape[0], wanted_shape[1]-data.shape[1]), pad_value)
    data = np.hstack((data, col_padding))
    return data

# Pad a short clip by repeating it until it is the correct length
def repeatclip(data, wanted_length):
    new_data = np.tile(data, (int(wanted_length / data.shape[0]) + 1, 1))
    return new_data[:wanted_length]

# Given a clip of length > wanted_length, returns list of clips that are of wanted length except last element may be equal to the remainder length
def split_clip(clip, wanted_length):
    return np.array(np.split(clip, np.arange(wanted_length,clip.shape[0],wanted_length), axis=0))

def load_dataset(data_path, data_aug={}, use_dump=True):
    mel_dump_file = os.path.join(data_path, "mel_dataset.pkl")
    if os.path.exists(mel_dump_file) and use_dump:
        with open(mel_dump_file, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = compute_feature(data_path, data_aug)
        with open(mel_dump_file, "wb") as f:
            pickle.dump(dataset, f)
    
    X = np.array([dataset["X"][i].transpose() for i in range(len(dataset["X"]))])
    Y = np.array([int(dataset["Y"][i]) for i in range(len(dataset["X"]))])
    uids = [dataset["uids"][i] for i in range(len(dataset["X"]))]
    
    #Going through each clip in the dataset and breaking them down into 10-second chunks
    for i in range(len(dataset["X"])):

        
        # If clip is shorter than 10 seconds, pad it by repeating the clip until it is 10 seconds long
        if X[i].shape[0] < max_col_len:
            X[i] = repeatclip(X[i], max_col_len)
            
        # If clip is longer than 10 seconds, split it into a list of clips that are each 10 seconds long
        elif X[i].shape[0] > max_col_len:
            clips = split_clip(X[i], max_col_len)
            
            # After spliting the clip evenly, remove the last clip in this list if it is less than 10 seconds
            if clips[-1].shape[0] < max_col_len:
                clips = clips[:-1]


            # Add these clips to the dataset
            X = np.concatenate((X,clips))
            
            # Adjusting the labels for the new inflated clip count
            if Y[i] == 0:
                new_y = np.zeros((clips.shape[0], ))
                new_y.fill(0)
                Y = np.concatenate((Y, new_y))
                uids = np.concatenate((uids, [(uids[i] + "-" + str(j+1)) for j in range(len(clips))]))
            else:
                new_y = np.ones((clips.shape[0], ))
                Y = np.concatenate((Y, new_y))
                uids = np.concatenate((uids, [(uids[i] + "-" + str(j+1)) for j in range(len(clips))]))
    

    # Filter out clips over 10 seconds long that have already been split up by the code above 
    inds = [i for i, x in enumerate(X) if x.shape[0] == max_col_len]
    X = np.array([X[i] for i in inds])
    Y = np.array([Y[i] for i in inds])
    uids = [uids[i] for i in inds]

    return X, Y, uids


def compute_feature(data_path, data_aug):
    
    # This dictionary maps an augmenation specified in our input dictionary to the augmentation function itself
    dAFunctions = {
        "pitch" : augment_pitch,
        "noise" : augment_noise,
        "speed" : augment_speed,
        "colored_noise" : add_colored_noise,
        "gaussian_noise" : add_gaussian_noise
    }
    
    
    print(f"Compute features for dataset {os.path.basename(data_path)}")
    labels_file = os.path.join(data_path, "labels.csv")
    if os.path.exists(labels_file):
        with open(labels_file, "r") as f:
            reader = csv.reader(f, delimiter=',')
            labels = {}
            next(reader)  # pass fields names
            for name, y in reader:
                labels[name] = y
    else:
        print("Warning: no label file detected.")
        wav_files = glob(os.path.join(data_path, "wav/*.wav"))
        labels = {os.path.basename(f)[:-4]: None for f in wav_files}
    i = 1
    X = []
    Y = []
    uids = []
    for file_id, y in labels.items():
        print(f"{i:04d}/{len(labels)}: {file_id:20s}", end="\r")
        spc = wav2spc(os.path.join(data_path, "wav", f"{file_id}.wav"))
        X.append(spc)
        Y.append(y)
        uids.append(file_id)
        #____________Data Aug Step___________
        
       
        filename = f"{file_id}.wav"
        sr = 44100
        signal, sample_rate = librosa.load(os.path.join(data_path, "wav", f"{file_id}.wav"), sr)
        
        #iterate through our dictionary and add any augmentations
        #key is the augmentation specified in the dictionary such as "pitch"
        for key in data_aug:
            # gaussian_noise specifically has a different formatting (one less parameter)
            if key == "gaussian_noise":
                augment_and_save(data_path, add_gaussian_noise, signal, sample_rate)
                #The augmentation is written to a file, then deleted post-pickling
                spc = wav2spc(os.path.join(data_path, "wav", "temp.wav"))
                X.append(spc)
                Y.append(y)
                uids.append(file_id)
                os.remove(data_path + "/wav/temp.wav")
                
            else:
                #Second loop for when we want more than one augmentation of one type, such as two pitch alterations
                # val is the elements of the array associated with an augmentation type
                for val in data_aug[key]:
                    augment_and_save(data_path, dAFunctions[key], signal, sample_rate, val)
                    #The augmentation is written to a file, then deleted post-pickling
                    spc = wav2spc(os.path.join(data_path, "wav", "temp.wav"))
                    X.append(spc)
                    Y.append(y)
                    uids.append(file_id)
                    os.remove(data_path + "/wav/temp.wav")
        
        #_____________________________________
        
        i += 1
    return {"uids": uids, "X": X, "Y": Y}


# In[33]:


def split_dataset(X, Y, random_state=0):
    split_generator = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    ind_train, ind_test = next(split_generator.split(X, Y))
    X_train, X_test = X[ind_train, :, :], X[ind_test, :, :]
    Y_train, Y_test = Y[ind_train], Y[ind_test]
    return ind_train, ind_test


# # Load dataset

# In[34]:


#Note: different augmentations take different amounts of time
#e.g. Altering pitch or speed has a higher complexity than something like adding gaussian noise.
'''
Dictionary formatting:
Options are "pitch", "speed", "noise, "gaussian_noise", and "colored_noise". These augmentations
should be the keys of the dictionary (omitting a key just means not including it in used augmentations) 
with the correlating values being an array. Each element in that array is one augmentation of that type
specified by the value. Users can use as many augmentations of each type as desired except gaussian noise. 
Gaussian noise is constant so only one is added. To ommit gaussian noise simply remove it from the dictionary

For a quick start, copy and paste the following and add augmentations to the arrays of each type:

data_aug = {
    "pitch" : [],
    "speed" : [],
    "noise" : []
    "gaussian_noise" : [], #Adding to this array serves no purpose
    "colored_noise" : []
           }

'''
data_aug = {
    "pitch" : [0.9, 0.95, 1.05, 1.1],
    "noise" : [0.005]
           }


X0, Y0, uids0 = load_dataset(os.path.join(datasets_dir, "ff1010bird_wav"), data_aug, False)
X1, Y1, uids1 = load_dataset(os.path.join(datasets_dir, "warblrb10k_public_wav"), data_aug, False)
X = np.concatenate([X0, X1]).astype(np.float32)/255
Y = np.concatenate([Y0, Y1])
uids = np.concatenate([uids0, uids1])


del X0, X1, Y0, Y1


Counter(Y)



ind_train, ind_test = split_dataset(X, Y)

X_train, X_test = X[ind_train, :, :, np.newaxis], X[ind_test, :, :, np.newaxis]
Y_train, Y_test = Y[ind_train], Y[ind_test]
uids_train, uids_test = uids[ind_train], uids[ind_test]
del X, Y



print("Training set: ", Counter(Y_train))
print("Test set: ", Counter(Y_test))



n_filter = 64
conv_reg = keras.regularizers.l2(1e-3)
norm = "batch"

spec = layers.Input(shape=[*X_train.shape[1:]], dtype=np.float32)

x = layers.Conv2D(n_filter, (3, 3), padding="same", kernel_regularizer=conv_reg, activation=None)(spec)
if norm == "batch":
    x = layers.BatchNormalization(momentum=0.95)(x)
elif norm == "group":
    x = group_norm(x, groups=4, channels_axis=-1, reduction_axes=[-3, -2])
x = layers.ReLU()(x)
x = layers.Conv2D(n_filter, (3, 3), padding="same", kernel_regularizer=conv_reg, activation=None)(x)
if norm == "batch":
    x = layers.BatchNormalization(momentum=0.95)(x)
elif norm == "group":
    x = group_norm(x, groups=4, channels_axis=-1, reduction_axes=[-3, -2])
x = layers.ReLU()(x)
x = layers.MaxPool2D((1, 2))(x)

x = layers.Conv2D(n_filter, (3, 3), padding="same", kernel_regularizer=conv_reg, activation=None)(x)
if norm == "batch":
    x = layers.BatchNormalization(momentum=0.95)(x)
elif norm == "group":
    x = group_norm(x, groups=4, channels_axis=-1, reduction_axes=[-3, -2])
x = layers.ReLU()(x)
x = layers.Conv2D(n_filter, (3, 3), padding="same", kernel_regularizer=conv_reg, activation=None)(x)
if norm == "batch":
    x = layers.BatchNormalization(momentum=0.95)(x)
elif norm == "group":
    x = group_norm(x, groups=4, channels_axis=-1, reduction_axes=[-3, -2])
x = layers.ReLU()(x)
x = layers.MaxPool2D((1, 2))(x)

x = layers.Conv2D(n_filter, (3, 3), padding="same", kernel_regularizer=conv_reg, activation=None)(x)
if norm == "batch":
    x = layers.BatchNormalization(momentum=0.95)(x)
elif norm == "group":
    x = group_norm(x, groups=4, channels_axis=-1, reduction_axes=[-3, -2])
x = layers.ReLU()(x)
x = layers.Conv2D(n_filter, (3, 3), padding="same", kernel_regularizer=conv_reg, activation=None)(x)
if norm == "batch":
    x = layers.BatchNormalization(momentum=0.95)(x)
elif norm == "group":
    x = group_norm(x, groups=4, channels_axis=-1, reduction_axes=[-3, -2])
x = layers.ReLU()(x)
x = layers.MaxPool2D((1, 2))(x)

x = reduce_max(x, axis=-2)

x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)

x = layers.TimeDistributed(layers.Dense(64, activation="sigmoid"))(x)
local_pred = layers.TimeDistributed(layers.Dense(1, activation="sigmoid"))(x)
pred = reduce_max(local_pred, axis=-2)

model = keras.Model(inputs=spec, outputs=pred)
model.summary()

# for predictions only
dual_model = keras.Model(inputs=spec, outputs=[pred, local_pred])



class DataGenerator(keras.utils.Sequence):
    def __init__(self, X, Y, batch_size=32):
        self.X = X
        self.Y = Y
        self.n = len(Y)
        self.batch_size = batch_size
        self.shuffle()
        
    def __len__(self):
        return int(np.floor(self.n)/self.batch_size)
    
    def __getitem__(self, index):
        batch_inds = self.inds[self.batch_size*index:self.batch_size*(index+1)]
        self.counter += self.batch_size
        if self.counter >= self.n:
            self.shuffle()
        return self.X[batch_inds, ...], self.Y[batch_inds]
    
    def shuffle(self):
        self.inds = np.random.permutation(self.n)
        self.counter = 0


if train:
    optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy', keras.metrics.FalseNegatives()])

    alpha = 0.5
    batch_size = 32

    data_generator = DataGenerator(X_train, Y_train, batch_size)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=1e-5)

    history = model.fit_generator(data_generator, steps_per_epoch=100, epochs=100,
                                  validation_data=(X_test, Y_test),
                                  class_weight={0: alpha, 1: 1-alpha},
                                  callbacks=[reduce_lr], verbose=1)

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save_weights(f"model_weights-{date_str}.h5")
