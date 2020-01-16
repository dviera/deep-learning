import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
import os

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, LSTM, Input, TimeDistributed, Flatten
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential


def data_loader(csv):
    dataset = []
    data = pd.read_csv(csv)
    human = data['human'].values
    machine = data['machine'].values
    
    for i in range(len(human)):
        dataset.append((human[i], machine[i]))
        
    return dataset, human, machine

def convert_to_dic(human, machine):
    human_vocab = set()
    machine_vocab = set()
    n_samples = len(human)
    
    for i in tqdm(range(n_samples)):
        human_vocab.update(tuple(human[i]))
        machine_vocab.update(tuple(machine[i]))
    
    human_dic = {v:k for k, v in enumerate(sorted(human_vocab) + ['<pad>', '<unk>'])}
    machine_dic = {v:k for k, v in enumerate(sorted(machine_vocab))}
    inv_machine_dic = {k:v for k, v in enumerate(sorted(machine_vocab))}
    
    return human_dic, machine_dic, inv_machine_dic


#####################################
# LOAD DATA                         #
#####################################
dataset, human, machine = data_loader('data/human-machine.csv')
human_dic, machine_dic, inv_machine_dic = convert_to_dic(human, machine)

#####################################
# AUXILIARY FUNCTION PREPROCESS     #
#####################################
def padding(string, Tx):
            
    if len(string) > Tx:
        return list(string[:Tx])
    else:
        return list(string) + ['<pad>'] * (Tx - len(string))
        

def char_to_int(string, human_dic, machine_dic, Tx, Ty, output = False):
    if not output:
        return [human_dic[i] if i in human_dic else human_dic['<unk>'] for i in padding(string, Tx)]
    else:
        return [machine_dic[i] for i in list(string)]

def to_categorical(x, num_classes):
    #return np.eye(num_classes, dtype='uint8')[x]
    return np.eye(num_classes)[x]
    
def preprocess(dataset, human_dic, machine_dic, Tx, Ty, unseen = False):
    
    if not unseen:
        X, y = zip(*dataset)
        X = np.array([char_to_int(i, human_dic, machine_dic, Tx, Ty, output = False) for i in X])
        y = np.array([char_to_int(i, human_dic, machine_dic, Tx, Ty, output = True) for i in y])
        Xoh = np.array(list(map(lambda x: to_categorical(x, len(human_dic)), X)))
        Yoh = np.array(list(map(lambda y: to_categorical(y, len(machine_dic)), y)))
        return X, y, Xoh, Yoh
    else:
        X = dataset
        X = np.array([char_to_int(i, human_dic, machine_dic, Tx, Ty, output = False) for i in X])
        return X

Tx = 30 # input sequence length
Ty = 10 # output sequence length
X, y, Xoh, yoh = preprocess(dataset, human_dic, machine_dic, Tx, Ty)

print("Shape of X: ", X.shape)
print("Shape of y: ", y.shape)
print("Shape of Xoh: ", Xoh.shape)
print("Shape of yoh: ", yoh.shape)


#------------------------
# split train-val-test  #
#------------------------

X_train = Xoh[:8000,:,:]
y_train = yoh[:8000,:]

X_val = Xoh[8000:9000,:,:]
y_val = yoh[8000:9000,:]

X_test = Xoh[9000:,:,:]
y_test = yoh[9000:,:]


#####################################
# KERAS MODEL                       #
#####################################

BATCH_SIZE = 64
NUM_LAYERS = 1
HIDDEN_UNITS_LSTM1 = 128
HIDDEN_UNITS_LSTM2 = 64
OUTPUT_SIZE = 11

inputs = Input(shape=(30, 37))
out = LSTM(units = HIDDEN_UNITS_LSTM1)(inputs)
out = RepeatVector(10)(out)
out = LSTM(units = HIDDEN_UNITS_LSTM2, return_sequences=True)(out)
out = Dense(OUTPUT_SIZE, activation="softmax")(out)


model = Model(inputs, out)
model.summary()

model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])


# lauch tensorboard cmd -> tensorboard --logdir logs/fit
tboard_log_dir = os.path.join("logs",
                              "fit",
                              datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),)

tb = TensorBoard(log_dir=tboard_log_dir)


history = model.fit(X_train,
          y_train,
          validation_data = (X_val, y_val),
          epochs=50,
          batch_size=BATCH_SIZE,
          callbacks=[tb])


model.predict(X_test[0,:,:].reshape(1,30,37)).argmax(axis=1)

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#------------------------
# predict               #
#------------------------

X_test.shape
def predict_keras(model, X_test, y_test, inv_machine_dic, num_to_print = 10):
    
    n_samples, seq_len, feature_len = X_test.shape
    if num_to_print > n_samples:
        return print("Number to print {} is greater than number of samples {}".format(num_to_print, n_samples))
    
    test_data = X_test[:num_to_print,:,:]
    prediction = model.predict(test_data).argmax(axis=2).tolist()
    
    actual_data = y_test[:num_to_print,:,:]
    
    for idx, pred in enumerate(prediction):
        predicted = [inv_machine_dic[i] for i in pred]
        actual = [inv_machine_dic[i] for i in actual_data[idx,:,:].argmax(axis=1).tolist()]
        print("-" * 30)
        print("Actual date: \t", "".join(actual))
        print("Predicted date: ", "".join(predicted))
      
        
predict_keras(model, X_test, y_test, inv_machine_dic, num_to_print=10)
