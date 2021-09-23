from __future__ import print_function
import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten, Conv1D, GlobalMaxPooling1D, Reshape, GlobalAveragePooling1D
from keras.layers.convolutional import MaxPooling1D
from pandas import read_csv
#from load_embedding import load_embeddings
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from sklearn.metrics import f1_score

print('Loading data...')

path_train_x = "X_train"
path_test_x = "X_test"
path_dev_x = "X_dev"
path_train_y = "Y_train"
path_test_y = "Y_test"
path_dev_y = "Y_dev"

def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values
	
x_train = load_file(path_train_x)
x_test = load_file(path_test_x)
x_dev = load_file(path_dev_x)
y_train = np.loadtxt(fname = path_train_y)
y_test = np.loadtxt(fname = path_test_y)
y_dev = load_file(path_dev_y)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print(len(x_dev), 'val sequences')

input_dim, output_dim = 300, 1

x_train = x_train.astype('float32')
y_train = y_train.astype('float32')

y_train_hot = np_utils.to_categorical(y_train)
print('New y_train shape: ', y_train_hot.shape) 
y_test_hot = np_utils.to_categorical(y_test)
print('New y_test shape: ', y_test_hot.shape) 
y_dev_hot = np_utils.to_categorical(y_dev)
print('New y_dev shape: ', y_dev_hot.shape) 

print('Build model...')

model_m = Sequential()

model_m.add(Reshape((300,1), input_shape=(300,)))
model_m.add(Flatten())
model_m.add(Dense(256,activation='relu'))
model_m.add(Dense(128, activation='relu'))
model_m.add(Dense(102, activation='softmax'))

model_m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model_m.summary())

model_m.fit(x_train, y_train_hot, 
          validation_data=(x_dev, y_dev_hot),
          batch_size =32,
          epochs=100)

y_pred = model_m.predict_classes(x_test)

loss = model_m.evaluate(x_test, y_test_hot)[0]
accuracy = model_m.evaluate(x_test, y_test_hot)[1]
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_micro = f1_score(y_test, y_pred, average='micro')
f1_normal = f1_score(y_test, y_pred, average=None)

print('loss', loss)
print('accuracy', accuracy)
print('f1_macro', f1_macro)
print('f1_micro', f1_micro)
print('f1_normal', f1_normal)

model_m.save("FIGER_cat_w2v_FC_model.h5")
print("Saved model to disk")

f = open("FIGER_cat_w2v_FC_model.txt", "a")
for i in range(len(x_test)):
    f.write(str(y_test[i]))
    f.write("\t")
    f.write(str(y_pred[i]))
    f.write("\n")
f.close()
