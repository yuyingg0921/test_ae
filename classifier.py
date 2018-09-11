import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop
import numpy as np
import sys
import time
from keras import backend as K
start = time.time()


K.tensorflow_backend._get_available_gpus()

'''
# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], -1) / 255.   # normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255.      # normalize
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)
'''

labels = []
with open("data_20news/test_labels.txt") as f_y:
	for line in f_y:
		line = line.split(",")
		for article in line:
			article = article.strip().strip("}").split(" ")
			#print (article[1].strip("\""))
			labels.append(article[1].strip("\""))
	
y_train = np.array(labels[:5272])
y_test = np.array(labels[5272:])
y_train = np_utils.to_categorical(y_train, num_classes=20)
y_test = np_utils.to_categorical(y_test, num_classes=20)

doc_vec = []
tmp = []

with open("output_20news/output_doc_vec") as f_x:
	for line in f_x:
		line = line.split("]")
		for article in line:
			article = article.split("[")  # get ["vectors inside"]
			try:
				tmp2 = []
				tmp = article[1].split(",")
				for each_tmp in tmp:
					tmp2.append(each_tmp.strip(" "))
				doc_vec.append(tmp2)
			except:
				pass

X_train = np.array(doc_vec[:5272])
X_test = np.array(doc_vec[5272:])


#print ("===================")
print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)

# Another way to build your neural net
'''
model = Sequential([
    Dense(32, input_dim=128),
    Activation('relu'),
    Dense(20),
    Activation('softmax'),
])
'''
model = Sequential()
model.add(Dense(20, input_dim=128, activation=None))
model.add(Dense(20, activation='softmax'))
# Another way to define your optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# We add metrics to get more results you want to see
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, epochs=200, batch_size=8)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier

y_prob = model.predict(X_test) 
y_classes = y_prob.argmax(axis=-1)
y_classes = np_utils.to_categorical(y_classes, num_classes=20)

loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)

end = time.time()
print ('time:', end-start)