from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

( x_train, y_train),(x_test,y_test) = mnist.load_data()


y_cat_test =  to_categorical(y_test)
y_cat_train =  to_categorical(y_train)

# normilizing the data
x_tarin = x_train / x_train.max()
x_test = x_test / x_test.max()

# reshaping the training and testing dtat to add colour channel

x_test = x_test.reshape(10000,28,28,1)
x_train = x_train.reshape(60000,28,28,1)

# building the model

from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten

# creating model object
model = Sequential()

# adding the layers

model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(28,28,1),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.summary()

# training the data

model.fit(x_train,y_cat_train,epochs=3)
model.metrics_names

from sklearn.metrics import classification_report

predictions = model.predict_classes(x_test)
print(classification_report(y_test,predictions))

model.save('mnist_model.h5')

