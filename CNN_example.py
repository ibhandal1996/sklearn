from PIL import Image
import numpy as no
import nmist #idk with this wouldn't work
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


#training

x_train = nmist.train_image()
y_train = nmist.train_label()

x_test = nmist.test_image()
y_test = nmist.test_label()

print(x_train.shape) #gives us the shape

x_train = x_train.reshape(-1, 28*28) #this reshapes x_train
x_test - x_test.reshape(-1,28*28) #the -1 get the first value of shape

#neural netwrorks work better with numbers between 0 to 1 instead of large nubers
x_train =(x_train/255)
x_test = (x_test/255)

#creating our model
clf = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(64,64))

#training the model
clf.fit(x_train, y_train)

#finding accuracy
prediction = clf.predict((x_test))
acc = confusion_matrix(y_test, prediction)

print(acc)

def accuracy(cm):
    diagonal = cm.trace()
    elements = cm.sum()
    return diagonal/elements

print(accuracy(acc)) #this gives us our real accuracy