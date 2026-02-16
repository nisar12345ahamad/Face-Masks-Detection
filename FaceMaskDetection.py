import os
import tensorflow as tf
print(tf.__version__)
import cv2


categories = ['with_mask' , 'without_mask']
data = []
for category in categories :
    path =os.path.join(category)

    label = categories.index(category)
    for file in os.listdir(path):

        img_path = os.path.join(path,file)
        img = cv2.imread(img_path)
        img = cv2.resize(img,(224 , 224))
        print(img.shape)

        data.append([img,label])
        print(len(data))

import random
print(random.shuffle(data))

X = []
Y = []

for features,label in data:
    X.append(features)
    Y.append(label)

print(len(X))
print(len(Y))

import numpy as np
X = np.array(X)
Y = np.array(Y)

print(X.shape)
print(Y.shape)


from sklearn.model_selection import train_test_split
X_train ,x_test, Y_train, y_test = train_test_split(X ,Y ,test_size=0.2)
print(X_train.shape)
print(Y_train.shape)

from keras.applications import VGG16
from keras.models import Sequential

# Load Pretrained VGG16
vgg = VGG16()

print(vgg.summary())

# Create new Sequential model
model = Sequential()

# Remove last layer
for layer in vgg.layers[:-1]:
    model.add(layer)

# Print new model summary
model.summary()

for layer in model.layers:
    layer.trainable = False

model.summary()

from keras.layers import Dense
model.add(Dense(1,activation='sigmoid'))
model.summary()

model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,Y_train , epochs=5, validation_data=(x_test,y_test))

import cv2
# woh function jo dikhaye ki mouth par mask hain ya nahi ?

def draw_label(img,text,pos,bg_color):
    # Draw label text size :
    text_size = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,cv2.FILLED)
    end_x = pos[0] + text_size[0][0] +2
    end_y = pos[1] + text_size[0][0] -2

    cv2.rectangle(img,pos,(end_x,end_y),bg_color,cv2.FILLED)
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(0,0,0),1,cv2.LINE_AA)

def detect_fact_mask(img):

    y_pred = model.predict(img.reshape(1,224,224,3))
    return y_pred[0][0]


cap = cv2.VideoCapture(0)

while True:
    ret ,frame =cap.read()

     # call the  detection method :
    img = cv2.resize(frame,(224 ,224))

    y_pred = detect_fact_mask(img)

    if y_pred == 0 :
        draw_label(frame ,"Mask" , (30,30),(0,0,255))
    else:
        draw_label(frame ,"No Mask", (30,30),(0,255,0))

    draw_label(frame,"face mask detection" ,(100,500),(255,0,0))
    cv2.imshow("window",frame)
    

    if cv2.waitKey(1) & 0xFF ==ord('x'):
     break

cap.release()
cv2.destroyAllWindows()

sample1 =cv2.imread('sample/mask.jpeg')
sample1 = cv2.resize(sample1,(224 ,224))

detect_fact_mask(sample1)


# FaceMask Detector
# import cv2

# while True:

#     ret , frame = cap.read()

#     coods = detect_face(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
#     for x,y,w,h in coods:

#         cv2.rectangleIntersectionArea(frame , (x,y),(x+w, y+h),(255,0,0),3)

#     cv2.imshow("window",frame)

#     if cv2.waitKey(1) & 0xFF == ord('x'):
#         break
#     cv2.destroyAllWindows()

# haar =cv2.CascadeClassifier('C:\train\haarcascade_frontalface_default.xml')

# def detect_face(img):

#     coods = haar.detectMultiScale(img)
#     return coods