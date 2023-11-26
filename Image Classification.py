#!/usr/bin/env python
# coding: utf-8

# In[2]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator#generate label
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
img=image.load_img("C:\\Users\\user\\OneDrive\\Desktop\\computer-vision\\basedata\\training\\dog\\d1.jpg")
plt.imshow(img)
cv2.imread("C:\\Users\\user\\OneDrive\\Desktop\\computer-vision\\basedata\\training\\dog\\d1.jpg").shape


# In[12]:


train=ImageDataGenerator(rescale=1/255)
validate=ImageDataGenerator(rescale=1/255)
train_dataset=train.flow_from_directory("C:\\Users\\user\\OneDrive\\Desktop\\computer-vision\\basedata\\training",
                                       target_size=(200,200),
                                       batch_size=3,
                                       class_mode="binary")
validation_dataset=validate.flow_from_directory("C:\\Users\\user\\OneDrive\\Desktop\\computer-vision\\basedata\\validation",
                                       target_size=(200,200),
                                       batch_size=3,
                                       class_mode="binary")


# In[13]:


train_dataset.class_indices
train_dataset.classes


# In[14]:


model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation="relu",input_shape=(200,200,3)),
                                                        tf.keras.layers.MaxPool2D(2,2),
                                                          
                                tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
                                                        tf.keras.layers.MaxPool2D(2,2),
                                                        
                                tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
                                                        tf.keras.layers.MaxPool2D(2,2),
                                tf.keras.layers.Flatten(),
                                  
                                tf.keras.layers.Dense(512,activation="relu"),
                                  
                                                         
                                tf.keras.layers.Dense(1,activation="sigmoid")#binary
                                 ])


# In[15]:


model.compile(loss="binary_crossentropy",
             optimizer=RMSprop(learning_rate=0.001),
             metrics=["accuracy"])


# In[19]:


from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_fit=model.fit(train_dataset,
    steps_per_epoch=3,  
    epochs=50,
    validation_data=validation_dataset,
    callbacks=[early_stopping])


# In[22]:


dir_path='C:\\Users\\user\\OneDrive\\Desktop\\computer-vision\\basedata\\testing'
for i in os.listdir(dir_path):
    img=image.load_img(dir_path+'//'+i,target_size=(200,200))
    plt.imshow(img)
    plt.show()
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    images=np.vstack([x])
    val=model.predict(images)
    if val==0:
        print("cat")
    elif(val==1):
        print("dog")


# In[ ]:




