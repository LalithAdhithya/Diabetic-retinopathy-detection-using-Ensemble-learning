#!/usr/bin/env python
# coding: utf-8

# ### This project aims to build an ensemble learning model which interfaces 3 CNN models namely DenseNet-121,ResNet-50 and Inception-V3 that is used to detect Diabetic retinopathy

# In[ ]:


#densenet
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# Set the path to your dataset
train_dir = 'Downloads/colored_images1/train'
test_dir = 'Downloads/colored_images1/test'

# Set the image dimensions and number of classes
img_width, img_height = 224, 224
num_classes = 5

# Data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load and augment training data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=16,
    class_mode='categorical'
)

# Load and augment testing data
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=16,
    class_mode='categorical'
)

# Load the pre-trained DenseNet121 model
base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))

# Add custom classification layers on top of the pre-trained model
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the model
model1 = Model(inputs=base_model.input, outputs=predictions)

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model1.fit(train_data, epochs=5, validation_data=test_data)

# Evaluate the model on the testing data
_, accuracy = model1.evaluate(test_data)
print('Accuracy:', accuracy)

#save
# Save the trained model
model1.save('diabetic_retinopathy_classification_model1.h5')



# In[ ]:


import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the path to your dataset
train_dir = 'Downloads/colored_images1/train'
test_dir = 'Downloads/colored_images1/test'

# Set the image dimensions and number of classes
img_width, img_height = 224, 224
num_classes = 5

# Data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load and augment training data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=16,
    class_mode='categorical'
)

# Load and augment testing data
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=16,
    class_mode='categorical'
)

# Load the pre-trained Inception-v3 model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add a new fully connected layer for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the final model
model2 = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model2.fit(train_data, epochs=5, validation_data=test_data)

# Save the trained model
model2.save('diabetic_retinopathy_classification_model2.h5')


# In[ ]:


#resnet50 add code


# In[ ]:


import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the input shape of your images
input_shape = (224, 224, 3)

# Create the base ResNet-50 model
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)

# Add a global average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully-connected layer
x = Dense(256, activation='relu')(x)

# Add the final classification layer
predictions = Dense(5, activation='softmax')(x)  # Replace num_classes with the number of classes in your dataset

# Create the model
model3 = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Set the directories for your training and testing datasets
train_dir = 'Downloads/colored_images1/train'
test_dir = 'Downloads/colored_images1/test'

# Set the batch size and number of training steps
batch_size = 16
num_train_steps = int(2930/ batch_size)

# Set the number of testing steps
num_test_steps = int(732 / batch_size)

# Load the training and testing datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical')

# Train the model
model3.fit(
    train_generator,
    steps_per_epoch=num_train_steps,
    epochs=5,  # Adjust the number of epochs as needed
    validation_data=test_generator,
    validation_steps=num_test_steps)

# Save the trained model
model3.save('diabetic_retinopathy_classification_model3.h5')


# In[ ]:


import tensorflow as tf
from tensorflow.keras.applications import ResNet50, InceptionV3, DenseNet121
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, concatenate, Dense
from tensorflow.keras.models import Model

# Load the pre-trained models
model_1 = tf.keras.models.load_model('diabetic_retinopathy_classification_model1.h5')
model_2 = tf.keras.models.load_model('diabetic_retinopathy_classification_model2.h5')
model_3 = tf.keras.models.load_model('diabetic_retinopathy_classification_model3.h5')
# Define the input shape for the models
input_shape = (240, 240, 3)
model_input = Input(shape=input_shape)

# Get the model outputs
model_1_output = model_1(model_input)
model_2_output = model_2(model_input)
model_3_output = model_3(model_input)
num_classes=5

# Concatenate the outputs
concatenated_outputs = concatenate([model_1_output, model_2_output,nodel_3_output])

# Add a classification layer on top
output = Dense(num_classes, activation='softmax')(concatenated_outputs)

# Create the ensemble model
ensemble_model = Model(inputs=model_input, outputs=output)

# Compile the ensemble model
ensemble_model.compile(optimizer='adam',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

# Train the ensemble model
ensemble_model.fit(train_data, validation_data=test_data,epochs=10, batch_size=16)

# Evaluate the ensemble model
test_loss, test_acc = ensemble_model.evaluate(test_data)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_acc:.4f}')


# In[ ]:




