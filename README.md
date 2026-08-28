# EX. NO. 4(a) – MACHINE LEARNING MODEL: LINEAR REGRESSIO

To build and train a **Convolutional Neural Network (CNN)** using TensorFlow for image classification using the **CIFAR-10 dataset**

# AIM, OBJECTIVES AND REQUIREMENTS

## OBJECTIVES

The objectives of this experiment are:

1. To understand the concept of Convolutional Neural Networks.
2. To load the CIFAR-10 image dataset.
3. To preprocess and normalize the image data.
4. To convert class labels into one-hot encoded format.
5. To visualize sample images from the dataset.
6. To design a CNN architecture using convolution and pooling layers.
7. To train the CNN model using the training dataset.
8. To evaluate the model using accuracy and loss.
9. To predict the classes of new test images.

## SOFTWARE REQUIREMENTS

* Python
* Google Colab / Jupyter Notebook
* TensorFlow
* Keras
* NumPy
* Matplotlib

## HARDWARE REQUIREMENTS

* Computer or Laptop
* Minimum 4 GB RAM
* Internet connection for downloading the dataset

---

# INTRODUCTION

## CONVOLUTIONAL NEURAL NETWORK

A **Convolutional Neural Network (CNN)** is a type of Deep Learning model mainly used for image processing and image classification tasks.

Unlike traditional machine learning algorithms, CNNs can automatically learn important features directly from images. These features may include:

* Edges
* Shapes
* Textures
* Patterns
* Object parts

A CNN processes an image through multiple layers. Each layer extracts more complex information from the image.

For example:

**Input Image → Convolution → ReLU → Pooling → Convolution → Fully Connected Layer → Output Class**

CNNs are widely used in:

* Image classification
* Face recognition
* Object detection
* Medical image analysis
* Self-driving cars
* Security systems
* Handwriting recognition

The main advantage of CNN is that it automatically extracts useful features from images without requiring manual feature engineering.

---

# CIFAR-10 DATASET

## ABOUT THE DATASET

The CIFAR-10 dataset is a commonly used dataset for image classification experiments.

It contains a total of **60,000 color images**.

### Dataset Distribution

| Dataset      | Number of Images |
| ------------ | ---------------: |
| Training Set |           50,000 |
| Test Set     |           10,000 |
| Total        |           60,000 |

Each image has a size of:

**32 × 32 pixels**

Each image contains three color channels:

* Red
* Green
* Blue

Therefore, the input image shape is:

```text
(32, 32, 3)
```

## CIFAR-10 CLASSES

The dataset contains the following ten classes:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

The objective of the CNN model is to correctly classify an input image into one of these ten categories.

---

# IMPORT LIBRARIES AND LOAD DATASET

##IMPORT REQUIRED LIBRARIES

TensorFlow and Keras are used for building the CNN model. Matplotlib is used for visualization, while NumPy is used for numerical operations.

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
```

### DESCRIPTION OF LIBRARIES

* **TensorFlow** – Used for deep learning and neural networks.
* **Keras** – Provides simple functions for building CNN models.
* **Datasets** – Used to load the CIFAR-10 dataset.
* **Layers** – Used to create CNN layers.
* **Models** – Used to create the neural network architecture.
* **NumPy** – Used for numerical operations.
* **Matplotlib** – Used for displaying images and graphs.
* **to_categorical()** – Used for one-hot encoding.

## LOAD CIFAR-10 DATASET

```python
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
```

Here:

* `X_train` contains training images.
* `y_train` contains training labels.
* `X_test` contains test images.
* `y_test` contains test labels.

The training dataset is used to teach the CNN, while the test dataset is used to evaluate its performance.

---

# PAGE 5 – DATA PREPROCESSING AND VISUALIZATION

## STEP 3: NORMALIZE IMAGE DATA

The original pixel values of an image range from **0 to 255**.

For better training performance, the values are converted into the range **0 to 1**.

```python
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
```

Normalization improves:

* Training stability
* Convergence speed
* Numerical efficiency
* Model performance

## ONE-HOT ENCODING

The labels are converted into a vector of length 10.

```python
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```

For example, if the image belongs to the **Airplane** class:

```text
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

If it belongs to the **Cat** class:

```text
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

## STEP 4: VISUALIZE SAMPLE IMAGES

```python
class_names = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

plt.figure(figsize=(10, 10))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i])
    plt.xlabel(class_names[np.argmax(y_train[i])])

plt.show()
```

This code displays a **4 × 4 grid containing 16 sample images** from the CIFAR-10 dataset.

---

# PAGE 6 – BUILDING THE CNN MODEL

## STEP 5: CREATE CNN ARCHITECTURE

The CNN model consists of multiple convolutional layers, pooling layers, dropout layers, and fully connected layers.

```python
model = models.Sequential()

model.add(layers.Conv2D(
    32, (3, 3),
    activation='relu',
    padding='same',
    input_shape=(32, 32, 3)
))

model.add(layers.Conv2D(
    32, (3, 3),
    activation='relu',
    padding='same'
))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
```

The first convolutional layers extract basic features such as edges and patterns.

### Second CNN Block

```python
model.add(layers.Conv2D(
    64, (3, 3),
    activation='relu',
    padding='same'
))

model.add(layers.Conv2D(
    64, (3, 3),
    activation='relu',
    padding='same'
))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
```

The second block extracts more detailed features from the image.

### Third CNN Block

```python
model.add(layers.Conv2D(
    128, (3, 3),
    activation='relu',
    padding='same'
))

model.add(layers.Conv2D(
    128, (3, 3),
    activation='relu',
    padding='same'
))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
```

The deeper layers learn complex features related to objects and shapes.

---

# PAGE 7 – FULLY CONNECTED LAYERS AND MODEL TRAINING

## FLATTEN AND DENSE LAYERS

The output from the convolutional layers is converted into a one-dimensional vector using the Flatten layer.

```python
model.add(layers.Flatten())

model.add(layers.Dense(
    512,
    activation='relu'
))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(
    10,
    activation='softmax'
))
```

### FUNCTIONS OF EACH LAYER

**Flatten Layer:**
Converts multidimensional feature maps into a single vector.

**Dense Layer:**
Learns relationships between extracted features.

**Dropout Layer:**
Randomly disables some neurons during training to reduce overfitting.

**Softmax Layer:**
Produces probability values for all ten classes.

## STEP 6: COMPILE THE MODEL

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

### PARAMETERS USED

* **Optimizer:** Adam
* **Loss Function:** Categorical Cross-Entropy
* **Performance Metric:** Accuracy

## STEP 7: TRAIN THE MODEL

```python
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.2
)
```

The model is trained for **30 epochs** with a batch size of **64**.

---

# PAGE 8 – MODEL PERFORMANCE AND GRAPH ANALYSIS

## STEP 8: PLOT TRAINING HISTORY

The training history contains accuracy and loss values for every epoch.

```python
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(
    history.history['accuracy'],
    label='Train Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
```

## MODEL LOSS GRAPH

```python
plt.subplot(1, 2, 2)

plt.plot(
    history.history['loss'],
    label='Train Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
```

## MODEL ACCURACY ANALYSIS

The training accuracy generally increases as the number of epochs increases. This shows that the CNN model is learning useful features from the training images.

Validation accuracy also improves during the training process. A small difference between training accuracy and validation accuracy indicates that the model is generalizing reasonably well.

If training accuracy becomes much higher than validation accuracy, it may indicate **overfitting**.

## MODEL LOSS ANALYSIS

Training loss decreases as the model learns from the training data.

Validation loss also decreases initially and may fluctuate after several epochs.

This indicates that the model is gradually converging toward an optimal solution.

---

# PAGE 9 – TEST PREDICTION, RESULT AND CONCLUSION

## STEP 9: PREDICT TEST IMAGES

The trained CNN model can now predict the class of unseen test images.

```python
def plot_predictions(index):

    img = X_test[index]

    true_label = class_names[
        np.argmax(y_test[index])
    ]

    pred_probs = model.predict(
        np.expand_dims(img, axis=0),
        verbose=0
    )

    pred_label = class_names[
        np.argmax(pred_probs)
    ]

    plt.imshow(img)

    plt.title(
        f"True: {true_label} | Pred: {pred_label}"
    )

    plt.axis('off')
    plt.show()
```

To predict the first five test images:

```python
for i in range(5):
    plot_predictions(i)
```

The output displays:

* The original test image
* The actual class label
* The predicted class label

## RESULT

The CIFAR-10 dataset was successfully loaded and preprocessed. A Convolutional Neural Network containing convolutional, pooling, dropout, flatten, and dense layers was successfully constructed.

The model was trained using the training dataset, and its performance was analyzed using training and validation accuracy and loss graphs. The trained model was also successfully used to predict the classes of unseen test images.

## CONCLUSION

Thus, the **Convolutional Neural Network (CNN)** was successfully implemented and trained using **TensorFlow and the CIFAR-10 dataset**.

The CNN successfully learned image features through convolutional layers and classified images into ten categories: **Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck**.

The performance of the model was evaluated using **accuracy and loss graphs**, and predictions were successfully generated for the test dataset.

Hence, the objective of **building and training a CNN for image classification** was successfully achieved.

### AIM

To predict house prices using regression models and compare the performance of different Machine Learning regression models based on *RMSE, MAE, and R² score*.

### OBJECTIVES

* To load and analyze the House Price Dataset.
* To perform Exploratory Data Analysis (EDA).
* To identify and treat outliers.
* To split the dataset into training and testing data.
* To apply feature scaling.
* To train different regression models.
* To evaluate the models using RMSE, MAE, and R².
* To compare the performance of all regression models.
* To identify the best-performing model for house price prediction.

---

# PAGE 2 – THEORY

## MACHINE LEARNING

Machine Learning is a branch of Artificial In…
