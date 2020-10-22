import tensorflow as tf
from frankModel import FrankNet  # The model we are gonna use to train
from logReader import Reader
from sklearn.model_selection import train_test_split
import time
import numpy as np
import os
print("Observed TF Version: ",tf.__version__)
print("Observed Numpy Version: ",np.__version__)

#! Training Configuration
EPOCHS = 1000000 #EPOCHS
INIT_LR = 1e-3   #LEARNING RATE
BS = 8          #Batch Size 
GPU_COUNT = 1    # Change this value if you are using multiple GPUs
MULTI_GPU = True #Change this to enable multi-GPU

#! Log Interpretation
STORAGE_LOCATION = "trained_models/behavioral_cloning"

#! Global training data storage
# TODO: This should be optimized?
observation = []
linear = []
angular = []


def load_data():
    global observation, linear, angular
    reader = Reader('train.log')
    observation, linear, angular = reader.read()
    observation = np.array(observation)
    linear = np.array(linear)
    angular = np.array(angular)
    print('Observation Length: ', len(observation))
    print('Linear Length: ', len(linear))
    print('Angular Length: ', len(angular))
    # exit()
    return


# -----------------------------------------------------------------------------
# Define custom loss functions for regression in Keras
# -----------------------------------------------------------------------------
# root mean squared error (rmse) for regression
def rmse(y_true, y_pred):
    from tensorflow.keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# mean squared error (mse) for regression
def mse(y_true, y_pred):
    from tensorflow.keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)

# coefficient of determination (R^2) for regression
def r_square(y_true, y_pred):
    from tensorflow.keras import backend as K
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


#!================================================================
# 0. Create directories
try:
    os.makedirs("trainedModel")
except FileExistsError:
    print("Directory already exists!")
except OSError:
    print("Create folder for trained model failed. Please check system permissions.")
    exit()    

# 1. Load all the datas
load_data()
print('Load all complete')

# 2. Split training and testing
observation_train, observation_valid, linear_train, linear_valid, angular_train, angular_valid = train_test_split(
    observation, linear, angular, test_size=0.2, shuffle=True)

# 3. Build the model using strategy
if MULTI_GPU:
    print("Currently using multiple GPUs")
    if tf.config.list_physical_devices('gpu'):
        strategy = tf.distribute.MirroredStrategy()
    else:
        print("Selected MultiGPU but only observe single GPU. Cancel Multi GPU training!")
        strategy = tf.distribute.get_strategy()
else:
    print("Currently using single GPUs")
    strategy = tf.distribute.get_strategy()

with strategy.scope():
    model = FrankNet.build(200, 150)

# 4. Define the loss function and weight
losses = {
    "Linear": "mse",
    "Angular": "mse"
}
lossWeights = {"Linear": 0.001, "Angular": 10.0}

# 5. Select optimizer
opt = tf.keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# 6. Select loss metrics
metrics_list = ["mse", rmse, r_square]

# 8. Compile model and plot to see
model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
              metrics=metrics_list)

# 9. Setup tensorboard
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir='logs/{}'.format(time.ctime()))

# 10. checkpoint
#? Keep track of the best validation loss model
filepath1 = "trainedModel/FrankNetBest_Validation.h5"
checkpoint1 = tf.keras.callbacks.ModelCheckpoint(
    filepath1, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

#? Keep track of the best loss model
filepath2 = "trainedModel/FrankNetBest_Loss.h5"
checkpoint2 = tf.keras.callbacks.ModelCheckpoint(
    filepath2, monitor='loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint1, checkpoint2, tensorboard]

# 11. GO!
history = model.fit(observation_train,
                    {"Linear": linear_train,
                        "Angular": angular_train}, validation_data=(observation_valid, {
                            "Linear": linear_valid, "Angular": angular_valid}),
                    epochs=EPOCHS, callbacks=callbacks_list, verbose=0)

model.save('trainedModel/FrankNet.h5')
