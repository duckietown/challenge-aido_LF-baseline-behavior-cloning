import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from frankModel import FrankNet
from logReader import Reader

import time
import os
import argparse

print("Observed TF Version: ", tf.__version__)
print("Observed Numpy Version: ", np.__version__)

#! Training Configuration
EPOCHS = 100
INIT_LR = 1e-3
BATCH_SIZE = 64
LOG_DIR = 'train.log'
EXPERIMENTAL = False
OLD_DATASET = False
observation = []
linear = []
angular = []


def configurables():
    global EPOCHS,INIT_LR,BATCH_SIZE,LOG_DIR,OLD_DATASET,EXPERIMENTAL
    parser = argparse.ArgumentParser(description="Training Parameter Setup")
    parser.add_argument('--experimental', help='Set if to use the experimental data loading method.',
                        action='store_true', default=False)
    parser.add_argument('--old_dataset', help='Set to use the old data log format',
                        action='store_true', default=False)
    parser.add_argument(
        '--epochs', help='Set the total training epochs', default=100)
    parser.add_argument('--learning_rate',
                        help='Set the total training epochs', default=1e-3)
    parser.add_argument('--batch_size', help='Set the batch size', default=16)
    parser.add_argument('--log_dir', help='Set the training log directory',default='train.log')
    args = parser.parse_args()
    EPOCHS = int(args.epochs)
    INIT_LR = float(args.learning_rate)
    BATCH_SIZE = int(args.batch_size)
    LOG_DIR = args.log_dir
    OLD_DATASET = args.old_dataset
    EXPERIMENTAL = args.experimental
    return


def load_data():
    global observation, linear, angular, OLD_DATASET
    reader = Reader(LOG_DIR)
    if OLD_DATASET:
        observation, linear, angular = reader.read()
    else:
        observation, linear, angular = reader.modern_read()
    observation = np.array(observation)
    linear = np.array(linear)
    angular = np.array(angular)
    print('Observation Length: ', len(observation))
    print('Linear Length: ', len(linear))
    print('Angular Length: ', len(angular))
    # exit()
    return


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
configurables()
load_data()
print('Load all complete')

# 2. Split training and testing
observation_train, observation_valid, linear_train, linear_valid, angular_train, angular_valid = train_test_split(
    observation, linear, angular, test_size=0.2, shuffle=True)

model = FrankNet.build(200, 150)

# 4. Define the loss function and weight
losses = {
    "Linear": "mse",
    "Angular": "mse"
}
lossWeights = {"Linear": 1, "Angular": 2}

# 5. Select optimizer
opt = tf.keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# 8. Compile model and plot to see
model.compile(optimizer=opt, loss=losses,
              loss_weights=lossWeights, metrics="mse")

# 9. Setup tensorboard
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir='logs/{}'.format(time.ctime()))

# 10. checkpoint
# ? Keep track of the best validation loss model
filepath1 = "trainedModel/FrankNetBest_Validation.h5"
checkpoint1 = tf.keras.callbacks.ModelCheckpoint(
    filepath1, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# ? Keep track of the best loss model
filepath2 = "trainedModel/FrankNetBest_Loss.h5"
checkpoint2 = tf.keras.callbacks.ModelCheckpoint(
    filepath2, monitor='loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint1, checkpoint2, tensorboard]

# 11. GO!
history = model.fit(x=observation_train,
                    y={"Linear": linear_train, "Angular": angular_train},
                    validation_data=(observation_valid,
                                     {"Linear": linear_valid, "Angular": angular_valid}),
                    epochs=EPOCHS,
                    callbacks=callbacks_list,
                    shuffle=True,
                    batch_size=BATCH_SIZE,
                    verbose=0)

model.save('trainedModel/FrankNet.h5')
