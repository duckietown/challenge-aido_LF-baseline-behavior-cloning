import argparse
import logging
import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from frankModel import FrankNet
from logReader import Reader

MODEL_NAME = "FrankNet"
logging.basicConfig(level=logging.INFO)


#! Default Configuration
EPOCHS = 100
INIT_LR = 1e-3
BATCH_SIZE = 64
LOG_DIR = "."
LOG_DIR = "/home/anthony/Duckietown/Datasets"
LOG_FILE = "slimds.log"

EXPERIMENTAL = False
OLD_DATASET = False


class DuckieTrainer:
    def __init__(
        self,
        epochs,
        init_lr,
        batch_size,
        log_dir,
        log_file,
        old_dataset,
        experimental,
    ):
        print("Observed TF Version: ", tf.__version__)
        print("Observed Numpy Version: ", np.__version__)

        self.create_dir()

        # 1. Load all the datas
        log_path = os.path.join(log_dir, log_file)
        logging.info(f"Loading Datafile {log_path}")
        self.observation, self.linear, self.angular = self.get_data(
            log_path, old_dataset
        )
        logging.info(f"Loading Datafile completed")

        # 2. Split training and testing
        (
            observation_train,
            observation_valid,
            linear_train,
            linear_valid,
            angular_train,
            angular_valid,
        ) = train_test_split(
            self.observation, self.linear, self.angular, test_size=0.2, shuffle=True
        )

        model = self.configure_model(lr=init_lr, epochs=epochs)

        callbacks_list = self.configure_callbacks()

        # 11. GO!
        history = model.fit(
            x=observation_train,
            y={"Linear": linear_train, "Angular": angular_train},
            validation_data=(
                observation_valid,
                {"Linear": linear_valid, "Angular": angular_valid},
            ),
            epochs=EPOCHS,
            callbacks=callbacks_list,
            shuffle=True,
            batch_size=BATCH_SIZE,
            verbose=0,
        )

        model.save(f"trainedModel/{MODEL_NAME}.h5")

    def create_dir(self):
        try:
            os.makedirs("trainedModel")
        except FileExistsError:
            print("Directory already exists!")
        except OSError:
            print(
                "Create folder for trained model failed. Please check system permissions."
            )
            exit()

    def configure_model(self, lr, epochs):
        losses = {"Linear": "mse", "Angular": "mse"}
        lossWeights = {"Linear": 1, "Angular": 2}
        model = FrankNet.build(200, 150)
        opt = tf.keras.optimizers.Adam(lr=lr, decay=lr / epochs)
        model.compile(
            optimizer=opt, loss=losses, loss_weights=lossWeights, metrics="mse"
        )
        return model

    def configure_callbacks(self):
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir="trainlogs/{}".format(
                f'{MODEL_NAME}-{datetime.now().strftime("%Y%m%d-%H")}'
            )
        )

        filepath1 = f"trainedModel/{MODEL_NAME}Best_Validation.h5"
        checkpoint1 = tf.keras.callbacks.ModelCheckpoint(
            filepath1, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
        )

        # ? Keep track of the best loss model
        filepath2 = f"trainedModel/{MODEL_NAME}Best_Loss.h5"
        checkpoint2 = tf.keras.callbacks.ModelCheckpoint(
            filepath2, monitor="loss", verbose=1, save_best_only=True, mode="min"
        )

        return [checkpoint1, checkpoint2, tensorboard]

    def get_data(self, file_path, old_dataset=False):
        """
        Returns (observation: np.array, linear: np.array, angular: np.array)
        """
        reader = Reader(file_path)

        observation, linear, angular = (
            reader.read() if old_dataset else reader.modern_read()
        )

        logging.info(
            f"""Observation Length: {len(observation)}
            Linear Length: {len(linear)}
            Angular Length: {len(angular)}"""
        )
        return (np.array(observation), np.array(linear), np.array(angular))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Parameter Setup")

    parser.add_argument(
        "--experimental",
        help="Set if to use the experimental data loading method.",
        action="store_true",
        default=EXPERIMENTAL,
    )
    parser.add_argument(
        "--old_dataset",
        help="Set to use the old data log format",
        action="store_true",
        default=OLD_DATASET,
    )
    parser.add_argument(
        "--epochs", help="Set the total training epochs", default=EPOCHS
    )
    parser.add_argument(
        "--learning_rate", help="Set the initial learning rate", default=INIT_LR
    )
    parser.add_argument("--batch_size", help="Set the batch size", default=BATCH_SIZE)
    parser.add_argument(
        "--log_dir", help="Set the training log directory", default=LOG_DIR
    )
    parser.add_argument(
        "--log_file", help="Set the training log file name", default=LOG_FILE
    )

    args = parser.parse_args()

    DuckieTrainer(
        epochs=args.epochs,
        init_lr=args.learning_rate,
        batch_size=args.batch_size,
        log_dir=args.log_dir,
        log_file=args.log_file,
        old_dataset=args.old_dataset,
        experimental=args.experimental,
    )
