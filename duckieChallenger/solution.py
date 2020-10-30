#!/usr/bin/env python3
import cv2
import numpy as np
import tensorflow as tf

from aido_schemas import (Context, DB20Commands, DB20Observations, EpisodeStart, JPGImage,
                          LEDSCommands, protocol_agent_DB20, PWMCommands, RGB, wrap_direct)


from helperFncs import Validation_Functions, SteeringToWheelVelWrapper, image_resize

MODEL = "FrankNet.h5"

#! Global Config
expect_shape = (480, 640, 3)
convertion_wrapper = SteeringToWheelVelWrapper()
eval_func = Validation_Functions()


class TensorflowTemplateAgent:

    def __init__(self):
        # define observation and output shapes
        self.dependencies = {'rmse': eval_func.rmse, 'mse': eval_func.mse,
                             'r_square': eval_func.r_square, 'r_square_loss': eval_func.r_square_loss}
        self.model = tf.keras.models.load_model(
                MODEL, custom_objects=self.dependencies)
        self.current_image = np.zeros(expect_shape)
        self.input_image = np.zeros((150, 200, 3))
        self.to_predictor = np.expand_dims(self.input_image, axis=0)

        #! for fun
        #self.led_counter = 0

    def init(self, context: Context):
        context.info('init()')

    def on_received_seed(self, data: int):
        np.random.seed(data)

    def on_received_episode_start(self, context: Context, data: EpisodeStart):
        context.info(f'Starting episode "{data.episode_name}".')

    #! Image pre-processing here
    def on_received_observations(self, data: DB20Observations):
        camera: JPGImage = data.camera
        self.current_image = jpg2rgb(camera.jpg_data)
        self.input_image = image_resize(self.current_image, width=200)
        self.input_image = self.input_image[0:150, 0:200]
        self.input_image = cv2.cvtColor(self.input_image, cv2.COLOR_RGB2YUV)
        self.to_predictor = np.expand_dims(self.input_image, axis=0)


    #! Modification here! Return with action

    def compute_action(self, observation):
        (linear, angular) = self.model.predict(observation)
        return linear, angular

    #! Major Manipulation here Should not always change
    def on_received_get_commands(self, context: Context):
        linear, angular = self.compute_action(self.to_predictor)
        #！ Speed surpression:
        linear = linear * 0.85
        angular = angular * 0.95
        #! Inverse Kinematics
        pwm_left, pwm_right = convertion_wrapper.convert(linear, angular)
        pwm_left = float(np.clip(pwm_left, -1, +1))
        pwm_right = float(np.clip(pwm_right, -1, +1))

        #! LED Commands Sherrif Duck
        grey = RGB(0.0, 0.0, 0.0)
        red = RGB(255.0, 0.0, 0.0)
        blue = RGB(0.0, 0.0, 255.0)

        led_commands = LEDSCommands(red, grey, blue, red, blue)

        #! Do not modify here!
        pwm_commands = PWMCommands(motor_left=pwm_left, motor_right=pwm_right)
        commands = DB20Commands(pwm_commands, led_commands)
        context.write('commands', commands)

    def finish(self, context: Context):
        context.info('finish()')


def jpg2rgb(image_data: bytes) -> np.ndarray:
    """ Reads JPG bytes as RGB"""
    from PIL import Image
    import io
    im = Image.open(io.BytesIO(image_data))
    im = im.convert('RGB')
    data = np.array(im)
    assert data.ndim == 3
    assert data.dtype == np.uint8
    return data


def main():
    node = TensorflowTemplateAgent()
    protocol = protocol_agent_DB20
    wrap_direct(node=node, protocol=protocol)


if __name__ == '__main__':
    main()
