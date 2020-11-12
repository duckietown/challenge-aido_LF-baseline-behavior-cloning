#!/usr/bin/env python3

"""
This is a custom script developed by FRANK based on duckietown
joystick script in order to allow user drive duckietown with joystick
and obtain log for further training.
"""

import argparse
import json
import sys
import cv2
import time
import gym
import numpy as np
import pyglet
import math
import logging

from log_util import Logger, SteeringToWheelVelWrapper
from log_schema import Episode, Step

from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv

class HumanDriver:
    def __init__(self, env, max_episodes, max_steps, log_file=None, downscale=False,playback=True):
        if not log_file:
            log_file = f"dataset.log"
        self.env = env
        self.env.reset()
        self.datagen = Logger(self.env, log_file=log_file)
        self.episode = 1
        self.max_episodes = max_episodes
        self.pwm_converter = SteeringToWheelVelWrapper()
        #! Temporary Variable Setup:
        self.last_reward = 0
        self.playback_buffer = []
        #! Enter main event loop
        pyglet.clock.schedule_interval(
            self.update, 1.0 / self.env.unwrapped.frame_rate, self.env)
        #! Get Joystick
        self.joysticks = pyglet.input.get_joysticks()
        assert self.joysticks, 'No joystick device is connected'
        self.joystick = self.joysticks[0]
        self.joystick.open()
        self.joystick.push_handlers(self.on_joybutton_press)
        pyglet.app.run()
        #! Log and exit
        self.datagen.close()
        self.env.close()

    def sleep_after_reset(self, seconds):
        for remaining in range(seconds, 0, -1):
            sys.stdout.write("\r")
            sys.stdout.write("{:2d} seconds remaining.".format(remaining))
            sys.stdout.flush()
            time.sleep(1)
        sys.stdout.write("\rGO!            \n")
        return

    def playback(self):
        #! Render Image
        if args.playback:
            for entry in self.playback_buffer:
                (recorded,action,reward)=entry
                x = action[0]
                z = action[1]
                canvas = cv2.cvtColor(recorded, cv2.COLOR_YUV2RGB)
                pwm_left, pwm_right = self.pwm_converter.convert(x, z)
                print('Linear: ', x, ' Angular: ', z, 'Left PWM: ', round(
                    pwm_left, 3), ' Right PWM: ', round(pwm_right, 3), ' Reward: ', round(reward, 2))
                #! Speed bar indicator
                cv2.rectangle(canvas, (20, 240), (50, int(240-220*x)),
                            (76, 84, 255), cv2.FILLED)
                cv2.rectangle(canvas, (320, 430), (int(320-150*z), 460),
                            (76, 84, 255), cv2.FILLED)

                cv2.imshow('Playback', canvas)
                cv2.waitKey(20)
        #! User interaction for log commitment
        qa = input('1 to commit, 2 to abort:        ')
        while not(qa == '1' or qa == '2'):
            qa = input('1 to commit, 2 to abort:        ')
        if qa == '2':
            self.datagen.reset_episode()
            print('Reset log. Discard current...')
        else:
            print("Comitting Episode")
            self.datagen.on_episode_done()
        self.playback_buffer = [] # reset playback buffer
        return        

    def image_resize(self,image, width=None, height=None, inter=cv2.INTER_AREA):
        """
        Resize an image with a given width or a given height 
        and preserve the aspect ratio.
        """
        dim = None
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))
        resized = cv2.resize(image, dim, interpolation=inter)
        return resized

    def on_key_press(self,symbol, modifiers):
        """
        This handler processes keyboard commands that
        control the simulation
        """
        if symbol == key.BACKSPACE or symbol == key.SLASH:
            print('RESET')
            self.playback()
            self.env.reset()
            self.env.render()
            self.sleep_after_reset(5)
        elif symbol == key.PAGEUP:
            self.env.unwrapped.cam_angle[0] = 0
            self.env.render()
        elif symbol == key.ESCAPE or symbol == key.Q:
            self.env.close()
            sys.exit(0)
    
    def on_joybutton_press(self,joystick, button):
        """
        Event Handler for Controller Button Inputs
        Relevant Button Definitions:
        3 - Y - Resets Env.
        """

        # Y Button
        if button == 3:
            print('RESET')
            self.playback()
            self.env.reset()
            self.env.render()
            self.sleep_after_reset(5)

    def update(self,dt,env):
        """
        This function is called at every frame to handle
        movement/stepping and redrawing
        """

        #! Joystick no action do not record
        if round(self.joystick.z, 2) == 0.0 and round(self.joystick.y, 2) == 0.0:
            return

        #! Nominal Joystick Interpretation
        x = round(self.joystick.y, 2) * 0.9  # To ensure maximum trun/velocity ratio
        z = round(self.joystick.z, 2) * 3.0
        
        #! Joystick deadband
        # if (abs(round(joystick.y, 2)) < 0.01):
        #     z = 0.0

        # if (abs(round(joystick.z, 2)) < 0.01):
        #     x = 0.0

        #! DRS enable for straight line
        if self.joystick.buttons[6]:
            x = -1.0
            z = 0.0

        action = np.array([-x, -z])
        pwm_left, pwm_right = self.pwm_converter.convert(-x, -z)

        #! GO! and get next
        # * Observation is 640x480 pixels
        (obs, reward, done, info) = self.env.step(action)

        if reward != -1000:
            print('Current Command: ', action,
                  ' speed. Score: ', reward)
            if ((reward > self.last_reward-0.02) or True):
                print('log')

                #! resize to Nvidia standard:
                obs_distorted_DS = self.image_resize(obs, width=200)

                #! Image pre-processing
                height, width = obs_distorted_DS.shape[:2]
                cropped = obs_distorted_DS[0:150, 0:200]

                # NOTICE: OpenCV changes the order of the channels !!!
                cropped_final = cv2.cvtColor(cropped, cv2.COLOR_BGR2YUV)
                self.playback_buffer.append((cropped_final,action,reward))
                step = Step(cropped_final, reward, action, done)
                self.datagen.log(step, info)
                self.last_reward = reward
            else:
                print('Bad Training Data! Discarding...')
                self.last_reward = reward
        else:
            print('!!!OUT OF BOUND!!!')

        if done:
            self.playback()
            self.env.reset()
            self.env.render()
            self.sleep_after_reset(5)
            return

        self.env.render()


if __name__ == '__main__':
    #! Parser sector:
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default=None)
    parser.add_argument('--map-name', default='small_loop_cw')
    parser.add_argument('--draw-curve', default=False,
                        help='draw the lane following curve')
    parser.add_argument('--draw-bbox', default=False,
                        help='draw collision detection bounding boxes')
    parser.add_argument('--domain-rand', default=True,
                        help='enable domain randomization')
    parser.add_argument('--playback', default=True,
                        help='enable playback after each session')
    parser.add_argument('--distortion', default=True)
    parser.add_argument('--steps', default=1500,
                        help='number of steps to record in one batch')
    parser.add_argument("--nb-episodes", default=1200,
                        help='set the total episoded number', type=int)
    parser.add_argument("--logfile", type=str, default=None)
    parser.add_argument("--downscale", action="store_true")
    args = parser.parse_args()

    #! Start Env
    if args.env_name is None:
        env = DuckietownEnv(
            map_name="loop_pedestrians",
            max_steps=args.steps,
            draw_curve=args.draw_curve,
            draw_bbox=args.draw_bbox,
            domain_rand=args.domain_rand,
            distortion=args.distortion,
            accept_start_angle_deg=4,
            full_transparency=True,
        )
    else:
        env = gym.make(args.env_name)

    node = HumanDriver(env,max_episodes=args.nb_episodes, max_steps=args.steps, log_file=args.logfile, downscale = args.downscale,playback=args.playback)
