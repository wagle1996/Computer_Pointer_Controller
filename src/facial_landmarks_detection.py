'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import logging as log
import sys
from model_reuse import Model
class FacialLandmarks(Model):
    '''
    Class for the facial_landmarks_detetion_Model.
    '''
    def __init__(self,model_name,device="CPU",extensions=None,threshold=0.6):
        Model.__init__(self, model_name, device='CPU', extensions=None)
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape
 


    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        img_processed = self.preprocess_input(image.copy())
        outputs = self.network.infer({self.input_name:img_processed})
        coords = self.preprocess_output(outputs,image)
        left_eye_x_min = coords['leftx'] - 10
        left_eye_x_max = coords['leftx'] + 10
        left_eye_y_min = coords['lefty'] - 10
        left_eye_y_max = coords['lefty'] + 10

        right_eye_x_min = coords['rightx'] - 10
        right_eye_x_max = coords['rightx'] + 10
        right_eye_y_min = coords['righty'] - 10
        right_eye_y_max = coords['righty'] + 10

        eye_coord = [[left_eye_x_min, left_eye_y_min, left_eye_x_max, left_eye_y_max],
                          [right_eye_x_min, right_eye_y_min, right_eye_x_max, right_eye_y_max]]
        left_eye_image = image[left_eye_y_min:left_eye_y_max, left_eye_x_min:left_eye_x_max]
        right_eye_image = image[right_eye_y_min:right_eye_y_max, right_eye_x_min:right_eye_x_max]

        return left_eye_image, right_eye_image, eye_coord
        
    def check_model(self):
        ''
            

    def preprocess_output(self, outputs,image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        h=image.shape[0]
        w=image.shape[1]
        #preprocessing_output  Adapted from https://github.com/Rahul24-06/Computer-pointer-Controller-using-Gaze-Estimation/blob/master/src/facial_landmarks_detection.py
        try:
            outputs = outputs[self.output_name][0]
            left_eye_x = int(outputs[0] * w)
            left_eye_y = int(outputs[1] * h)
            right_eye_x = int(outputs[2] * w)
            right_eye_y = int(outputs[3] * h)

        
        except Exception as e:
            log.error("Error While preprocessing output for facial landmark detection model" + str(e))  
        return {'leftx': left_eye_x, 'lefty': left_eye_y,
                'rightx': right_eye_x, 'righty': right_eye_y}     
