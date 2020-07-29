'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import numpy as np
from model_reuse import Model
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import logging as log
import sys
class HeadPose(Model):
    def __init__(self,model_name,device="CPU",extensions=None,threshold=0.6):
        Model.__init__(self, model_name, device='CPU', extensions=None)
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape
                
        
    def predict(self, image):
        try:
            self.preprocessed_image = self.preprocess_input(image)
            self.results = self.network.infer(inputs={self.input_name: self.preprocessed_image})
            self.output = self.preprocess_output(self.results)
        except Exception as e:
            log.error("Error While predicting from head pose estimation Model" + str(e))
        return self.output

    def check_model(self):
        pass


   
    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        output = []
        try:
            output.append(outputs['angle_y_fc'].tolist()[0][0])
            output.append(outputs['angle_p_fc'].tolist()[0][0])
            output.append(outputs['angle_r_fc'].tolist()[0][0])
        except Exception as e:
            log.error("Error While preprocessing output in Head Pose Estimation Model" + str(e))
        return output
        
