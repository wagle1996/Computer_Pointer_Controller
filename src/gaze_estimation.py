'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

from model_reuse import Model
import time
import math
import logging as log
import numpy as np
class GazeEstimation(Model):
    """
    Class for the Gaze Estimation Model.
    """

    def __init__(self, model_path, device='CPU', extensions=None, threshold=0.6):
        """
        This will initiate Gaze Estimation Model class object
        """
        Model.__init__(self, model_path, device, extensions, threshold)
        self.model_name = 'Gaze Estimation Model'
        self.input_name = [i for i in self.model.inputs.keys()]
        self.input_shape = self.model.inputs[self.input_name[1]].shape
        self.output_name = [o for o in self.model.outputs.keys()]

    def predict(self, left_eye_image, right_eye_image, hpa, request_id=0):
        """
        This method will take image as a input and
        does all the preprocessing, postprocessing
        """
        try:
            left_eye_image = self.preprocess_input(left_eye_image)
            right_eye_image = self.preprocess_input(right_eye_image)
            self.network.start_async(request_id, inputs={'left_eye_image': left_eye_image,
                                                         'right_eye_image': right_eye_image,
                                                         'head_pose_angles': hpa})
            if self.wait() == 0:
                outputs = self.network.requests[0].outputs
                mouse_cord, gaze_vector = self.preprocess_output(outputs, hpa)
        except Exception as e:
            log.error("Error While Prediction in Gaze Estimation Model" + str(e))
        return mouse_cord, gaze_vector

    def preprocess_output(self, outputs, hpa):
        gaze_vector = outputs[self.output_name[0]][0]                                      # result is in orthogonal coordinate system (x,y,z. not yaw,pitch,roll)and not normalized
        mouse_cord = (0, 0)
        roll = hpa[2]
        try:                                         
            gaze_vector1 = gaze_vector / np.linalg.norm(gaze_vector)                            # normalize the gaze vector
            vcos = math.cos(math.radians(roll))
            vsin = math.sin(math.radians(roll))
            tmpx =  gaze_vector1[0]*vcos + gaze_vector1[1]*vsin
            tmpy = gaze_vector1[0]*vsin + gaze_vector1[1]*vcos
            mouse_cord=(tmpx,tmpy)
        except Exception as e:
            log.error("Error While preprocessing output in Gaze Estimation Model" + str(e))
        return mouse_cord, gaze_vector1

        
