'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import logging as log
import numpy as np
from openvino.inference_engine import IECore, IENetwork
from model_reuse import Model


class FaceDetect(Model):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self,model_name,device="CPU",extensions=None,threshold=0.6):
        Model.__init__(self, model_name, device='CPU', extensions=None)
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape
    

    def predict(self, image, prob_threshold):
        '''
        This method is meant for running predictions on the input image.
        '''
        input_img = self.preprocess_input(image)  #preprocess input from model_reuse
        self.outputs = self.network.infer({self.input_name:input_img})
        face_coords = self.preprocess_output(self.outputs, prob_threshold)
        if (len(face_coords)==0):
            return 0, 0
        #face_coords = face_coords[0] # Taking first face
        height=image.shape[0]
        width=image.shape[1]
        for coords in face_coords:
            coords = coords* np.array([width, height, width, height])
            coords = coords.astype(np.int32)
            cropped_face = image[coords[1]:coords[3], coords[0]:coords[2]]

        #cropped_face = image[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]]   #cropping the face area
        return cropped_face, face_coords


    def check_model(self):
        pass

    def preprocess_output(self, outputs, prob_threshold):
        '''
        Method to process outputs before feeding them into the next model for
        inference or for the last step of the app
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''

        #preprocessing_output  Adapted from https://github.com/Rahul24-06/Computer-pointer-Controller-using-Gaze-Estimation/blob/master/src/facial_landmarks_detection.py
        try:
            face =[]
            output = outputs[self.output_name][0][0]
            for obj in output:
                if obj[2]>prob_threshold:
                    x_min=obj[3]
                    y_min=obj[4]
                    x_max=obj[5]
                    y_max=obj[6]
                    face.append([x_min,y_min,x_max,y_max])
        except Exception as e:
            log.error("Error While preprocessing output for facial detection model" + str(e))  
        return face   
