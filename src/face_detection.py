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
        img_processed = self.preprocess_input(image.copy())
        outputs = self.network.infer({self.input_name:img_processed})
        coords = self.preprocess_output(outputs, prob_threshold)
        if (len(coords)==0):
            return 0, 0
        coords = coords[0] #take the first detected face
        h=image.shape[0]
        w=image.shape[1]
        coords = coords* np.array([w, h, w, h])
        coords = coords.astype(np.int32)
        
        cropped_face = image[coords[1]:coords[3], coords[0]:coords[2]]
        return cropped_face, coords

    def check_model(self):
        ''

    def preprocess_output(self, outputs, prob_threshold):
        '''
        Method to process outputs before feeding them into the next model for
        inference or for the last step of the app
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        try:
            coords =[]
            outs = outputs[self.output_name][0][0]
            for out in outs:
                conf = out[2]
                if conf>prob_threshold:
                    x_min=out[3]
                y_min=out[4]
                x_max=out[5]
                y_max=out[6]
                coords.append([x_min,y_min,x_max,y_max])
        except Exception as e:
            log.error("could not preprocess output")
        return coords

