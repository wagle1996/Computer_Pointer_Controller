from openvino.inference_engine import IECore, IENetwork
import cv2
import logging


class Model:
    #initializing and loading model adapted from my previous project of Udacity https://github.com/wagle1996/Smart-Queuing-System/blob/master/person_detect.py
    def __init__(self, model_name, device='CPU', extensions=None,threshold=0.6):
        self.model_name = model_name
        self.model_weights = ".".join(self.model_name.split(".")[:-1])+".bin"
        self.device=device
        self.extensions = extensions
        self.threshold=threshold
        self.input_name=None
        self.model_structure = self.model_name
        #print (self.model_weights)
        #try:
        #self.model = IECore().read_network(self.model_structure, self.model_weights)
        try:
            self.model = IENetwork(self.model_structure, self.model_weights)
        
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape


    def load_model(self):
        self.ie=IECore()
        self.network = self.ie.load_network(network=self.model, device_name=self.device, num_requests=1) 
        supported_layers_path = self.ie.query_network(network=self.model, device_name=self.device)
        keys=self.model.layers.keys()
        for l in keys:
            unsupported_layers_path=""
            if l not in supported_layers_path:
                unsupported_layers_path=l
        if len(unsupported_layers_path) != 0:
            log.error("Unsupported layers found: {}".format(unsupported_layers_path))
            sys.exit(1)

    def predict(self):
        pass

    def preprocess_output(self):
        pass

    def preprocess_input(self, image):
        """
        Input: image
        Description: We have done basic preprocessing steps
            1. Resizing image according to the model input shape
            2. Transpose of image to change the channels of image
            3. Reshape image
        Return: Preprocessed image
        """
        try:
            image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
            image = image.transpose((2, 0, 1))
            image = image.reshape(1, *image.shape)
        except Exception as e:
            self.logger.error("Error While preprocessing Image in " + str(self.model_name) + str(e))
        return image

    def wait(self):
        '''
        Checks the status of the inference request.
        '''
        status = self.network.requests[0].wait(-1)
        return status
