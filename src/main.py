import os
import sys
import time
import cv2
import numpy as np
import math
from argparse import ArgumentParser
import logging as logging

from face_detection import FaceDetect
from facial_landmarks_detection import FacialLandmarks
from gaze_estimation import GazeEstimation
from head_pose_estimation import HeadPose
from mouse_controller import MouseController
from input_feeder import InputFeeder


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fd", "--face_detection_model", required=True, type=str,
                        help="Path to an .xml file with Face Detection trained model.")
    parser.add_argument("-fld", "--facial_landmarks_detection_model", required=True, type=str,
                        help="Path to an .xml file with Facial Landmark Detection trained model.")
    parser.add_argument("-hpe", "--head_pose_estimation_model", required=True, type=str,
                        help="Path to an .xml file with Head Pose Estimation trained model.")
    parser.add_argument("-ge", "--gaze_estimation_model", required=True, type=str,
                        help="Path to an .xml file with Gaze Estimation trained model.")

    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file or CAM")

    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")

    parser.add_argument("-pt", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Probability threshold for detection fitering.")
    parser.add_argument("-flags", "--output_flags", required=False, nargs='+',
                        default=[],
                        help="Specify flag for visualization for different model(Space separated if multiple values)"
                             "fdm for faceDetectionModel, lrm for landmarkRegressionModel"
                             "hp for headPoseEstimationModel, gze for gazeEstimationModel")

    return parser


def main():
    args = build_argparser().parse_args()
    logger = logging.getLogger('main')
    # initialize variables with the input arguments for easy access



    fdm=args.face_detection_model
    ldm=args.facial_landmarks_detection_model
    hpem=args.head_pose_estimation_model
    gem=args.gaze_estimation_model
    output_flags = args.output_flags
    input_filename = args.input
    device_name = args.device
    prob_threshold = args.prob_threshold
    cpu_extension=args.cpu_extension


    if input_filename.lower() == 'cam':
        feeder = InputFeeder(input_type='cam')
    else:
        if not os.path.isfile(input_filename):
            logger.error("Unable to find specified video file")
            exit(1)
        feeder = InputFeeder(input_type='video', input_file=input_filename)

    # initialize model
    face_detection_model = FaceDetect(fdm, device_name,cpu_extension,prob_threshold)
    landmark_detection_model = FacialLandmarks(ldm, device_name,cpu_extension,prob_threshold)
    head_pose_estimation_model = HeadPose(hpem, device_name,cpu_extension,prob_threshold)
    gaze_estimation_model = GazeEstimation(gem, device_name,cpu_extension,prob_threshold)

    mouse_controller = MouseController('medium', 'fast')

    # load Models
    start_model_load_time = time.time()
    face_detection_model.load_model()
    logger.info("Face Detection Model Loaded...")
    FDMT=time.time()-start_model_load_time
    start1=time.time()
    landmark_detection_model.load_model()
    logger.info("landmark_estimation Model Loaded...")
    LDMT=time.time()-start1
    start2=time.time()
    head_pose_estimation_model.load_model()
    logger.info("Head pose estimation model Loaded...")
    hpem=time.time()-start2
    start3=time.time()
    gaze_estimation_model.load_model()
    logger.info("Gaze_estimation model loaded..")
    gem=time.time()-start3
    total_model_load_time = time.time() - start_model_load_time

    feeder.load_data()
    
     
    frame_count = 0
    start_inference_time = time.time()
    for ret, frame in feeder.next_batch():

        if not ret:
            break
        
        frame_count += 1
        if frame_count%5==0:
            cv2.imshow('video',cv2.resize(frame,(500,500)))
        key = cv2.waitKey(60)
        try:
            
            image,fc = face_detection_model.predict(frame,args.prob_threshold)
            #face_cords1=face_cords[0]
            #face_c = face_cords1.astype(np.int8)
            #print (image.shape)
            if type(image) == int:
                logger.warning("Unable to detect the face")
                if key == 27:
                    break
                continue
            #for cord in face_c:
                #face1=cord.astype(np.int32) 
                    # cord = (xmin,ymin,xmax,ymax)
                    # get face landmarks
                    # crop face from frame
            #face = image[face_cords1[1]:face_cords1[3],face_cords1[0]:face_cords1[2]]
            #print (face.shape)
            left_eye_image, right_eye_image, eye_coords = landmark_detection_model.predict(image) #using the output of face detection model 
            print (left_eye_image.shape)
            print (right_eye_image.shape)
            pose_output = head_pose_estimation_model.predict(image)
            mouse_coord, gaze_vector = gaze_estimation_model.predict(left_eye_image, right_eye_image, pose_output)
            yaw = pose_output[0]
            pitch = pose_output[1]
            roll = pose_output[2]
            focal_length = 950.0
            scale = 50
            eye_buffer=10
            center_of_face = (image.shape[1] / 2, image.shape[0] / 2, 0)
            logger.info("processing frame "+str(frame_count))
        except Exception as e:
            logger.warning("Could not predict using model " + str(e) + " for frame " + str(frame_count))
            continue

        #image = cv2.resize(frame, (500, 500))
        total_inference_time=time.time()-start_inference_time
        
        
        if (len(output_flags)!=0):
            
            preview_frame = frame
            
            #for Face detection model
            if 'fdm' in output_flags:
                preview_frame = image
                cv2.rectangle(image,(fc[1],fc[3]),(fc[0],fc[2]),(0,0,255),3)
            
            #for landmark detection Model    
            if 'lrm' in output_flags:
                cv2.rectangle(image, (eye_coords[0][0]-eye_buffer, eye_coords[0][1]-eye_buffer), (eye_coords[0][2]+eye_buffer, eye_coords[0][3]+eye_buffer), (0,255,0), 3) #10 is eye buffer area from centre
                cv2.rectangle(image, (eye_coords[1][0]-eye_buffer, eye_coords[1][1]-eye_buffer), (eye_coords[1][2]+eye_buffer, eye_coords[1][3]+eye_buffer), (0,255,0), 3) #10 is the eye buffer area from centre  
             
            #for head pose Estimation Model   
            if 'hp' in output_flags:
                cv2.putText(preview_frame, "Pose Angles: yaw:{:.2f}  pitch:{:.2f}  roll:{:.2f}".format(yaw, pitch, roll), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.25, (255, 0, 0), 1)

            #for gaze estimation model
            if 'gze' in output_flags:
                cv2.putText(preview_frame, "Gaze Cords: x= {:.2f} , y= {:.2f} , z= {:.2f}".format(gaze_vector[0], gaze_vector[1], gaze_vector[2]), (10, 30),cv2.FONT_HERSHEY_COMPLEX, 0.25, (0, 255, 0), 1)
                x, y, w = int(gaze_vector[0]*12), int(gaze_vector[1]*12), 160
                le = cv2.line(left_eye_image, (x-w, y-w), (x+w, y+w), (255,0,255), 1)    #https://arxiv.org/pdf/1805.04771.pdf
                re = cv2.line(right_eye_image, (x-w, y-w), (x+w, y+w), (255,0,255), 1)


        cv2.imshow("Visualization", cv2.resize(preview_frame,(500,500)))
        #out_video.write(preview_frame)
        if frame_count%5==0:
            mouse_controller.move(mouse_coord[0], -1*mouse_coord[1])    
        if key==27:
                break
    logger.error("VideoStream ended...")
    print("total_model_load time is {:} ms".format(1000* total_model_load_time/frame_count))
    print("fps is {:}".format(int(feeder.get_fps())))
    print("total inference time is{:} ms".format(1000*total_inference_time/frame_count))
    print("fdmt loading time is{:} ms".format(1000*FDMT/frame_count))
    print("ldmt loading time is{:} ms".format(1000*LDMT/frame_count))
    print("hpem loading tiem{:} ms".format(1000*hpem/frame_count))
    print("gzem loading time{:} ms".format(1000*hpem/frame_count))
    cv2.destroyAllWindows()
    feeder.close()
if __name__ == '__main__':
    main() 
        
