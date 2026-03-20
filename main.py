import cv2 
import numpy as np
import mediapipe as mp
from pathlib import Path
import os 


class MonkeyPoseClassifier:

    def __init__(self):
        pass 
    def load_images(self , folder:Path): 
        '''
        To Load the Images of Dataset
        '''
        pass 
    def detect_thumbs_up(self , hands_position):
        """
        To Detect weather the thumbs up or not 
        """
        pass
    def detect_thumbs_down(self , hands_position):
        """
        To Detect weather the thumbs down or not 
        Can remove this function if need handled by thumbsup function 
        """
        pass 
    def detect_index_finger_pointing_up(self , hands_position):
        '''
        To Detect weather the Index Finger Pointing Up or not
        '''
        pass
    def detect_hands_on_head(self , hands_position, pose_postion):
        """
        To Detect weather the hands are on the hand or not 
        """
        pass 
    def detect_hands_raised(self , pose_position):
        '''
        To detect weather the hands raised above the head or not 
        '''
        pass 
    def detect_mouth_open(self , pose_position):
        '''
        To Detect weather the Mouth is open or Not.
        '''
        pass 
    def detect_arms_crossed(self , pose_position):
        """
        To Detect weather arms are crossed or not .
        """
        pass
    def detect_hands_near_face(self , pose_position , hands_position):
        """
        To Detect hands near face or not 
        """
        pass
    def detect_hands_together(self , hands_position):
        """
        To Detect hands together holded or not  
        """
        pass
    def detect_mischievous(self, pose_landmarks, hand_landmarks):
        """
        To Detect Mischeievous
        """
        pass
    def eye_aspect_ratio(self, landmarks, eye_indices):
        """
        returns the eye_aspect ratio
        """
        pass
    def detect_wink(self, face_mesh_results):
        """
        To Detect eye wink or not 
        """
        pass 
    def detect_flirty(self, hand_landmarks, face_mesh_results):
        """
        To Detect filrty or not 
        """
        pass 
    def identify_match_pose(self,
                            pose_landmarks, hand_landmarks,
                            face_detected,face_mesh_results=None):
        """
        To match the pose of user and display the matched pose from dataset 
        """
        pass 
    def run(self):
        """
        To Runs this class 
        """
        pass 



def main_func():
    """
     Calls the Main Function 
    """
    pass 


if __name__ == '__main__':
    main_func()


