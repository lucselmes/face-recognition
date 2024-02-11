import face_recognition
import cv2
import numpy as np
import pickle as pkl

class videoHandler():
    """ Handles streaming of videos and displaying the video with the detected faces overlayed """

    def __init__(self, camera_index):
        self.video_capture = cv2.VideoCapture(camera_index)

    def is_camera_working(self):
        return self.video_capture.isOpened()

    def get_frame(self):
        """ Returns the current frame of the video """
        ret, frame = self.video_capture.read()
        return ret, frame
    
    def process_frame(self, frame):
        """ Processes the frame of the video """
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        return small_frame
    
    def display_frame_with_faces(self, frame, face_locations, face_names):

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

class faceStorage():

    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

    def add_face_from_image(self, name, face_encoding):
        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(name)
    
    def remove_face(self, name):
        index = self.known_face_names.index(name)
        self.known_face_encodings.pop(index)
        self.known_face_names.pop(index)

    def backup_faces_to_file(self):
        """ Saves the known faces to a pickle file """
        with open('known_faces.pkl', 'wb') as f:
            pkl.dump(self.known_face_encodings, f)
            pkl.dump(self.known_face_names, f)

    def read_faces_from_file(self):
        """ Reads the known faces from a pickle file """
        # If the attributes are not empty, then ask for confirmation of overwriting
        if self.known_face_encodings or self.known_face_names:
            print("Reading faces from file will overwrite current face encoding. Do you wish to continue? (y/n)")
            response = input()
            if response.lower() != 'y':
                return

        with open('known_faces.pkl', 'rb') as f:
            self.known_face_encodings = pkl.load(f)
            self.known_face_names = pkl.load(f)

class faceRecognizer():

    def __init__(self):
        self.face_storage = faceStorage()

    def detect_faces(self, frame):
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        return face_locations, face_encodings
    
    def compare_faces(self, face_encodings):
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.face_storage.known_face_encodings, face_encoding)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            if True in matches:
                first_match_index = matches.index(True)
                name = self.face_storage.known_face_names[first_match_index]

            face_names.append(name)
        
        return face_names