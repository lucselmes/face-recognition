import cv2

import tkinter as tk
from tkinter import Label, Entry, Button
from PIL import Image, ImageTk

import threading
import sys

class ReturnValueThread(threading.Thread):
    """ Allows us to grab return value of the thread """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.result = None

    def run(self):
        if self._target is None:
            return  # could alternatively raise an exception, depends on the use case
        try:
            self.result = self._target(*self._args, **self._kwargs)
        except Exception as exc:
            print(f'{type(exc).__name__}: {exc}', file=sys.stderr)  # properly handle the exception

    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        return self.result
    
class DialogBoxHandler():
    """ Handles tasks related with making dialog boxes """

    def __init__(self):
        pass
    
    def create_name_face_dialog(self, face_image):
        self.dialog_result = None
        window = tk.Tk()
        window.title("Unknown Face Detected")

        prompt_message = "Unknown face detected. Please enter the name of the person:"
        prompt_label = Label(window, text=prompt_message, wraplength=300)
        prompt_label.pack()

        # Convert the face image to a format that can be displayed in Tkinter
        img = Image.fromarray(face_image)
        imgtk = ImageTk.PhotoImage(image=img)

        # Display the face image
        img_label = Label(window, image=imgtk)
        img_label.image = imgtk  # Keep a reference!
        img_label.pack()

        # Entry for the name
        name_var = tk.StringVar()
        name_entry = Entry(window, textvariable=name_var)
        name_entry.pack()

        def on_ok():
            self.dialog_result = name_var.get()
            window.destroy()
        
        # OK button
        ok_button = Button(window, text="OK", command=on_ok)
        ok_button.pack()

        # Run the dialog
        window.mainloop()
        return self.dialog_result
    
    def run_prompt_on_seperate_thread(self, face_image):
        thread = ReturnValueThread(target=self.create_name_face_dialog, args=(face_image,))
        thread.start()
        return thread.join()

class VideoHandler():
    """ Handles tasks related to streaming of videos and handling and modifying individual frames """

    def __init__(self, camera_index, resize_factor):
        self.video_capture = cv2.VideoCapture(camera_index)
        self.resize_factor = resize_factor
        self.scale_up_factor = 1/resize_factor # Factor needed to scale up the face boundary boxes to the original frame size

    def is_camera_working(self):
        return self.video_capture.isOpened()

    def get_frame(self):
        """ Returns the current frame of the video """
        ret, frame = self.video_capture.read()
        return ret, frame
    
    def process_frame(self, frame):
        """ Processes the frame of the video """
        # Resize frame of video to smaller size (default 1/4 size) for faster face recognition processing
        return cv2.resize(frame, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
    
    def scale_up_boundary_box(self, face_location):
        """ Scales up a given boundary box to the original frame size """
        return tuple([int(x * self.scale_up_factor) for x in face_location])
    
    def display_frame_with_faces(self, frame, face_locations, face_names):

        # Display the results
        for face_location, name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled down
            top, right, bottom, left = self.scale_up_boundary_box(face_location)

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

    def picture_from_boundary_box(self, frame, face_location):
        """ Unpacks boundary box and returns the face image"""
        top, right, bottom, left = face_location
        return frame[top:bottom, left:right]

class GUIHandler(VideoHandler, DialogBoxHandler):
    """ Handles all GUI related tasks """

    def __init__(self, camera_index, resize_factor):
        super(GUIHandler, self).__init__(camera_index=camera_index, resize_factor=resize_factor)

    def handle_unknown_face(self, frame, face_location):
        # Get face image
        scaled_boundary_box = self.scale_up_boundary_box(face_location)
        face_image = self.picture_from_boundary_box(frame, scaled_boundary_box)

        # Start the process for putting a name to a unknown face in a new thread
        return self.run_prompt_on_seperate_thread(face_image)