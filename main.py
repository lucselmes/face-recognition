from helpers import *
import argparse
from threading import Thread
import tkinter as tk
from tkinter import Label, Entry, Button
import cv2
from PIL import Image, ImageTk

class VideoStream:
    def __init__(self, camera_index):
        self.video_handler = videoHandler(camera_index)
        self.face_recognizer = faceRecognizer()
        self.processing_unknown_face = False # Flag to avoid multiple prompts for the same face

    def create_custom_dialog(self, face_image):
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

    def prompt_for_name(self, face_image, face_encoding):
        name = self.create_custom_dialog(face_image)
        if name:
            print(f"Adding {name} to the database")
            self.face_recognizer.face_storage.add_face_from_image(name, face_encoding)

        self.processing_unknown_face = False

    def handle_unknown_face(self, frame, face_location, face_encoding):

        # Get the face image from the frame
        top, right, bottom, left = face_location
        print(top, right, bottom, left)
        face_image = frame[top:bottom, left:right]
        print(face_image)

        # Start the process for adding a new face in a new thread
        thread = Thread(target=self.prompt_for_name, args=(face_image, face_encoding))
        thread.start()

    def stream_video(self):
        # Variable process_this_frame will be used to skip everything other frame -> Less smooth but faster
        face_locations = []
        face_names = []
        face_encodings = []
        process_this_frame = True

        while True:
            ret, frame = self.video_handler.get_frame()

            if process_this_frame:
                # Process the frame
                rgb_small_frame = self.video_handler.process_frame(frame)

                # Detect faces in the frame and compare this to existing face database
                face_locations, face_encodings = self.face_recognizer.detect_faces(rgb_small_frame)
                face_names = self.face_recognizer.compare_faces(face_encodings)

                # If a face is unknown, prompt user to add it to the database
                for i, name in enumerate(face_names):
                    if name == "Unknown" and not self.processing_unknown_face:
                        self.processing_unknown_face = True # Set flag to avoid multiple prompts for the same face
                        self.handle_unknown_face(rgb_small_frame, face_locations[i], face_encodings[i])

            process_this_frame = not process_this_frame

            # Display the results
            self.video_handler.display_frame_with_faces(frame, face_locations, face_names)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
        self.video_handler.video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Face recognition using webcam')
    parser.add_argument('--camera-index', type=int, default=0, dest="camera_index", help='Index of the camera to use')
    args = parser.parse_args()
    stream = VideoStream(args.camera_index)

    # Check if the camera is working before starting the stream
    if not stream.video_handler.is_camera_working():
        # Exit the program
        print("Camera is not working")
        exit(1)
    else:
        print("Camera is working")
        # Run the program
        stream.stream_video()