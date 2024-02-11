from face_recognizer import FaceRecognizer
from gui import GUIHandler
import argparse
import cv2

class RecognizerStream():
    def __init__(self, camera_index, resize_factor, init_from_cache):
        self.gui_handler = GUIHandler(camera_index, resize_factor)
        self.face_recognizer = FaceRecognizer(init_from_cache)
        self.processing_unknown_face = False # Flag to avoid a prompt for another unknown face when we are already processing one

    def handle_and_add_unknown_face(self, frame, face_location, face_encoding):

        # Display prompt to add the unknown face to the database
        name = self.gui_handler.handle_unknown_face(frame, face_location)

        # Add the unknown face to the database
        if name:
            print(f"Adding {name} to the database")
            self.face_recognizer.add_face_to_dataset(name, face_encoding)
        else:
            print("Unknown face not added to the database")

        self.processing_unknown_face = False

    def run_processing_logic(self, frame):
        # Process the frame
        rgb_small_frame = self.gui_handler.process_frame(frame)

        # Detect faces in the frame and compare this to existing face database
        face_locations, face_encodings = self.face_recognizer.detect_faces(rgb_small_frame)
        face_names = self.face_recognizer.compare_faces(face_encodings)

        # If a face is unknown, prompt user to add it to the database
        for i, name in enumerate(face_names):
            if name == "Unknown" and not self.processing_unknown_face:
                self.processing_unknown_face = True
                self.handle_and_add_unknown_face(frame, face_locations[i], face_encodings[i])

        return face_locations, face_names

    def stream_video(self):
        process_this_frame = True # process_this_frame will be used to skip everything other frame -> Less smooth but faster

        while True:
            ret, frame = self.gui_handler.get_frame()

            if process_this_frame:
                face_locations, face_names = self.run_processing_logic(frame)

            process_this_frame = not process_this_frame

            # Display the results
            self.gui_handler.display_frame_with_faces(frame, face_locations, face_names)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
        self.gui_handler.video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Face recognition using webcam')
    parser.add_argument('--camera-index', type=int, default=1, dest="camera_index", help='Index of the camera to use. Defaults to 1 (external camera). 0 is the built-in camera.')
    parser.add_argument('--resize-factor', type=float, default=0.25, dest="resize_factor", help='Amount to resize the video frames when processing')
    parser.add_argument('--init-from-cache', action='store_true', dest="init_from_cache", help='Initialize the face recognizer from the cache')
    args = parser.parse_args()
    stream = RecognizerStream(args.camera_index, args.resize_factor, args.init_from_cache)

    # Check if the camera is working before starting the stream
    if not stream.gui_handler.is_camera_working():
        # Exit the program
        print("Camera is not working")
        exit(1)
    else:
        print("Camera is working")
        # Run the program
        stream.stream_video()