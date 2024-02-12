from face_recognizer import FaceRecognizer
from gui import GUIHandler
import argparse
import cv2

class RecognizerStream():
    def __init__(self, camera_index, resize_factor, init_from_cache, cache_path, expiration_time):
        
        self.gui_handler = GUIHandler(camera_index=camera_index, 
                                      resize_factor=resize_factor)
        self.face_recognizer = FaceRecognizer(init_from_cache=init_from_cache,
                                                      cache_path=cache_path,
                                                      expiration_time=expiration_time)
        
        self.processing_unknown_face = False # Flag to avoid a prompt for another unknown face when we are already processing one

    def add_face_to_permanent_storage(self, name, face_encoding):
        # Add the unknown face to the database
        if name:
            print(f"Adding {name} to the database")
            self.face_recognizer.permanent_storage.add_face_to_dataset(name, face_encoding)
        else:
            print("Unknown face not added to the database")
        self.processing_unknown_face = False

    def handle_faces(self, frame, faces_metadata, face_locations, face_encodings):
        # Handle the faces case by case
        for i, faces_metadata in enumerate(faces_metadata):
            if faces_metadata['Code'] == -1: # This is an unknown face that is not stored in ephemeral storage
                self.handle_unknown_face(frame, face_locations[i], face_encodings[i])
            elif faces_metadata['Code'] == 1: # This is a known face that hasn't appeared recently
                self.handle_known_face()

    def handle_unknown_face(self, frame, face_location, face_encoding):
        # Add face to ephemeral storage
        self.face_recognizer.ephemeral_storage.add_face_to_dataset("Unknown", face_encoding)

        # Prompt user to add it to the permanent database
        if not self.processing_unknown_face:
            self.processing_unknown_face = True
            name = self.gui_handler.prompt_for_name_of_unknown_face(frame, face_location)
            self.add_face_to_permanent_storage(name, face_encoding)

    def handle_known_face(self):
        # If a face is known, do something, for example, greet the person, play some music or some sound effect
        pass
    
    def handle_key_press(self, key):
        if key == ord('q'):
            return True
        elif key == ord('s'):
            self.face_recognizer.save_cache()
            print("Current cache of faces saved to file")
        return False

    def stream_video(self):
        process_this_frame = True # process_this_frame will be used to skip everything other frame -> Less smooth but faster

        while True:
            ret, frame = self.gui_handler.get_frame()

            if process_this_frame:
                rgb_small_frame = self.gui_handler.process_frame(frame)

                # Check if any faces are detected
                face_locations, face_encodings = self.face_recognizer.detect_faces(rgb_small_frame)

                # Cross reference these faces with the database
                faces_metadata = self.face_recognizer.retrieve_metadata_from_faces(face_encodings)

                # Handle the faces case by case
                self.handle_faces(frame, faces_metadata, face_locations, face_encodings)

            process_this_frame = not process_this_frame
            face_names = [metadata['Name'] for metadata in faces_metadata]

            # Display the results
            self.gui_handler.display_frame_with_faces(frame, face_locations, face_names)

            key = cv2.waitKey(1) & 0xFF

            if self.handle_key_press(key):
                break

        # Release handle to the webcam
        self.gui_handler.video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Face recognition using webcam')
    parser.add_argument('--camera-index', type=int, default=1, dest="camera_index", help='Index of the camera to use. Defaults to 1 (external camera). 0 is the built-in camera.')
    parser.add_argument('--resize-factor', type=float, default=0.25, dest="resize_factor", help='Amount to resize the video frames when processing')
    parser.add_argument('--init-from-cache', action='store_false', dest="init_from_cache", help='Initialize the face recognizer from the cache')
    parser.add_argument('--cache-path', type=str, default="cache", dest="cache_path", help='Path to the cache directory for the face recognizer')
    parser.add_argument('--expiration-time', type=int, default=0.5, dest="expiration_time", help='Time in hours after which a face is considered expired')
    args = parser.parse_args()
    stream = RecognizerStream(camera_index=args.camera_index, 
                              resize_factor=args.resize_factor, 
                              init_from_cache=args.init_from_cache, 
                              cache_path=args.cache_path,
                              expiration_time=args.expiration_time)

    # Check if the camera is working before starting the stream
    if not stream.gui_handler.is_camera_working():
        # Exit the program
        print("Camera is not working")
        exit(1)
    else:
        print("Camera is working")
        # Run the program
        stream.stream_video()