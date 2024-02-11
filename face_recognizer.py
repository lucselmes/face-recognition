import face_recognition
import pickle as pkl

class FaceStorage():
    """ Handles storage and retrieval of known faces """
    def __init__(self, init_from_cache):
        self.cache_path = "" # Need to implement and test caching logic
        if init_from_cache:
            self.read_faces_from_file()
        else:
            self.known_face_encodings = []
            self.known_face_names = []

    def add_face_to_dataset(self, name, face_encoding):
        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(name)
    
    def remove_face_from_dataset(self, name):
        index = self.known_face_names.index(name)
        self.known_face_encodings.pop(index)
        self.known_face_names.pop(index)

    def backup_faces_to_file(self):
        """ Saves the known faces to a pickle file """
        with open(self.cache_path, 'wb') as f:
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

        with open(self.cache_path, 'rb') as f:
            self.known_face_encodings = pkl.load(f)
            self.known_face_names = pkl.load(f)

class FaceRecognizer(FaceStorage):
    """ Handles tasks related to recognizing faces """

    def __init__(self, init_from_cache):
        super(FaceRecognizer, self).__init__(init_from_cache=init_from_cache)

    def detect_faces(self, frame):
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        return face_locations, face_encodings
    
    def compare_faces(self, face_encodings):
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]

            face_names.append(name)
        
        return face_names