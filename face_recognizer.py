import face_recognition
import pickle as pkl
import os
import numpy as np
import json
from datetime import datetime

class FaceStorage():
    """ 
    Stores known faces information in the following format:
    face_encodings_metadata = {serialized_face_encoding : {'Name': name, 'Last_time_seen': last_time_seen}
    
    Faces are stored unserialized in an additional list "face_encodings" to improve performance when comparing faces.
    """

    def __init__(self, init_from_cache, cache_path, cache_file_name, expiration_time):
        self.face_encodings_metadata = {}
        self.face_encodings = []
        self.cache_path = cache_path
        self.cache_file_name = cache_file_name
        if init_from_cache:
            self.load_cache()
        self.expiration_time = expiration_time

    def serialize_face_encoding(self, face_encoding):
        return json.dumps(face_encoding.tolist())
    
    def deserialize_face_encoding(self, serialized_face_encoding):
        return np.array(json.loads(serialized_face_encoding))

    def add_face_to_dataset(self, name, face_encoding):
        """ Adds a face to the dataset """
        serialized_face_encoding = self.serialize_face_encoding(face_encoding)
        self.face_encodings_metadata[serialized_face_encoding] = {'Name': name, 'Last_time_seen': datetime.now()}
        self.face_encodings.append(face_encoding)

    def remove_face_from_dataset(self, serialized_face_encoding):
        """ Removes a face from the dataset """
        try:
            deserialized_face_encoding = self.deserialize_face_encoding(serialized_face_encoding)
            self.face_encodings.remove(deserialized_face_encoding)
        except:
            print("Removing face from dataset failed. Face not found in dataset.")
            print("Continuing with the rest of the program.") 
            pass
        else:
            _ = self.face_encodings_metadata.pop(serialized_face_encoding)

    def get_face_metadata_from_encoding(self, face_encoding):
        """ Returns the metadata of a face given its encoding """
        serialized_face_encoding = self.serialize_face_encoding(face_encoding)
        return self.face_encodings_metadata[serialized_face_encoding]

    def update_last_time_seen(self, serialized_face_encoding):
        """ Updates the last time a face was seen """
        self.face_encodings_metadata[serialized_face_encoding]['Last_time_seen'] = datetime.now()

    def get_time_since_last_seen(self, serialized_face_encoding):
        """ Returns the time since a face was last seen """
        last_time_seen = self.face_encodings_metadata[serialized_face_encoding]['Last_time_seen']
        time_difference = datetime.now() - last_time_seen
        return time_difference
    
    def is_face_expired(self, serialized_face_encoding):
        """ Checks if a face is still valid """
        time_since_last_seen = self.get_time_since_last_seen(serialized_face_encoding)
        # If the face has not been seen for more than the expiration time (hours) remove it from the database
        return time_since_last_seen.seconds > self.expiration_time * 3600
    
    def retrieve_metadata_of_nearest_match(self, face_encoding):
        """ Returns the nearest match to a face from the dataset """
        result = None
        matches = face_recognition.compare_faces(self.face_encodings, face_encoding)
        if True in matches:
            matched_face_encoding = self.face_encodings[matches.index(True)]
            result = self.get_face_metadata_from_encoding(matched_face_encoding)
        return result
    
    def save_cache(self):
        """ Saves the known faces to a pickle file """
        # Create folder if it doesn't exist
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        with open(f"{self.cache_path}/{self.cache_file_name}", 'wb') as f:
            pkl.dump(self.face_encodings_metadata, f)
            pkl.dump(self.face_encodings, f)

    def load_cache(self):
        """ Reads the known faces from a pickle file """
        print("Loading cache from file")
        if os.path.exists(f"{self.cache_path}/{self.cache_file_name}"):
            with open(f"{self.cache_path}/{self.cache_file_name}", 'rb') as f:
                self.face_encodings_metadata = pkl.load(f)
                self.face_encodings = pkl.load(f)
        else:
            print("Cache file not found. Starting with an empty database.")

class EphemeralFaceStorage(FaceStorage):
    """ Meant to handle storage and retrieval of faces viewed over the past X hours (X=expiration_time).
     Clean out expired faces from the dataset by running .clean() """

    def __init__(self, init_from_cache, cache_path, expiration_time):
        """ Expiration time is in hours """
        super(EphemeralFaceStorage, self).__init__(
            init_from_cache=init_from_cache,
            cache_path=cache_path,
            cache_file_name='temp.cache', # 'temp.cache' is used to distinguish it from 'permanent.cache
            expiration_time=expiration_time,
            )
    
    def remove_face_if_expired(self, serialized_face_encoding):
        """ Removes a face if it has passed expiration time """
        if self.is_face_expired(serialized_face_encoding):
            self.remove_face_from_dataset(serialized_face_encoding)

    def clean(self):
        """ Removes all expired faces from the dataset """
        for serialized_face_encoding in list(self.face_encodings_metadata.keys()):
            self.remove_face_if_expired(serialized_face_encoding)

class PermanentFaceStorage(FaceStorage):
    """ Handles storage and retrieval of known faces """
    def __init__(self, init_from_cache, cache_path, expiration_time):
        super(PermanentFaceStorage, self).__init__(
            init_from_cache=init_from_cache,
            cache_path=cache_path,
            cache_file_name='permanent.cache', # 'permanent.cache' is used to distinguish it from 'temp.cache
            expiration_time=expiration_time,
            )
    
class FaceRecognizer():
    """ 
    Implements a high-level interface for recognizing faces from a picture.

    Two types of storage are used:
    1. A PermanentFaceStorage object used to store and retrieve known faces
    2. An EphemeralFaceStorage object used to store and retrieve faces that have appeared recently

    The distinction between these two types of storage allows us to handle faces in a flexible manner
    
    """

    def __init__(self,
                 init_from_cache,
                 cache_path,
                 expiration_time):
        self.ephemeral_storage = EphemeralFaceStorage(init_from_cache=init_from_cache, 
                                              cache_path=cache_path,
                                              expiration_time=expiration_time)
        self.permanent_storage = PermanentFaceStorage(init_from_cache=init_from_cache, 
                                              cache_path=cache_path,
                                              expiration_time=expiration_time)

    def save_cache(self):
        """ Saves the known faces to a pickle file """
        print("Saving cache to file")
        self.permanent_storage.save_cache()
        self.ephemeral_storage.save_cache()

    def detect_faces(self, frame):
        # Find all the faces and face encodings in the current frame of video
        detected_face_locations = face_recognition.face_locations(frame)
        detected_face_encodings = face_recognition.face_encodings(frame, detected_face_locations)
        return detected_face_locations, detected_face_encodings

    def retrieve_metadata_from_faces(self, detected_face_encodings):
        # Remove expired faces from ephemeral storage before retreiving metadata
        self.ephemeral_storage.clean()

        faces_metadata = []
        for face_encoding in detected_face_encodings:
            # See if the face appears in the ephemeral storage
            metadata = self.ephemeral_storage.retrieve_metadata_of_nearest_match(face_encoding)
            if metadata:
                metadata["Code"] = 0 # Code 0 means the face has appeared recently
                faces_metadata.append(metadata)
            else:
                # If the face does not appear in the ephemeral storage, check the permanent storage
                metadata = self.permanent_storage.retrieve_metadata_of_nearest_match(face_encoding)
                if metadata:
                    metadata["Code"] = 1 # Code 1 means the face hasn't appeared recently but is known
                    faces_metadata.append(metadata)
                else:
                    # If the face does not appear in the permanent storage, it is a completely unknown face 
                    faces_metadata.append({"Name": "Unknown", "Last_time_seen": None, "Code": -1}) # Code -1 means we should take actions to handle the unknown face
        
        return faces_metadata