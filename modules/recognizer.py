import cv2, os, pickle
import numpy as np
import onnxruntime as ort

class ArcFaceRecognizer:
    def __init__(self, model_path, db_path, thres=0.9, use_gpu=True):
        providers = ['CUDAExecutionProvider'] if use_gpu and ort.get_device() == "GPU" else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.db_path = db_path
        self.thres = thres
        self.db = self.load_db()

    def preprocess(self, face_img):
        face_img = cv2.resize(face_img, (112, 112))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = np.transpose(face_img, (2, 0, 1)).astype(np.float32) / 255.0
        return np.expand_dims(face_img, 0)

    def get_embedding(self, face_img):
        input_blob = self.preprocess(face_img)
        emb = self.session.run(None, {self.input_name: input_blob})[0]
        return emb.flatten()

    def recognize(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:
            return None, "Unknown"
        emb = self.get_embedding(face_img)
        best_name, best_score = "Unknown", 1.0
        for name, db_emb in self.db.items():
            dist = np.linalg.norm(emb - db_emb)
            if dist < best_score and dist < self.thres:
                best_name, best_score = name, dist
        return emb, best_name

    def load_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                return pickle.load(f)
        return {}

    def save_db(self):
        with open(self.db_path, "wb") as f:
            pickle.dump(self.db, f)

    def add_face(self, name, emb):
        self.db[name] = emb
        self.save_db()
