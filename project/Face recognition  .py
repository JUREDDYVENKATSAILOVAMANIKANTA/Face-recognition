#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install --upgrade numpy opencv-python


# In[ ]:


import tkinter as tk
from PIL import Image, ImageTk
import cv2

class FaceRecognitionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Face Recognition App")

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(master, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.btn_snapshot = tk.Button(master, text="Snapshot", width=20, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        self.update()

    def snapshot(self):
        ret, frame = self.vid.read()
        if ret:
            cv2.imwrite("snapshot.png", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.master.after(10, self.update)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()


# In[ ]:


import tkinter as tk
from PIL import Image, ImageTk
import cv2

class FaceRecognitionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Face Recognition App")

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("trainer.yml")

        self.labels = ["Unknown", "Person1", "Person2"]  # Update with your labels
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.canvas = tk.Canvas(master, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.btn_snapshot = tk.Button(master, text="Snapshot", width=20, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        self.update()

    def snapshot(self):
        ret, frame = self.vid.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                id_, confidence = self.recognizer.predict(roi_gray)
                if confidence < 50:
                    label = self.labels[id_]
                else:
                    label = "Unknown"
                cv2.putText(frame, label, (x, y), self.font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
            cv2.imwrite("snapshot.png", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.master.after(10, self.update)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




