import tkinter as tk
from tkinter import filedialog, Text
# import os
import cv2
import numpy as np
import sys
import time
import PIL
from PIL import ImageTk, Image
import os
# import imageio

# https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
prototxt_path = "weights/deploy.prototxt.txt"
# https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel 
model_path = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# load Caffe model
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
apps = []
root = tk.Tk()
root.title('welcome!')
file = ''
def blurimage(file):
    apps.append(file)
    # read the desired image
    image = cv2.imread(file)
    # get width and height of the image
    h, w = image.shape[:2]
    # gaussian blur kernel size depends on width and height of original image
    kernel_width = (w // 7) | 1
    kernel_height = (h // 7) | 1
    # preprocess the image: resize and performs mean subtraction
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    # set the image into the input of the neural network
    model.setInput(blob)
    # perform inference and get the result
    output = np.squeeze(model.forward())
    for i in range(0, output.shape[0]):
        confidence = output[i, 2]
        # get the confidence
        # if confidence is above 40%, then blur the bounding box (face)
        if confidence > 0.4:
            # get the surrounding box cordinates and upscale them to original image
            box = output[i, 3:7] * np.array([w, h, w, h])
            # convert to integers
            start_x, start_y, end_x, end_y = box.astype(np.int)
            # get the face image
            face = image[start_y: end_y, start_x: end_x]
            # apply gaussian blur to this face
            face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 0)
            # put the blurred face into the original image
            image[start_y: end_y, start_x: end_x] = face
    cv2.imwrite(file.split(".")[0]+"_blurred.jpg", image)
    img = Image.open(file.split(".")[0]+"_blurred.jpg")
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = tk.Label(frame, image=img)
    panel.image = img
    panel.pack()
    # cv2.imshow("image", image)
    cv2.waitKey(0)



def blurvideo(file):
    apps.append(file)
    cap = cv2.VideoCapture(file)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    _, image = cap.read()
    print(image.shape)
    out = cv2.VideoWriter("output.avi", fourcc, 20.0, (image.shape[1], image.shape[0]))
    while True:
        start = time.time()
        captured, image = cap.read()
        # get width and height of the image
        if not captured:
            break
        h, w = image.shape[:2]
        kernel_width = (w // 7) | 1
        kernel_height = (h // 7) | 1
        # preprocess the image: resize and performs mean subtraction
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        # set the image into the input of the neural network
        model.setInput(blob)
        # perform inference and get the result
        output = np.squeeze(model.forward())
        for i in range(0, output.shape[0]):
            confidence = output[i, 2]
            # get the confidence
            # if confidence is above 40%, then blur the bounding box (face)
            if confidence > 0.4:
                # get the surrounding box cordinates and upscale them to original image
                box = output[i, 3:7] * np.array([w, h, w, h])
                # convert to integers
                start_x, start_y, end_x, end_y = box.astype(np.int)
                # get the face image
                face = image[start_y: end_y, start_x: end_x]
                # apply gaussian blur to this face
                face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 0)
                # put the blurred face into the original image
                image[start_y: end_y, start_x: end_x] = face
        cv2.imshow("image", image)
        if cv2.waitKey(1) == ord("q"):
            break
        time_elapsed = time.time() - start
        fps = 1 / time_elapsed
        # print("FPS:", fps)
        out.write(image)
    cv2.destroyAllWindows()
    cap.release()
    out.release()

# def stopencoding():
#     cv2.destroyAllWindows()
#     print("stopping...")
#     cap.release()
#     out.release()
    



# def video_stream():
#     file = "output.avi"
#     app = tk.Frame(root, bg="white")
#     lmain = tk.Frame(app)
#     lmain = tk.Label(app)
#     lmain.grid()
#     cap = cv2.VideoCapture(0)
#     _, frame = cap.read()
#     cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
#     img = Image.fromarray(cv2image)
#     imgtk = ImageTk.PhotoImage(image=img)
#     lmain.imgtk = imgtk
#     lmain.configure(image=imgtk)
#     lmain.after(1, video_stream)


def cleanoutput():
    for x in apps:
        if x.endswith(".jpg"):
            x = x.split(".")[0]+"_blurred.jpg"
        elif x.endswith(".3gp"):
            x = "output.avi"
        if os.path.exists(x):
            os.remove(x)
            # apps.pop(x)
        else:
            print(x+" was not found")


def blurrealtime():
    cap = cv2.VideoCapture(0)
    while True:
        start = time.time()
        _, image = cap.read()
        # get width and height of the image
        h, w = image.shape[:2]
        kernel_width = (w // 7) | 1
        kernel_height = (h // 7) | 1
        # preprocess the image: resize and performs mean subtraction
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        # set the image into the input of the neural network
        model.setInput(blob)
        # perform inference and get the result
        output = np.squeeze(model.forward())
        for i in range(0, output.shape[0]):
            confidence = output[i, 2]
            # get the confidence
            # if confidence is above 40%, then blur the bounding box (face)
            if confidence > 0.4:
                # get the surrounding box cordinates and upscale them to original image
                box = output[i, 3:7] * np.array([w, h, w, h])
                # convert to integers
                start_x, start_y, end_x, end_y = box.astype(np.int)
                # get the face image
                face = image[start_y: end_y, start_x: end_x]
                # apply gaussian blur to this face
                face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 0)
                # put the blurred face into the original image
                image[start_y: end_y, start_x: end_x] = face
        cv2.imshow("image", image)
        if cv2.waitKey(1) == ord("q"):
            break
        time_elapsed = time.time() - start
        fps = 1 / time_elapsed
        # print("FPS:", fps)
    cv2.destroyAllWindows()
    cap.release()




def selectImage():
    for widget in frame.winfo_children():
        widget.destroy()
    openapp = filedialog.askopenfilename(initialdir="./", title="select file"
    ,filetypes=(("images", "*.jpg"),("all", "*.*")))
    file = openapp
    # print(file)
    label = tk.Label(frame,text=file,bg="#471651")
    label.pack()
    blurimage(file)




def selectVideo():
    for widget in frame.winfo_children():
        widget.destroy()
    openapp = filedialog.askopenfilename(initialdir="./", title="select file"
    ,filetypes=(("videos", "*.3gp"),("all", "*.*")))
    file = openapp
    print(file)
    # label = tk.Label(frame,text=file,bg="#471651")
    # label.pack()
    blurvideo(file)



canvas = tk.Canvas(root, height = 500, width=500, bg="#fa66e6")
canvas.pack()

frame = tk.Frame(root, bg="#c666fa")
frame.place(relwidth = 0.8, relheight=0.8, relx=0.1, rely=0.1)

openImg = tk.Button(root, text = "Blur Image", padx=10,pady=10, fg="white", bg="#fa66e6", command=selectImage)
openImg.pack()

openVid = tk.Button(root, text = "Blur Video", padx=10,pady=10, fg="white", bg="#fa66e6", command=selectVideo)
openVid.pack()

openCam = tk.Button(root, text = "Realtime", padx=10,pady=10, fg="white", bg="#fa66e6", command=blurrealtime)
openCam.pack()

deleteoutput = tk.Button(root,text = "delete output", padx=10,pady=10, fg="white", bg="#fa66e6", command=cleanoutput)
deleteoutput.pack()

# stopeverything = tk.Button(root,text = "stop", padx=10,pady=10, fg="white", bg="#fa66e6", command=stopencoding)
# stopeverything.pack()

root.mainloop()