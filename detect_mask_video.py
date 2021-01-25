# USAGE
# python detect_mask_video.py

# import the necessary packages
# وارد کردن پکیج های مورد نیاز
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from playsound import playsound
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	# ابعاد فریم را گرفته و از روی آن یک blob درست می کند
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	#بلاب را از شبکه عصبی عبور داده و تشخیص چهره را انجام می دهد
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	# تعریف لیست چهره ها، مکان جهره ها در تصویر و احتمال اینکه ماسک داشته باشند
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		# احتمال اینکه چهره در عکس باشد
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		# حذف کردن احتمالات خیلی ضعیف، صورت هایی که احتمال پایین شناسایی دارند، حذف می شوند
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			# مختصات ایکس و ایگرگ باکسی که چهره در آن قرار می گیرد را به دست می دهد
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			# مطمین می شویم که باکس احاطه کننده ی چهره در تصویر قرار می گیرد
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			# مختصات چهره را پیدا می کند، کانال رنگ آن را از bgr به  rgb تغییر میدهد
			# و اندازه آن را به ۲۲۴ در ۲۲۴ تعییر می دهد
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			# چهره و باکس اطراف آن را به لیست چهره ها اضافه می کند
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	# فقط زمانی این کد را اجرا کن که چهره ای در تصویر شناسایی کرده
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		# برای محاسبه سریعتر، به جای اینکه تک تک چهره ها را یکی یکی به شبکه بدهیم، همه ی آنها را یکجا به شبکه می دهیم
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	# یک تیوپل دو بعدی از مکان چهره ها در تصویر را بر می گرداند
	return (locs, preds)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
# مدل تشخیص چهره را از دیسک می خوانیم
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
# مدل تشخیص ماسک را از دیسک می خوانیم
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
# دوربین را روشن کرده و اجازه می دهیم که راه اندازی شود
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	# استریم ویدیو را دریافت کرده و اندازه آن را به ۴۰۰ پیکسل تقلیل میدهد
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	# چهره های داخل فریم را تشخیص داده و مشخص می کند که ماسک دارند یا خیر
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	# حلقه فور برای همه چهره های داخل تصویر
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		# تشخیص لیبل و رنگ باکسی که چهره را در بر می گیرد
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# include the probability in the label
		# افزودن عدد احتمال به لیبل احاطه کننده ی عکس
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		# نمایش باکس رنگی و احتمال اینکه ماسک داشته باشد بر روی تصویر
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

		# Alarm when "No Mask" detected
		# پخش الارم اگر ماسک نداشته باشد
		if mask < withoutMask:
			path = os.path.abspath("Alarm.wav")
			playsound(path)

	# show the output frame
	# نمایش فریم خروجی
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	# اگر دکمه q فشرده شد، از لوپ حارج شود
	if key == ord("q"):
		break

# do a bit of cleanup
# تمیزکاری (پاک کردن فایل های اضافه و ازاد کردن رم)
cv2.destroyAllWindows()
vs.stop()
