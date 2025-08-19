# Face Detection with OpenCV

This is a simple Python project to detect faces in **images** and through a **webcam feed** using OpenCV's deep learning-based face detector.

## Requirements

* Python 3.x
* OpenCV
* NumPy
* imutils

Install dependencies:

```bash
pip install opencv-python numpy imutils
```

## Run Face Detection on Images

```bash
python detect_faces.py --image sample.jpg --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel --confidence 0.5
```

## Run Face Detection with Webcam

```bash
python detect_faces_webcam.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel --confidence 0.5
```

## Notes

* Press `q` while the webcam window is open to quit the video stream.
* Make sure to keep `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel` in the same project folder.
