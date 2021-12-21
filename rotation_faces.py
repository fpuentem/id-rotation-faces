import cv2
import pytesseract # https://learnopencv.com/deep-learning-based-text-recognition-ocr-using-tesseract-and-opencv/
import numpy as np
import re
import imutils
import time
import dlib
import os

DEBUG = 1

def convert_and_trim_bb(image, rect):
	# extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	# compute the width and height of the bounding box
	w = endX - startX
	h = endY - startY
	# return our bounding box coordinates
	return (startX, startY, w, h)

def rotate(path_file):
    # Read image
    image = cv2.imread(path_file)

    # Resize
    image_small = imutils.resize(image, width=768)
    # Convert the image to grayscale and flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.bitwise_not(gray)
    if DEBUG:
        start = time.time()
        print("[INFO] performing angle detection with pytesseract...")
        
    rot_data = pytesseract.image_to_osd(image_small)
    
    if DEBUG:
        end = time.time()
        print("[INFO] angle detection took {:.4f} seconds".format(end - start))    
        print("[OSD] " + rot_data)
    
    rot = re.search('(?<=Rotate: )\d+', rot_data).group(0)

    angle = float(rot)
    
    if DEBUG:
        print("[ANGLE] " + str(angle))
    
    # rotate the image to deskew it
    rotated = imutils.rotate_bound(image, angle) #added
    
    # base name of file 
    base = os.path.splitext(path_file)[0]
    
    if DEBUG:
        filename = os.path.join(os.path.dirname(__file__), 'pics/', base + '_rot_DEBUG.jpg')

    else: 
        filename = os.path.join(os.path.dirname(__file__), 'pics/', base + '_rot.jpg')
    
    # Save image with correct orientation
    cv2.imwrite(filename, rotated)

    return filename

def find_faces(path_file):
    # load dlib's HOG + Linear SVM face detector
    if DEBUG:    
        print("[INFO] loading HOG + Linear SVM face detector...")
    
    detector = dlib.get_frontal_face_detector()

    # load the input image from disk, resize it, and convert it from
    # BGR to RGB channel ordering (which is what dlib expects)
    image = cv2.imread(path_file)
    image_small = imutils.resize(image, width=600)
    rgb = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)
    # perform face detection using dlib's face detector
    if DEBUG:
        start = time.time()
        print("[INFO] performing face detection with dlib...")
    
    rects = detector(rgb, 1)
    
    if DEBUG:
        end = time.time()
        print("[INFO] face detection took {:.4f} seconds".format(end - start))
    # convert the resulting dlib rectangle objects to bounding boxes,
    # then ensure the bounding boxes are all within the bounds of the
    # input image

    boxes = [convert_and_trim_bb(image_small, r) for r in rects]

    if DEBUG:
        # draw the bounding box on our image
        for (x, y, w, h) in boxes:
            cv2.rectangle(image_small, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # base name of file 
    base = os.path.splitext(path_file)[0]

    if boxes:
        faces = True
        if DEBUG:
            filename = os.path.join(os.path.dirname(__file__), 'pics/', base + '_front_DEBUG.jpg')

        else: 
            filename = os.path.join(os.path.dirname(__file__), 'pics/', base + '_front.jpg')
        
        
        cv2.imwrite(filename, image)
    
    else:
        faces = False
        filename = None

    return (faces, filename)

if __name__ == '__main__':
    # Tests
    ## Front image 
    print("******** Test 1 ************")
    print("Rotate:")
    correct_file = rotate('/home/elf/projects/vt-projects/id-rotation-faces/pics/CD_IMAGEVISREAR.jpg')
    print(correct_file)
    print("Is it the front?:")
    f, p = find_faces(correct_file)
    print(f)
    print(p) 

    ## Back image 
    print("******** Test 2 ************")
    print("Rotate:")
    correct_file = rotate('/home/elf/projects/vt-projects/id-rotation-faces/pics/CD_IMAGEVIS.jpg')
    print(correct_file)
    print("Is it the front?:")
    f, p = find_faces(correct_file)
    print(f)
    print(p)