import cv2
import numpy as np

def show_image(img):
	while(True):
		cv2.imshow('image',cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
		k = cv2.waitKey(0)
		if k == ord('w'):
			cv2.destroyAllWindows()
			return


def add_text_to_image(text, img):

	font                   = cv2.FONT_HERSHEY_SIMPLEX
	bottomLeftCornerOfText = (10,50)
	fontScale              = 0.5
	fontColor              = (255,0,0)
	lineType               = 2
	img2 = np.copy(img)
	cv2.putText(img2,text, 
	    bottomLeftCornerOfText, 
	    font, 
	    fontScale,
	    fontColor,
	    lineType)
	return img2

def convert_image_structure(image):
	img_tr = image.transpose((1, 2, 0))
	img_tr = np.rot90(img_tr, axes=(1,0))
	return img_tr


def convert_1channel_to_3channel(image):
	img = image.copy()
	# reshape for imshow
	img_ = np.empty([img.shape[2], img.shape[3], 3])
	img_[:,:,0] = img[:,:].T
	#img_[:,:,1] = img[:,:].T
	#img_[:,:,2] = img[:,:].T
	return img_

def create_image_with_bb(image, bb):
	img = image.copy()
	cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])),(255,0,0),1)
	return img

def rotate_bb(bb, m):
	return (m - bb[3], bb[0], m - bb[1], bb[2])


def make_square_image(image):
	n = image.shape[0]
	m = image.shape[1]
	if n > m:
		image_sq = np.pad(image, ((0,0), (0,n-m), (0,0)), mode = 'constant', constant_values=0)
	else:
		image_sq = np.pad(image, ((0,m-n), (0,0), (0,0)), mode = 'constant', constant_values=0)
	return image_sq

def make_square_image_middle(image):
	n = image.shape[0]
	m = image.shape[1]
	if n > m:
		size1 = int((n-m)/2)
		size2 = int((n-m+1)/2)
		image_sq = np.pad(image, ((0,0), (0,size1), (0,0)), mode = 'constant', constant_values=0)
		image_sq = np.pad(image_sq, ((0,0), (size2,0), (0,0)), mode = 'constant', constant_values=0)
	else:
		size1 = int((m-n)/2)
		size2 = int((m-n+1)/2)
		image_sq = np.pad(image, ((0,size1), (0,0), (0,0)), mode = 'constant', constant_values=0)
		image_sq = np.pad(image_sq, ((size2, 0), (0,0), (0,0)), mode = 'constant', constant_values=0)

	return image_sq

def resize_image(image, size):
	n = image.shape[0]
	m = image.shape[1]
	return cv2.resize(image, (0,0), fx=size/m, fy=size/n) 

def resize_bb(bb, size_image, size):
	xmin = bb[0] * size / float(size_image)
	ymin = bb[1] * size / float(size_image)
	xmax = bb[2] * size / float(size_image)
	ymax = bb[3] * size / float(size_image)
	return (xmin, ymin, xmax, ymax)

