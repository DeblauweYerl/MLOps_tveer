import numpy as np
from cv2 import INTER_CUBIC, MORPH_CLOSE, morphologyEx, resize
import cv2

from scipy import ndimage as ndi
from skimage.color import label2rgb, rgb2gray
from skimage.draw import polygon
from skimage.io import imread
from skimage.measure import find_contours
from skimage.morphology import disk
from skimage.transform import resize

import base64
import logging
import math
import os
import re
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from glob import glob
from io import BytesIO
import matplotlib.pyplot as plt

def print_multiple_images(*images, **kwargs):
    amount_of_images = len(images)
    rows = math.ceil(len(images) / 5)
    cols = 5 if len(images) > 5 else len(images)
    fig, axes = plt.subplots(rows, cols, figsize=(24, 3 * rows))
    axes_1d = axes.ravel() if amount_of_images > 1 else [axes]
    for i in range(len(axes_1d)):
        try:
            axes_1d[i].imshow(images[i], **kwargs)
            axes_1d[i].axis('off')    
            axes_1d[i].set_title(images[i].shape, fontdict={'fontsize': 20})
        except IndexError:
            axes_1d[i].axis('off')
    plt.show()

def load_image(img_url, background, crops = None):
    print(background.shape)
    original_image = imread('{}'.format(img_url))
    original_image = original_image[:, :, :3]
    if crops is not None:
        original_image = crop_sides(original_image, crops) # Cropping the sides left and right
    original_image = original_image / 255
    # resized_image = cv2.resize(original_image, dsize=new_size, interpolation=cv2.INTER_CUBIC)
    no_background = (original_image - background) # We still have to normalize the result, as it could be that we have negative values
    no_background -= np.min(no_background)
    no_background /= np.max(no_background)

    return (original_image, no_background)


# region utils

#functions----------------------------------------------------------
def crop_sides(img, crops):
	return img[crops[0][0]:crops[0][1], crops[1][0]:crops[1][1]]

def pad_image_3d(resized_image, padding):
	try:
		def calculate_padding(resized_object, desired_shape):
			return (
				(math.floor((desired_shape - resized_object.shape[0]) / 2), math.floor((desired_shape - resized_object.shape[0]) / 2)),
				(math.ceil((desired_shape - resized_object.shape[0]) / 2), math.ceil((desired_shape - resized_object.shape[0]) / 2)),
				(0,0)
			)
		return np.pad(resized_image, calculate_padding(resized_image, padding), mode='constant', constant_values=0)
	except ValueError as ex:
		logging.error(f"[pad_image] {ex}")

def squarify_3d(M, val):
	try:
		(a, b, c) = M.shape
		if a>b:
			padding = ((0, 0), (math.floor((a-b) / 2), math.ceil((a-b) / 2)), (0, 0))
		else:
			padding = ((math.floor((b-a) / 2), math.ceil((b-a) / 2)), (0, 0), (0, 0))
		return np.pad(M, padding, mode='constant', constant_values=val)
	except ValueError as ex:
		logging.error(f"[squarify] {ex}")  

def crop_image(img,tol=0):
	# img is 2D or 3D image data
	# tol  is tolerance
	mask = img>tol
	if img.ndim==3:
		mask = mask.all(2)
	m,n = mask.shape
	mask0,mask1 = mask.any(0),mask.any(1)
	col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
	row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
	return img[row_start:row_end,col_start:col_end]

def square_resize_pad(objects, desired_size):
	squared_objects = [squarify(rgb2gray(obj), 0) for obj in objects]
#     squared_objects = [squarify_3d(obj, 0) for obj in objects]
	
#     sizes = [x.shape[0] for x in squared_objects] # As they are all squares, I can take one size as an int
#     max_size = max(sizes)
#     rescaled_sizes = [math.ceil((size / max_size) * desired_size) for size in sizes]
	
	resized_and_padded = [
		pad_image(
			resize(squared_objects[i], (desired_size, desired_size), preserve_range=True,  anti_aliasing=True),
			desired_size
		)
		for i in range(len(squared_objects))
	]
	return np.asarray(resized_and_padded) / 255, None

def square_resize_pad_3d(objects, desired_size):
#     squared_objects = [squarify(rgb2gray(obj), 0) for obj in objects]
	squared_objects = [squarify_3d(obj, 0) for obj in objects]
	
#     sizes = [x.shape[0] for x in squared_objects] # As they are all squares, I can take one size as an int
#     max_size = max(sizes)
#     rescaled_sizes = [math.ceil((size / max_size) * desired_size) for size in sizes]
	
	resized_and_padded = [
		pad_image_3d(
			resize(squared_objects[i], (desired_size, desired_size), preserve_range=True,  anti_aliasing=True),
			desired_size
		)
		for i in range(len(squared_objects))
	]
	return np.asarray(resized_and_padded), None

def find_objects(resized_image, labeled_segmentation, lower_limit=np.NINF, upper_limit=np.Inf):
    contours = find_contours(labeled_segmentation, 0.9, fully_connected='high', positive_orientation='low')
    objects = []
    r_mask = np.zeros_like(resized_image, dtype='int8')
    for contour in contours:
        contour_mask = r_mask.copy()
        rr, cc = polygon(contour[:, 0], contour[:, 1], contour_mask.shape[:2])
        contour_mask[rr, cc, :] = 1
        final_mask = resized_image * contour_mask
        cropped_image = crop_image(final_mask)
        if cropped_image.size > lower_limit: # Bigger than 16 x 16, as that could be the smallest object, probably. Smaller than 400 x 400
            objects.append(cropped_image)
        
    return objects, [obj.size for obj in objects]


#functions----------------------------------------------------------
def threshold_image(image):
	try:
		#convert image to HSV space
		image = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_BGR2HSV)

		#threshold saturation and value
		range1 = (0, 70, 70)
		range2 = (180, 255, 255)
		mask = cv2.inRange(image, range1, range2)

		#convert mask to right format
		mask = mask.astype('float')
		mask /= 255
		return mask
	except ValueError as ex:
		logging.error(f"[threshold image]{ex}")  
def increase_mask(mask, close_kernel):
	try:
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
		filled = cv2.morphologyEx(mask, cv2.MORPH_OPEN, close_kernel)
		return filled
	except ValueError as ex:
		logging.error(f"[increase_mask] {ex}") 
		
def segmentation(original_img, segmentation):
	try:
		labeled_segmentation, labels = ndi.label(segmentation)
		original_masked = original_img * np.expand_dims(segmentation, axis=-1)
		return (original_masked, labeled_segmentation, labels)
	except ValueError as ex:
		logging.error(f"[segmentation] {ex}")        

def squarify(M, val):
	try:
		(a, b) = M.shape
		if a>b:
			padding = ((0, 0), (math.floor((a-b) / 2), math.ceil((a-b) / 2)))
		else:
			padding = ((math.floor((b-a) / 2), math.ceil((b-a) / 2)), (0, 0))
		return np.pad(M, padding, mode='constant', constant_values=val)
	except ValueError as ex:
		logging.error(f"[squarify] {ex}")  

def pad_image(resized_image, padding):
	try:
		def calculate_padding(resized_object, desired_shape):
			return (math.floor((desired_shape - resized_object.shape[0]) / 2), math.ceil((desired_shape - resized_object.shape[0]) / 2), resized_object.shape[-1])
		return np.pad(resized_image, calculate_padding(resized_image, padding), mode='constant', constant_values=0)
	except ValueError as ex:
		logging.error(f"[pad_image] {ex}")
		  
def print_results(img_dict):
	try:
		counted_classification = defaultdict(int)
		for classification in img_dict['classification']:
			counted_classification[classification] += 1
		
		return counted_classification
	except ValueError as ex:
		logging.error(f"print_result {ex}") 
		
def add_image_to_dict(img):
	try:
		return {
			'id': uuid.uuid4(), # Increment the ID
			'url': img,
			'state': "UNPROCESSED",
			'objects': [],
			'classification': [],
			**split_name(img),
		}
	except ValueError as ex:
		logging.error(f"add_image_to_dict {ex}")  

# def split_name(name):
# 	full_name = name.split("/")[-1]
# 	object_name = full_name.split("_")[-2]
# 	orientation = full_name.split("_")[-1].split(".")[0]
# 	object_identifier = full_name.split(".")[-2]
# 	return {
# 		"full_name": full_name,
# 		"object_name": object_name,
# 		"orientation": orientation,
# 		"object_identifier": f"{object_name}.{orientation}.{object_identifier}"
# 	}

def split_name(name):
    object_name = name.split("\\")[-2]
    object_identifier = name.split("\\")[-1].split(".png")[0]
    return {
        "object_name": object_name,
        "object_identifier": f"{object_name}.{object_identifier}"
    }


#endregion utils