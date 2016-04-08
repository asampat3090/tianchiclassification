import os
import sys
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

output_folder = './output'
image_folder = './images'
rect_folder = './box'
IMAGE_SIZE = 128
PIXEL_DEPTH = 255.0
TRAINING_SIZE = 10
TEST_SIZE = 1

def read_bounding_box(path):
	with open(path) as f:
		line = f.readline()
		box = map(int, map(float, line.split(',')))
		return box


def process_image(class_folder, image):
	image_id = image.split('.')[0]
	image_path = os.path.join(image_folder, class_folder, image)
	origin_image = cv2.imread(image_path)
	box_file_path = os.path.join(rect_folder, class_folder, image_id + '.txt')
	box = read_bounding_box(box_file_path)
	# box_img = origin_image.copy()
	cv2.rectangle(box_img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255,0, 0))
	# plt.imsave('box.png', box_img)
	crop_img = origin_image[box[1]: box[1] + box[3], box[0]: box[0] + box[2]]
	# plt.imsave('crop.png', crop_img)
	scaled_img = cv2.resize(crop_img, (IMAGE_SIZE, IMAGE_SIZE))
	# plt.imsave('scaled.png', scaled_img)
	gray_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)
	# cv2.imwrite('gray.png', gray_img)
	image_data = (gray_img.astype(float) - PIXEL_DEPTH / 2) / PIXEL_DEPTH
	# plt.imsave('final.png', image_data)
	return image_data


def load_images_for_class(class_folder):
	image_class = os.path.join(image_folder, class_folder)
	counter = 0
	training_data = np.ndarray(shape=(TRAINING_SIZE, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
	test_data = np.ndarray(shape=(TEST_SIZE, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
	for image in os.listdir(image_class):
		image_data = process_image(class_folder, image)
		if image_data.shape != (IMAGE_SIZE, IMAGE_SIZE):
			raise Exception('Unexpected image shape: %s' % str(image_data.shape))
		# print scaled_img.shape
		# print gray_img.shape
		# cv2.imwrite(output_path + '.png', gray_img)
		if counter < TRAINING_SIZE:
			training_data[counter, :, :] = image_data
		else:
			test_data[counter - TRAINING_SIZE, :, :] = image_data
		counter = counter + 1
		if counter % 500 == 0:
			print 'finish %s' % str(float(counter) / (TRAINING_SIZE + TEST_SIZE))
		if counter >= TRAINING_SIZE + TEST_SIZE:
			break
	return training_data, test_data


if __name__ == '__main__':
	for class_folder in os.listdir(image_folder):
		print "processing class %s" % class_folder
		training_data, test_data = load_images_for_class(class_folder)
		output_path = os.path.join(output_folder, class_folder)
		with open(os.path.join(output_folder, 'training', class_folder + '.pickle'), 'w') as f:
			pickle.dump(training_data, f, pickle.HIGHEST_PROTOCOL)
		with open(os.path.join(output_folder, 'test', class_folder + '.pickle'), 'w') as f:
			pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)
		print training_data.shape, test_data.shape
