import os
import numpy as np
from scipy import misc, io
import image_processing as ip
import scipy
import random
from keras.models import load_model

import CNN
import utils

image_dir = "101_ObjectCategories"
annotation_dir = "Annotations"

classes = ['panda', 'platypus', 'cougar_body', 'rhino', 'scorpion', 'kangaroo', 'flamingo', 'elephant', 'crocodile', 'dolphin',
			'ant', 'butterfly', 'crab', 'dalmatian', 'emu', 'lobster', 'llama', 'sea_horse', 'wild_cat', 'pigeon']

classes += ['underwater', 'sky']
def get_dataset(classes):
	images = []
	labels = []
	for c in classes:
		dir_path = image_dir + "/" + c
		for image_path in os.listdir(dir_path):
			filename = os.fsdecode(image_path)
			image = misc.imread(dir_path + "/" + filename)
			if len(image.shape) == 3:
				images.append(image)
				labels.append(classes.index(c))

	aux = list(zip(images, labels))
	random.shuffle(aux)
	images, labels = zip(*aux)

	return (np.array(images), np.array(labels))

def resize_images(images):
	images_res = []
	for img in images:
		img2 = ip.make_square_image_middle(img)
		img2 = ip.resize_image(img2, 104)
		images_res.append(img2)
	return images_res


def split_dataset(images, labels):
	x_train = np.concatenate((images[1:][::4], images[2:][::4], images[3:][::4]))
	y_train = np.concatenate((labels[1:][::4], labels[2:][::4], labels[3:][::4]))
	x_test = images[::4]
	y_test = labels[::4]
	return (x_train, y_train, x_test, y_test)


images, labels = get_dataset(classes)
images2 = resize_images(images)
labels = CNN.to_categorical(labels, len(classes))


x_train, y_train, x_test, y_test = split_dataset(images2, labels)

#model = CNN.create_model(x_train[0].shape)
#CNN.train_model(model, np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test), 32, 30)
#CNN.save_model(model, "saved_models", "background_animals_model")

save_dir = "saved_models"
model_name = "background_animals_model"

model_path = os.path.join(save_dir, model_name)
model = load_model(model_path)

nr_true = 0
nr_false = 0
predictions = model.predict(np.array(images2))
i = 0
for y_pred in predictions:

	max_c = max(y_pred)
	c_index = np.where(y_pred == max_c)
	if labels[i][c_index] == 1:
		nr_true += 1
	else:
		nr_false += 1
	i += 1

print(nr_true / float(len(images)))

'''for i in range(len(images)):
	img = images[i]
	y_pred = predictions[i]
	max_p = max(y_pred)
	c_index = np.where(y_pred == max_p)[0][0]
	print(c_index)
	c = classes[c_index]
	text = c + " " + str(max_p)
	img = ip.add_text_to_image(text, img)
	ip.show_image(img)'''

filename = os.fsdecode("dolphin.jpg")
image = scipy.ndimage.imread(filename, mode='RGB')
ip.show_image(image)

image2 = ip.make_square_image_middle(image)
image2 = ip.resize_image(image2, 104)

y_pred = model.predict(np.array([image2]))[0]
max_p = max(y_pred)
c_index = np.where(y_pred == max_p)[0][0]
print(c_index)
c = classes[c_index]
text = c + " " + str(max_p)
img = ip.add_text_to_image(text, image)
ip.show_image(img)


image2 = ip.make_square_image_middle(image)
image2 = ip.resize_image(image2, 315)
step = 7
scale = 1.5
crop_images = []
num_cells = int(210/(step*scale))

for i in range(num_cells):
	for j in range(num_cells):
		x_min = int(step * scale * j)
		y_min = int(step * scale* i)
		x_max = int(x_min + 104 * scale)
		y_max = int(y_min + 104 * scale)
		crop_img = image2[y_min:y_max, x_min:x_max]
		crop_img = ip.resize_image(crop_img,104)
		print(crop_img.shape)
		crop_images.append(crop_img)

for img in crop_images:
	ip.show_image(img)
	break
print(np.array(crop_images).shape)
predictions = model.predict(np.array(crop_images))

max_max_p = 0
max_x_min = 0
max_y_min = 0
max_x_max = 0
max_y_max = 0

for i in range(num_cells):
	for j in range(num_cells):
		x_min = int(step * scale * j)
		y_min = int(step * scale* i)
		x_max = int(x_min + 104 * scale)
		y_max = int(y_min + 104 * scale)
		index = num_cells * i + j
		prediction = predictions[index]
		max_p = max(prediction)
		c_index = np.where(prediction == max_p)[0][0]
		c = classes[c_index]
		
		if c == "underwater":
			max_max_p = max_p
			max_x_min = x_min
			max_y_min = y_min
			max_x_max = x_max
			max_y_max = y_max
			cx, cy, _, _ = utils.convert_coordinates_limits_to_center(x_min, y_min, x_max, y_max)
			image2 = ip.create_image_with_bb(image2, (x_min, y_min, x_max, y_max))
			print(c)
			print(max_p)
			ip.show_image(image2)


