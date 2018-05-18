import numpy as np
import utils


class Dataset:

	def __init__(self, images, classes_in_images, classes, bb_in_images, grid_dim=13, num_anchors=3, anchor_dim=0):
		self.images = images
		self.classes_in_images = classes_in_images
		self.classes = classes
		self.bb_in_images = bb_in_images
		self.grid_dim = grid_dim
		self.num_anchors = num_anchors
		self.anchor_dim = anchor_dim
		self.image_height = images.shape[1]
		self.image_width = images.shape[2]


	def generate_anchors(self, x_center, y_center):
		anchors = []
		for i in range(self.num_anchors):
			w = (1 - float(i) / (2 * self.num_anchors - 2)) * self.anchor_dim
			h = (1 - float(self.num_anchors - i - 1) / (2 * self.num_anchors - 2)) * self.anchor_dim
			anchors.append((x_center, y_center, h, w))

		return anchors

	def get_anchor_index(self, xmin, ymin, xmax, ymax):
		if self.num_anchors == 1:
			return 0

		index = 0
		max_iou = -1

		x_center, y_center, _, _ = utils.convert_coordinates_limits_to_center(xmin, xmax, ymin, ymax)
		anchors = self.generate_anchors(x_center, y_center)
		i = 0
		for anchor in anchors:
			convert_anchor = utils.convert_coordinates_center_to_limits(anchor[0], anchor[1], anchor[2], anchor[3])
			iou = utils.compute_IOU((xmin, ymin, xmax, ymax), convert_anchor)
			if iou > max_iou:
				max_iou = iou
				index = i
			i+= 1

		return index

	def create_label(self, index):
		#third_dim = (1 + len(self.classes) + 4)
		third_dim = 5
		label = np.zeros((self.grid_dim,self.grid_dim,third_dim * self.num_anchors))
		bb_img = self.bb_in_images[index]
		grid_size_x = float(self.image_width) / self.grid_dim 
		grid_size_y = float(self.image_height) / self.grid_dim 

		# bb_in_images = [[(class_index, xmin, ymin, xmax, ymax), ...], [..], .. ]
		# bb_img = [(class_index, xmin, ymin, xmax, ymax), ...]


		for (class_index, xmin, ymin, xmax, ymax) in bb_img:
			x_center, y_center, height, width = utils.convert_coordinates_limits_to_center(xmin, ymin, xmax, ymax)
			anchor_index = self.get_anchor_index(xmin, ymin, xmax, ymax)
		
			i_grid = int(y_center * self.grid_dim / float(self.image_height))
			j_grid = int(x_center * self.grid_dim / float(self.image_width))
			if label[i_grid][j_grid][anchor_index*third_dim] != 1:
				label[i_grid][j_grid][anchor_index*third_dim] = 1 # There is an object
				label[i_grid][j_grid][anchor_index*third_dim + 1] = float(x_center - grid_size_x * j_grid) / grid_size_x
				label[i_grid][j_grid][anchor_index*third_dim + 2] = float(y_center - grid_size_y * i_grid) / grid_size_y
				label[i_grid][j_grid][anchor_index*third_dim + 3] = float(height) / grid_size_y
				label[i_grid][j_grid][anchor_index*third_dim + 4] = float(width) / grid_size_x
				#label[i_grid][j_grid][anchor_index*third_dim + 5 + class_index] = 1

		return label

	def prepare_data(self):
		xtrain = []
		ytrain = []
		xtest = []
		ytest = []

		for i in range(self.images.shape[0]):
			label = self.create_label(i)
			if i % 5 != 0:
				xtrain.append(self.images[i])
				ytrain.append(label)
			else:
				xtest.append(self.images[i])
				ytest.append(label)

		return (np.array(xtrain), np.array(ytrain), np.array(xtest), np.array(ytest))

'''ds = Dataset(np.zeros((10, 10, 8, 3)), None, ['DA', 'NU', 'NU STIU'], [[(1, 3, 3, 4, 4)]], 4, 1, 1, 1)
label = ds.create_label(0)
print(label[1])
'''
