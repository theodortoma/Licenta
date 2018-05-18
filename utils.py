import numpy as np

def compute_surface(bb):
	return (bb[2] - bb[0]) * (bb[3] - bb[1])

#(xmin, ymin, xmax, ymax)
def compute_IOU(bb1, bb2):
	xmin_int = max(bb1[0], bb2[0])
	ymin_int = max(bb1[1], bb2[1])
	xmax_int = min(bb1[2], bb2[2])
	ymax_int = min(bb1[3], bb2[3])

	if (xmax_int <= xmin_int) or (ymax_int <= ymin_int):
		return 0

	intersection = compute_surface((xmin_int, ymin_int, xmax_int, ymax_int))
	surface1 = compute_surface(bb1)
	surface2 = compute_surface(bb2)
	union = surface1 + surface2 - intersection

	return float(intersection) / union

def convert_coordinates_center_to_limits(x_center, y_center, height, width):
	xmin = x_center - width / 2.0
	ymin = y_center - height / 2.0
	xmax = x_center + width / 2.0
	ymax = y_center + height / 2.0

	return (xmin, ymin, xmax, ymax)

def convert_coordinates_limits_to_center(xmin, ymin, xmax, ymax):
	x_center = (xmax + xmin) / 2.0
	y_center = (ymax + ymin) / 2.0
	height = (ymax - ymin)
	width = (xmax - xmin)

	return (x_center, y_center, height, width)

def extract_bb_from_cnn_label(label, num_classes, img_size):
	grid_dim = label.shape[0]
	cell_size = img_size / float(grid_dim)
	third_dim = 5
	num_anchors = int(label.shape[2] / third_dim)

	print(label[7][12])
	print(label[8][7])
	print(label[8][11])

	bboxes = []
	for i in range(grid_dim):
		for j in range(grid_dim):
			for a in range(num_anchors):
				label[i][j] = label[i][j] * 10
				print(label[i][j]*10)
				if i == 8 and j == 7:
					label[i][j] /= label[i][j][0]
					x_center = j * cell_size + label[i][j][a*third_dim+1] * cell_size
					y_center = i * cell_size + label[i][j][a*third_dim+2] * cell_size
					height = label[i][j][a*third_dim+3] * cell_size
					width = label[i][j][a*third_dim+4] * cell_size

					'''pmax = 0
					c = -1
					for index in range(10):
						p = label[i][j][a*third_dim + index + 5]
						if p > pmax:
							pmax = p
							c = index'''
					bboxes.append((0, 0, x_center, y_center, height, width))

	return bboxes