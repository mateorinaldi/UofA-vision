import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from time import sleep
import tracking_detectors

video_path = 'videos/Georges.h264'

if video_path.split('.')[-1] == 'h264':  # If type is h264, we have to convert it to mp4
	new_video_path = video_path[:-4]+'mp4'
	os.popen(f'echo y | ffmpeg -i {video_path} {new_video_path}')  # Convert the video to mp4
	video_path = new_video_path
	print(video_path)

image_path = 'image.jpg'

'''camera parameter'''

try:
	import picamera

	camera = picamera.PiCamera()
	camera.resolution = (1296, 972)
	camera.shutter_speed = 15000
	camera.awb_mode = 'auto'
	camera.iso = 800
except:
	print('error camera')


# video_analyser("videos/orange.h264", new_orange_jacket_color_properties)


def create_empty_positions_graph(video_path):
	cap = cv2.VideoCapture(video_path)
	ret, frame = cap.read()

	height, width = frame.shape[:2]

	tab = np.zeros((height, width))

	cap.release()

	return tab


def positions_graph_filler(video_path, positions_list, circles_radius=100):
	positions_graph = create_empty_positions_graph(video_path)

	circle_array = positions_graph  # For each position, we use this array to create the area where operator is

	for position in positions_list:
		if position[0] is not None:
			circle_array.fill(0)  # To remove every '1' in this array
			cv2.circle(circle_array, (position[0], position[1]), circles_radius, 1, -1)

			positions_graph = positions_graph + circle_array

	maxi = np.max(positions_graph)

	positions_graph = positions_graph / maxi

	plt.imshow(positions_graph, cmap='jet')

	plt.colorbar()
	plt.savefig('graph.jpg')
	plt.show()

	return positions_graph


# positions_list_pixels = video_analyser(video_path, blue_references_color_properties, 25)

# positions_graph = positions_graph_filler(video_path, positions_list, circles_radius)


def find_homography_matrix(real_world_pts, image_pts):
	# Convert points to homogeneous coordinates
	real_world_pts = np.hstack((real_world_pts, np.ones((len(real_world_pts), 1))))
	image_pts = np.hstack((image_pts, np.ones((len(real_world_pts), 1))))

	# Compute the homography matrix
	homography_matrix, _ = cv2.findHomography(image_pts, real_world_pts)

	return homography_matrix


def convert_coordinates_to_reel_life(homography_matrix, coordinates_pixel):
	# Convert point to homogeneous coordinates
	point_image = np.array([[coordinates_pixel[0]], [coordinates_pixel[1]], [1]])

	# Compute the point in real-world coordinates
	point_real_world = np.dot(homography_matrix, point_image)
	point_real_world /= point_real_world[2]

	# Return the real-world coordinates in centimeter
	return np.array([int(point_real_world[0]), int(point_real_world[1])])


def positions_in_pixels_to_centimeters(homography_matrix, positions_list_pixels):
	positions_list_centimeters = np.empty((0, 2))
	for position_pixel in positions_list_pixels:
		if position_pixel[0] is None:
			position_centimeters = position_pixel
		else:
			position_centimeters = convert_coordinates_to_reel_life(homography_matrix, position_pixel)

		positions_list_centimeters = np.vstack([positions_list_centimeters, position_centimeters])

	return positions_list_centimeters


def solos_points_remover(
		positions_list):  # To remove every point which is solo/strange, probably due to an error of vision recognition
	corrected_positions_list = positions_list.copy()
	list_lenght = len(corrected_positions_list)

	booleean_tab = np.zeros(list_lenght)  # Create a boolean array to identify quickly where solo points are
	for i, elt in enumerate(positions_list):
		if elt[0] is None:  # If the hard hat hasn't been detected
			booleean_tab[i] = False
		else:  # If the hard hat has been detected
			booleean_tab[i] = True

	for ind_position_to_check in range(3, list_lenght - 3):
		count_True_in_neighborhood = 0
		for neighbor_value in booleean_tab[ind_position_to_check - 3:ind_position_to_check + 4]:  # Check if most of neighbors are True or False
			if neighbor_value:
				count_True_in_neighborhood += 1

		if count_True_in_neighborhood <= 3 and booleean_tab[
			ind_position_to_check]:  # Most neighbors are False but the position is True
			booleean_tab[ind_position_to_check] = False
			corrected_positions_list[ind_position_to_check] = [None, None]
		elif count_True_in_neighborhood >= 4 and not booleean_tab[
			ind_position_to_check]:  # Most neighbors are True but the position is False
			booleean_tab[ind_position_to_check] = True
			average = np.array([0.0, 0.0])
			for neighbor_value in corrected_positions_list[ind_position_to_check - 3:ind_position_to_check + 4]:  # We tak a look at the 6 neighbors
				if neighbor_value[0] is not None:
					average += np.array(neighbor_value) / count_True_in_neighborhood

			corrected_positions_list[ind_position_to_check] = [int(average[0]), int(average[1])]

	return corrected_positions_list


def print_positions(positions_list, xlim=None, ylim=None, xlabel=None, ylabel=None):
	# Create a graph
	fig, ax = plt.subplots()

	# For every position
	for position in positions_list:
		ax.scatter(position[0], position[1], s=5)

	# Set the x and y limits if specified
	if xlim is not None:
		ax.set_xlim(xlim)
	if ylim is not None:
		ax.set_ylim(ylim)
	if xlabel is not None:
		plt.xlabel(xlabel)
	if ylabel is not None:
		plt.ylabel(ylabel)

	# Print the result
	plt.show()


def path_separator(positions_list, minimum_number_of_points=10):
	list_of_positions_list = []

	temp_list = []

	for position in positions_list:
		if position[0] != None:  # If there is a position
			temp_list.append(position)
		elif len(temp_list) != 0:  # if this is the end of a list
			if len(temp_list) >= minimum_number_of_points:
				list_of_positions_list.append(temp_list)
			temp_list = []  # Reinitialize temp_list
	if len(temp_list) != 0:  # At the end if the sequence finish with a position
		list_of_positions_list.append(temp_list)

	return list_of_positions_list


def curve_smoother(points, window_size=5):
	# Defines a window of size window_size to smooth each coordinate
	half_window = window_size // 2

	# Initialize a list to store the smoothed points
	smoothed_points = []

	# Go through each point in the original order
	for i in range(len(points)):
		# Computes the moving average for the x coordinate
		x_values = [points[j][0] for j in range(max(0, i - half_window), min(len(points), i + half_window + 1))]
		x_smooth = sum(x_values) / len(x_values)

		# Computes the moving average for the y coordinate
		y_values = [points[j][1] for j in range(max(0, i - half_window), min(len(points), i + half_window + 1))]
		y_smooth = sum(y_values) / len(y_values)

		# Adds the smoothed point to the list of smoothed points
		smoothed_points.append((x_smooth, y_smooth))

	# Displays the origin points
	x_orig = [p[0] for p in points]
	y_orig = [p[1] for p in points]
	plt.scatter(x_orig, y_orig)

	# Displays the smoothed curve
	x_smooth = [p[0] for p in smoothed_points]
	y_smooth = [p[1] for p in smoothed_points]
	plt.plot(x_smooth, y_smooth)

	# Displays the graphic
	plt.show()

	# Returns a list of smoothed points
	return smoothed_points


def positions_cleaner(positions_list, number_of_smoothering=3):
	cleaned_positions_list = np.copy(positions_list)
	for i in range(number_of_smoothering):
		cleaned_positions_list = curve_smoother(cleaned_positions_list)

	return cleaned_positions_list


def positions_to_distance(positions):
	distance = 0
	for i in range(1, len(positions)):
		distance += math.sqrt(
			(positions[i][0] - positions[i - 1][0]) ** 2 + (positions[i][1] - positions[i - 1][1]) ** 2)
	return int(distance)


# def old_calibrate_camera():
#     image_path = input("Give the image path : ")
#     number_of_points = int(input("Give the number of points you want to give (4 minumum) : "))
#     real_world_pts = np.empty((0, 2))
#     for i in range(number_of_points):
#         x = int(input(f"Give the x coordinate of point {i+1} : "))
#         y = int(input(f"Give the y coordinate of point {i+1} : "))
#         real_world_pts = np.vstack([real_world_pts, [x,y]])

#     reference_points_coordinates_pixel = define_reference_points(image_path, number_of_points)
#     homography_matrix = find_homography_matrix(real_world_pts, reference_points_coordinates_pixel)

#     return homography_matrix


def get_real_life_coordinates():
	coordinates_calibration = open("coordinates_calibration.txt", "r")  # "r" to read
	raw_real_life_coordinates = coordinates_calibration.readlines()
	real_life_coordinates = np.empty((0, 2))
	for position in raw_real_life_coordinates:
		raw_tab = position.split(',')
		if len(raw_tab) == 2:  # If there is exactly one ','
			x = raw_tab[0].replace("\n", "")  # Remove line jumps
			x = x.replace(" ", "")  # Remove spaces
			y = raw_tab[1].replace("\n", "")  # Remove line jumps
			y = y.replace(" ", "")  # Remove spaces

			real_life_coordinates = np.vstack((real_life_coordinates, [int(x), int(y)]))

	coordinates_calibration.close()

	return real_life_coordinates


def save_homography_matrix(homography_matrix):
	np.savetxt('homography_matrix.txt', homography_matrix)


def get_homography_matrix():
	homography_matrix = np.loadtxt('homography_matrix.txt')
	return homography_matrix


def calibrate_camera():
	input("first check in the 'coordinates_calibration.txt' file that the real life coordinates are the good ones\nWhen it is the case, press enter")
	real_life_coordinates = get_real_life_coordinates()
	number_of_points = len(real_life_coordinates)
	for position_number in range(1, number_of_points + 1):
		input(
			f"Go to position {position_number} {real_life_coordinates[position_number - 1]}\nThen press enter and turn on yourself\n")
		print('wait')
		camera.start_recording(f"videos_calibration/point{position_number}.h264")
		sleep(5)  # Record for 5pseconds
		camera.stop_recording()

	reference_positions = []
	for num_point in range(1, number_of_points + 1):
		reference_positions.append(tracking_detectors.color_finder(f"videos_calibration/point{num_point}.h264"))

	average_positions_in_pixels = np.empty((0, 2))
	for position in reference_positions:
		average = [0, 0]
		number_of_points = 0
		for point in position:
			if point[0] != None:  # If the operator is detected
				average[0] += point[0]
				average[1] += point[1]
				number_of_points += 1
		average[0] /= number_of_points
		average[1] /= number_of_points

		average_positions_in_pixels = np.vstack((average_positions_in_pixels, [int(average[0]), int(average[1])]))

	homography_matrix = find_homography_matrix(real_life_coordinates, average_positions_in_pixels)
	save_homography_matrix(homography_matrix)

	return homography_matrix


global homography_matrix


real_world_pts = np.array(
	[[320, 120], [240, 120], [160, 120], [80, 120], [0, 120], [0, 0], [80, 0], [160, 0], [240, 0], [320, 0]])
image_pts = np.array(
	[[1160, 287], [988, 256], [759, 245], [512, 253], [282, 270], [210, 579], [498, 590], [822, 581], [1095, 579],
	 [1261, 573]])

# homography_matrix = find_homography_matrix(real_world_pts, image_pts)



def main_test():
	recalibrate = input("Would you like to recalibrate ? ")
	if recalibrate == "y" or recalibrate == "Y":  # If the camera is not installed the same way (change of location or orientation)
		homography_matrix = calibrate_camera()
	else:
		homography_matrix = get_homography_matrix()  # Take the default homography_matrix (will be stored in a file)

	video_path = input("Give the video path : ")

	positions = tracking_detectors.color_finder(video_path)  # Taking every positions of the operator

	positions_graph_filler(video_path, positions)  # Print the position graph

	print_positions(positions, xlim=(0, 1920), ylim=(0, 1080), xlabel="pixels", ylabel="pixels")

	positions_without_solos_points = solos_points_remover(positions)  # Remove strange points

	print_positions(positions_without_solos_points, xlim=(0, 1920), ylim=(0, 1080), xlabel="pixels", ylabel="pixels")

	positions_in_centimeter = positions_in_pixels_to_centimeters(homography_matrix, positions_without_solos_points)  # Get real life positions

	list_of_paths = path_separator(positions_in_centimeter)  # Cut parts where the operator isn't detected

	cleaned_paths = [positions_cleaner(path) for path in list_of_paths]  # Smoothing of each path

	for path in cleaned_paths:
		print_positions(path, xlim=(-200, 600), ylim=(-100, 400), xlabel="centimeters", ylabel="centimeters")
		print(positions_to_distance(path))


# main_test()
