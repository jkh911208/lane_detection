# References:
# https://github.com/MehdiSv/FindLanes
# https://www.youtube.com/playlist?list=PLQVvvaa0QuDdttJXlLtAJxJetJcqmqlQq
#
# Gyuhyong Jeon (James)
#

import cv2
import numpy as np
from numpy.polynomial import Polynomial as P


cap = cv2.VideoCapture('sample/sample_0.mp4')
cap.set(cv2.CAP_PROP_FPS, 15)
width = 1150
height = 720
guide_left_s = (500 - 600) / ((width/2 - 70) - (width/2 - 200))
guide_right_s = (500 - 600) / ((width/2 + 70) - (width/2 + 200))
PREV_LEFT_X1 = None
PREV_LEFT_X2 = None
PREV_RIGHT_X1 = None
PREV_RIGHT_X2 = None

def filter_color(image):
	# Blur the Image
	image_blur = image
	image_blur = cv2.GaussianBlur(image,(3,3),0)
	#return image_blur

	gray_min = np.array([100, 100, 100], dtype=np.uint8)
	gray_max = np.array([255, 255, 255], dtype=np.uint8)
	gray_mask = cv2.inRange(image_blur, gray_min, gray_max)

	yellow_min = np.array([0, 90, 115], dtype=np.uint8)
	yellow_max = np.array([50, 150, 140], dtype=np.uint8)
	yellow_mask = cv2.inRange(image_blur, yellow_min, yellow_max)

	filtered_image = cv2.bitwise_and(image_blur, image_blur, mask=cv2.bitwise_or(yellow_mask, gray_mask))
	return filtered_image

def process_image(image):
	# Edge detection
	processed_image = cv2.Canny(image,30,100)

	# region of interest
	processed_image = roi(processed_image)

	return processed_image

def roi(image):
	mask = np.zeros_like(image)
	region = np.array([[0.15*width,1.0*height],[0.85*width, 1.0*height],[0.55*width,0.65*height],[0.45*width,0.65*height]],dtype=np.int32)
	#region = np.array([[0.15*width,1.0*height],[0.3*width,1.0*height],[0.5*width,0.7*height],[0.7*width,1.0*height],[0.85*width, 1.0*height],[0.55*width,0.55*height],[0.45*width,0.55*height]],dtype=np.int32)
	try:
		cv2.fillPoly(mask, [region], 255)
	except:
		return image;
	masked = cv2.bitwise_and(image, mask) 
	return masked

def find_lines(image):
	lines = cv2.HoughLinesP(image, 1, np.pi/90, 10, np.array([]), 10, 50)

	g1 = np.array([[width/2 - 70,500,width/2 - 200,600]], dtype=np.int32)
	g2 = np.array([[width/2 + 70,500,width/2 + 200,600]], dtype=np.int32)
	g3 = np.array([[width/2,0,width/2,720]], dtype=np.int32)

	guide_lines= [g1,g2,g3]
	
	return lines, guide_lines

def slope(line):
	if (line[2] - line[0]) == 0:
		return 0
	else:
		return (float(line[3]) - line[1]) / (float(line[2]) - line[0])

def draw_lanes(image, lines):
	current_left_s = np.array([])
	current_right_s = np.array([])
	global PREV_LEFT_X1, PREV_LEFT_X2, PREV_RIGHT_X1, PREV_RIGHT_X2
	left_x = np.array([])
	left_y = []
	right_x = []
	right_y = []
	try:
		for line in lines:
			coords = line[0]
			s = slope(coords)

			if -0.3 < s < 0.3:
				continue
			if coords[0] < (width/2) < coords[2]:
				continue
			if coords[2] < (width/2) < coords[0]:
				continue

			if (guide_left_s * 1.4) < s < (guide_left_s * 0.6):
				if coords[0] < (width / 2) and coords[2] < (width / 2):
					current_left_s = np.append(current_left_s, s)
					left_x = np.append(left_x, (coords[0], coords[2]), axis=0) 
					left_y += [coords[1], coords[3]]
			elif (guide_right_s * 0.6) < s < (guide_right_s * 1.4):
				if coords[0] > (width / 2) and coords[2] > (width / 2):
					current_right_s = np.append(current_right_s, s)
					right_x += [coords[0], coords[2]]
					right_y += [coords[1], coords[3]]
	except:
		pass

	y1 = height
	y2 = height / 2 + 120
	
	if len(left_x) <= 1 or len(right_x) <= 1:
		#if PREV_LEFT_X1 is not None:
			#cv2.line(image, (int(PREV_LEFT_X1), int(y1)), (int(PREV_LEFT_X2), int(y2)), [255,0,0],3)
			#cv2.line(image, (int(PREV_RIGHT_X1), int(y1)), (int(PREV_RIGHT_X2), int(y2)), [0,255,0],3)
		return

	left_poly = P.fit(np.array(left_x), np.array(left_y), 1)
	right_poly = P.fit(np.array(right_x), np.array(right_y), 1)

	left_x1 = (left_poly - y1).roots()
	right_x1 = (right_poly - y1).roots()

	left_x2 = (left_poly - y2).roots()
	right_x2 = (right_poly - y2).roots()

	if PREV_LEFT_X1 is not None:
		left_x1 = PREV_LEFT_X1 * 0.7 + left_x1 * 0.3
		left_x2 = PREV_LEFT_X2 * 0.7 + left_x2 * 0.3
		right_x1 = PREV_RIGHT_X1 * 0.7 + right_x1 * 0.3
		right_x2 = PREV_RIGHT_X2 * 0.7 + right_x2 * 0.3

	PREV_LEFT_X1 = left_x1
	PREV_LEFT_X2 = left_x2
	PREV_RIGHT_X1 = right_x1
	PREV_RIGHT_X2 = right_x2

	cv2.line(image, (int(left_x1), int(y1)), (int(left_x2), int(y2)), [255,0,0],3)
	cv2.line(image, (int(right_x1), int(y1)), (int(right_x2), int(y2)), [0,255,0],3)

	mean_left_s = np.mean(current_left_s)
	mean_right_s = np.mean(current_right_s)

	if mean_left_s > guide_left_s * 1.3 and mean_right_s < guide_right_s * 1.3:
		cv2.putText(image,'Straight',(10,700), cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
	elif mean_left_s > guide_left_s * 1.3 and mean_right_s > guide_right_s * 1.3:
		cv2.putText(image,'Turning Right',(10,700), cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
	elif mean_left_s < guide_left_s * 1.3 and mean_right_s < guide_right_s * 1.3:
		cv2.putText(image,'Turning Left',(10,700), cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
	

while(True):
	_, frame = cap.read()
	roi_image = roi(frame)
	colored_image = filter_color(frame)
	processed_image = process_image(colored_image)
	lines, guide_lines = find_lines(processed_image)
	draw_lanes(frame,lines)
	#draw_guide_lanes(frame,guide_lines)

	#cv2.imshow('ROI', roi_image)
	#cv2.imshow('colored_image', colored_image)
	#cv2.imshow('Processed', processed_image)
	cv2.imshow('frame', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



cap.release()
cv2.destroyAllWindows()