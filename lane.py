import cv2
import numpy as np

cap = cv2.VideoCapture('sample/sample_21.mp4')
width = 1150
height = 720
guide_left_s = (500 - 600) / ((width/2 - 70) - (width/2 - 200))
guide_right_s = (500 - 600) / ((width/2 + 70) - (width/2 + 200))

def filter_color(image):
	# Blur the Image
	image_blur = image
	#image_blur = cv2.GaussianBlur(image,(3,3),0)
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
	#print((float(g1[0][3]) - g1[0][1]) / (float(g1[0][2]) - g1[0][0]))
	#print((float(g2[0][3]) - g2[0][1]) / (float(g2[0][2]) - g2[0][0]))

	guide_lines= [g1,g2,g3]
	
	return lines, guide_lines

def slope(line):
	if (line[2] - line[0]) == 0:
		return 0
	else:
		return (float(line[3]) - line[1]) / (float(line[2]) - line[0])

def draw_lanes(image, lines):
	last_left_s = np.array([])
	last_right_s = np.array([])
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
				cv2.line(image,(coords[0],coords[1]),(coords[2], coords[3]), [255,0,0],3)
				if coords[0] < (width / 2) or coords[2] < (width / 2):
					last_left_s = np.append(last_left_s, s)
			elif 0.5 < s < 1:
				cv2.line(image,(coords[0],coords[1]),(coords[2], coords[3]), [0,255,0],3)
				if coords[0] > (width / 2) or coords[2] > (width / 2):
					last_right_s = np.append(last_right_s, s)
			elif s == 0:
				cv2.line(image,(coords[0],coords[1]),(coords[2], coords[3]), [0,0,255],3)
	except:
		pass

	mean_left_s = np.mean(last_left_s)
	mean_right_s = np.mean(last_right_s)

	if mean_left_s > guide_left_s and mean_right_s > guide_right_s:
		print("Turning Right")
	elif mean_left_s < guide_left_s and mean_right_s < guide_right_s:
		print("Turning Left")
	elif mean_left_s > guide_left_s and mean_right_s < guide_right_s:
		print("Straight")




while(True):
	_, frame = cap.read()
	roi_image = roi(frame)
	colored_image = filter_color(frame)
	processed_image = process_image(colored_image)
	lines, guide_lines = find_lines(processed_image)
	draw_lanes(frame,lines)
	#draw_lanes(frame,guide_lines)

	#cv2.imshow('ROI', roi_image)
	#cv2.imshow('colored_image', colored_image)
	#cv2.imshow('Processed', processed_image)
	cv2.imshow('frame', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



cap.release()
cv2.destroyAllWindows()