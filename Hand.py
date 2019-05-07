import cv2 as cv
import time
import numpy as np
import argparse
global frameHeight
global frameWidth
global inWidth
global inHeight
global BODY_PARTS

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                   "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }
def main(args):
	
	POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                   ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                   ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                   ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
	frameHeight =0
	frameWidth =0
	frame = cv.imread(args.input)
	frameWidth = frame.shape[1]
	frameHeight = frame.shape[0]	
	inWidth=368
	inHeight=368
	proto="pose_deploy_linevec.prototxt"
	model="pose_iter_440000.caffemodel"
	inp = cv.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
	net = cv.dnn.readNetFromCaffe(proto,model)
	net.setInput(inp)

	output = net.forward() #4-D Matrix
	Keypoints=getKeyPoints(output,frameWidth,frameHeight,frame)
	ValidateHeadTorsoHand(Keypoints,frame)
	
def getKeyPoints(output,frameWidth,frameHeight,frame):	
	H = output.shape[2]
	W = output.shape[3]
	global points
	# Empty list to store the detected keypoints
	points = []
	HeadPoints=[]
	HandPoints=[]
	for i in range(len(BODY_PARTS)-1):
		# confidence map of corresponding body's part.
		probMap = output[0, i, :, :]
			
		# Find global maxima of the probMap.
		minVal, prob, minLoc, point = cv.minMaxLoc(probMap)
		# Scale the point to fit on the original image
		x = (frameWidth * point[0]) / W
		y = (frameHeight * point[1]) / H
		#print(prob)
		if prob > 0.3 : 
			cv.circle(frame, (int(x), int(y)), 5, (0, 255, 255), thickness=-1, lineType=cv.FILLED)
			cv.putText(frame, "{}".format(i), (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, lineType=cv.LINE_AA)
			
			# Add the point to the list if the probability is greater than the threshold
			points.append((int(x), int(y)))
			#points={i : (int(x), int(y))}
		else :
			points.append(None)
			#points={i : None}
	print(points)
	return points
def ValidateHeadTorsoHand(Keypoints,frame):
	HeadTorsoHand=[]
	for i in Keypoints:
		if i is not None:
			HeadTorsoHand.append(Keypoints.index(i))
	if len(HeadTorsoHand) > 11:
		print("Valid Hand")
		cv.imshow("outputput-Keypoints",frame)
		cv.waitKey(0)
	else:
		print("InValid Hand")
	
def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', help='Path to input image.')
	# parser.add_argument('--proto', help='Path to proto image.')
	# parser.add_argument('--model', help='Path to model image.')
	return parser.parse_args()
	

if __name__ == '__main__':
    main(parse_args())
	



