import cv2 as cv
import time
import numpy as np
import argparse
import queue
import gc 
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
	global frameQueue_g
	frameHeight =0
	frameWidth =0
	#Body=[]
	cap = cv.VideoCapture(args.input)
	frameQueue_g = queue.Queue()
	print(frameQueue_g)
	status = True
	proto="pose_deploy_linevec.prototxt"
	model="pose_iter_440000.caffemodel"
	while status or not frameQueue_g.empty():
		status,frame = cap.read()
		if status:
			pos_frame = cap.get(cv.CAP_PROP_POS_FRAMES)
			Body=[]
			print(str(pos_frame)+ "frames")
			frameWidth = frame.shape[1]
			frameHeight = frame.shape[0]
			inWidth=368
			inHeight=368
			inp = cv.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
								(0, 0, 0), swapRB=False, crop=False)
			net = cv.dnn.readNetFromCaffe(proto,model)
			net.setInput(inp)

			output = net.forward() #4-D Matrix
			Keypoints=getKeyPoints(output,frameWidth,frameHeight,frame)
			#frame=ValidateBody(Keypoints,frame)
			
			for i in Keypoints:
				if i is not None:
					Body.append(Keypoints.index(i))
			if len(Body) > 13:
				print("Valid Body",len(Body))
				cv.imshow("outputput-Keypoints",frame)
			#cv.waitKey(1) & 0xff
				if cv.waitKey(10) & 0xFF == ord('q'):
					break
			else:
				print("InValid Body",len(Body))
		
		else:
			cap.set(cv.CAP_PROP_POS_FRAMES, pos_frame-1)
			print("frame is not ready")
				# It is better to wait for a while for the next frame to be ready
			cv.waitKey(1000)
		# cap.release()
		# cv.destroyAllWindows()
		# gc.collect()
	cap.release()
	cv.destroyAllWindows()
	gc.collect()
	return
	
def getKeyPoints(output,frameWidth,frameHeight,frame):	
	H = output.shape[2]
	W = output.shape[3]
	global points
	# Empty list to store the detected keypoints
	points = []
	HeadPoints=[]
	HandPoints=[]
	#cv.imshow('Read Video',frame)
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
			cv.circle(frame, (int(x), int(y)), 2, (0, 255, 255), thickness=-1, lineType=cv.FILLED)
			cv.putText(frame, "{}".format(i), (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, lineType=cv.LINE_AA)
			
			# Add the point to the list if the probability is greater than the threshold
			points.append((int(x), int(y)))
			#points={i : (int(x), int(y))}
		else :
			points.append(None)
			#points={i : None}
	print(points)
	return points
# def ValidateBody(Keypoints,frame):
	# Body=[]
	# for i in Keypoints:
		# if i is not None:
			# Body.append(Keypoints.index(i))
	# if len(Body) > 15:
		# print("Valid Body")
		# return frame
		# # cv.imshow("outputput-Keypoints",frame)
		# # cv2.waitKey(1) & 0xff
		
		# #cv.waitKey(0)
	# else:
		# print("InValid Body")
	# #gc.collect()
		# return
	
def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', help='Path to input Video.')
	# parser.add_argument('--proto', help='Path to proto image.')
	# parser.add_argument('--model', help='Path to model image.')
	return parser.parse_args()
	

if __name__ == '__main__':
    main(parse_args())
	



