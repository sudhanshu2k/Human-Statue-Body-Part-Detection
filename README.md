# Human-Statue-Body-Part-Detection

Getting Started
Copy python script Head.py, Torso.py, Hand.py, Body.py and Body_Video.py on your local Linux system and put it inside current working directory.
Prerequisites:
•	Python 3.5+
•	NumPy
•	OpenCV >3.3
Installation
1.Download pose_deploy_linevec.prototxt from below link and put it inside current working directory.
https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/pose/coco/pose_deploy_linevec.prototxt
2.Download pose_iter_440000.caffemodel by using below command:
wget http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel

Running the Script
1.To test Head of the Human Statue, run below command:
    python3.5 Head.py --input =input Image
2. To test Torso of the Human Statue, run below command:
    python3.5 Torso.py --input =input Image
3. To test Hand of the Human Statue, run below command:
    python3.5 Hand.py --input =input Image
4. To test complete body of the Human Statue, run below command:
    python3.5 Body.py --input =input Image
4. To test complete body of the Human Statue for video input, run below command:
     python3.5 Body_Video.py --input= (Input Video Path)






