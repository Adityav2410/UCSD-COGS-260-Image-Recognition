Prerequisites:

1. You need Python 2.7.x to run this script. If you are using python 3.x, then just change the print statement in the code to make it work.

2. You need to have python opencv library before running it.

Running command:

#The code returns the accuracy of the edge detector when compared against the ground truth
python evaluate.py OUTPUT_FILE_PATH GROUND_TRUTH_PATH

where:
OUTPUT_FILE_PATH: Path of the image containing the edges obtained using edge detector
GROUND_TRUTH_PATH: Path of the corresponding ground truth image (present in the folder data/ground_truth)
