import numpy as np 
import cv2 as cv 
import imutils

# A function that takes an image and a template (that the image should be aligned to) , 
# the maxFeatures we need to retrieve and the percentage i want based on to keep the image, 
# and returns the aligned image
def align_image(image,template, maxFeatures = 500, keepPercent= 0.2, debug = False):
    # convert the images to grayscale
    imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    templateGray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    # detect ORB keypoints and descriptors in the image and template
    orb = cv.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)

    # match the features using the Hamming distance and sort them according to their distance
    method = cv.DescriptorMatcher_BRUTEFORCE_HAMMING
    matcher = cv.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)
    matches = sorted(matches, key=lambda x: x.distance) #to keep the best matches we sort them based on their distance and then we will keep only the top matches based on the keepPercent parameter
   
    # keep only the top matches based on the keepPercent parameter
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]

    if debug:  
        matchedVis = cv.drawMatches(image, kpsA, template, kpsB, matches, None)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv.imshow("Matched Keypoints", matchedVis)
        cv.waitKey(0)

    # extract the matched keypoints' coordinates
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")

    # loop over the matches and extract the coordinates of the matched keypoints
    for i,m in enumerate(matches):
        ptsA[i] = kpsA[m.queryIdx].pt #queryIdx is the index of the keypoint in the image and pt is the coordinates of the keypoint
        ptsB[i] = kpsB[m.trainIdx].pt #trainIdx is the index of the keypoint in the template

    # compute the homography matrix using RANSAC algorithm
    (H, mask) = cv.findHomography(ptsA, ptsB, method=cv.RANSAC)

    # use the homography matrix to warp the input image to align with the template
    (h,w) = template.shape[:2] #we take the height and width of the template to use it in the warping process

    aligned = cv.warpPerspective(image, H, (w,h))

    return aligned

image = cv.imread("../data/image.jpg")
template = cv.imread("../data/main.png")

print("Aligning the image to the template...")

aligned = align_image(image, template, debug=True)


aligned = imutils.resize(aligned, width=700)
template = imutils.resize(template, width=700)

stacked = np.hstack([template, aligned])
# create an overlay of the aligned image on top of the template to visualize the alignment
overlay = template.copy()
output = aligned.copy()
cv.addWeighted(overlay, 0.5, output, 0.5,0, output)
cv.imshow("Aligned Image stacked", stacked)
cv.imshow("Aligned Image Overlay", output)
cv.waitKey(0)