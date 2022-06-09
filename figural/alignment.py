import numpy as np
import cv2

def alignImages(source, target, max_features=500, good_match_percent=.15, save_matches=False):
    ''' Image registration with opencv. 
    
    `source` image is the one that will be aligned, to target `target` image.
    
    Returns the remapped source, and the homography map.

    Base on code from https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
    '''

    if type(source) is not np.ndarray:
        source = np.array(source)
    if type(target) is not np.ndarray:
        target = np.array(target)

    # Convert images to grayscale
    if source.ndim == 3:
        source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    if target.ndim == 3:
        target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(source, None)
    keypoints2, descriptors2 = orb.detectAndCompute(target, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    matches = list(matches)
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * good_match_percent)
    matches = matches[:numGoodMatches]

    if save_matches:
        imMatches = cv2.drawMatches(source, keypoints1, target, keypoints2, matches, None)
        cv2.imwrite(save_matches, imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width = target.shape[:2]
    sourceReg = cv2.warpPerspective(source, h, (width, height))

    return sourceReg, h