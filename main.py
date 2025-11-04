import cv2 as cv
import dlib
import numpy as np
import math
import mediapipe as mp

def cascade_ROI(frame: np.ndarray, face_detector: cv.CascadeClassifier):
    """ Function that extracts a Region of Interest(ROI) around the face using a Haar Cascade Classifier """
    gray_frame = cv.cvtColor(frame, code=cv.COLOR_BGR2GRAY)  # Converting frame to grayscale
    face_instances = face_detector.detectMultiScale(image=gray_frame, scaleFactor=1.5, minNeighbors=3)  # Face detection
    
    if len(face_instances) == 0:  # No face detected
        return False, None, (None, None), (None, None)
    elif len(face_instances) == 1:  # Only 1 face detected
        x, y, w, h = face_instances[0]  
    else:
        # Extracting the biggest face
        widths = [face_instances[idx][2] for idx in range(len(face_instances))]
        face_idx = widths.index(max(widths))
        x, y, w, h = face_instances[face_idx]
    roi = frame[y:y+h, x:x+w]
    return True, roi, (x, y), (x+w, y+h)

def template_ROI(frame: np.ndarray, template: np.ndarray):
    """ Function that extracts a Region of Interest(ROI) around the face using template matching."""
    res = cv.matchTemplate(image=frame, templ=template, method=cv.TM_CCORR_NORMED)
    _, max_val, _, (x1, y1) = cv.minMaxLoc(res)
    x2 = x1 + ROI.shape[1]
    y2 = y1 + ROI.shape[0]
    roi = frame[y1:y2, x1:x2]
    return roi, (x1, y1), max_val

def get_facial_landmarks(gray_ROI: np.ndarray, landmark_predictor: dlib.shape_predictor) -> np.ndarray:
    """ Function that extracts facial landmark points from a Region of Interest, and storing them into a Numpy ndarray. """
    rect = dlib.rectangle(0, 0, gray_ROI.shape[1] - 1, gray_ROI.shape[0] - 1)
    face_shape = landmark_predictor(gray_ROI, rect)

    landmarks = []
    for point_idx in range(face_shape.num_parts):
        landmark = face_shape.part(point_idx)
        vec = np.array([landmark.x, landmark.y])
        landmarks.append(vec)
    landmarks = np.stack(landmarks, axis=0)
    return landmarks

def get_face_features(facial_landmarks: np.ndarray, roi_origin: tuple[int, int]):
    """ Function that extracts centroid and size of the face using the facial landmark array. """
    origin = np.array(roi_origin)
    relative_centroid = np.average(facial_landmarks, axis=0)

    euc_distances = np.linalg.norm(facial_landmarks-relative_centroid)
    size_scale = np.average(euc_distances)
    centroid = relative_centroid + origin

    return (int(centroid[0].item()), int(centroid[1].item())), size_scale.item()

def compute_hand_centroids(landmarks, id_dict: dict, w: int, h: int) -> np.ndarray:
    """ Function that computes the centroid of the palm and wrist from hand landmarks. """
    
    # Palm centroid computation
    palm_indices = id_dict['Palm']
    palm_array = []
    # Loop over the landmarks
    for idx in palm_indices:
        lm = landmarks[idx]
        # Convert to pixel coordinates
        x = lm.x * w
        y = lm.y * h
        # Store pixel coordinates
        palm_array.append(np.array([x, y]))
    
    # Convert the list of arrays to single array
    palm_array = np.stack(palm_array, axis=0)
    
    # Palm centroid is at the average of the coordinates
    palm_centroid = np.mean(palm_array, axis=0)

    # Writs is a single landmark, which is converted to a vector
    wrist_x, wrist_y = landmarks[id_dict['Wrist']].x*w, landmarks[id_dict['Wrist']].y*h
    wrist_centroid = np.array([wrist_x, wrist_y])
    return palm_centroid, wrist_centroid

def plot_finger(frame, points: list[np.ndarray]):
    """ Function that draws lines to connect the points. """
    # Loop over each connection (Base -> Lower middle, Lower middle -> Upper middle, etc.)
    for index in range(1, len(points)):
        # Extract coordinates
        x1, y1 = int(points[index-1][0]), int(points[index-1][1])
        x2, y2 = int(points[index][0]), int(points[index][1])
        # Draw blue line
        cv.line(frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

def is_finger_extended(palm_point, base_point, lower_middle_point, upper_middle_point, tip_point) -> bool:
    """ Function that determines whether a finger is extended or not. """
    # Computing euclidian distance between palm centroid and each landmark.
    base_distance = np.linalg.norm(base_point - palm_point)
    lower_middle_distance = np.linalg.norm(lower_middle_point - palm_point)
    upper_middle_distance = np.linalg.norm(upper_middle_point - palm_point)
    tip_distance = np.linalg.norm(tip_point - palm_point)

    # If the euclidian distance increases all the way from base to tip, then the finger is extended
    is_extended = base_distance < lower_middle_distance < upper_middle_distance < tip_distance
    return is_extended

def overlay_image(overlay_dict: dict[str, dict[str]], overlay_id: str, frame: np.ndarray, face_centroid: tuple[int, int], face_scale: float):
    """ Function that overlays a PNG image over the frame, based on its Alpha channel """
    overlay = overlay_dict[overlay_id]  # Extract the overlay
    h, w, _ = overlay['Frame'].shape # Get its width and height
    
    C = np.array([   # Coordinate transfer (to center of the overlay)
        [1, 0, -w//2],
        [0, 1, -h//2],
        [0, 0, 1]])
    
    # Scaling the overlay to match face size
    scale_factor = (overlay['Scale']*face_scale) / w 
    S = np.array([  # Scaling
        [scale_factor, 0, 0],
        [0, scale_factor, 0],
        [0, 0, 1]])
    
    T = np.array([  # Translation to face centroid
        [1, 0, face_centroid[0] + overlay['Position'][0]*face_scale],
        [0, 1, face_centroid[1] + overlay['Position'][1]*face_scale],
        [0, 0, 1]])
    
    M = T  @ S @ C  # Combine to form one transformation array
    M = M[:2, :]  # Only the Affine part
    
    # 'Warping' here is simply warping it into place in the frame
    warped_overlay = cv.warpAffine(src=overlay['Frame'], M=M, dsize=(frame.shape[1], frame.shape[0]),
                                   flags=cv.INTER_NEAREST)  # The Nearest interpolation is needed to not polute the alpha channel and keep it binary.
    
    binary_mask = (warped_overlay[:, :, 3] > 0).astype(np.uint8) # Use alpha channel as binary map.
    binary_mask_3ch = cv.cvtColor(binary_mask, cv.COLOR_GRAY2BGR)  # Convert to three channels
    inverse_mask_3ch = 1 - binary_mask_3ch  # Invert the mask for the other pixels
    
    frame_overlay = np.multiply(warped_overlay[:, :, :3], binary_mask_3ch)  # Apply binary mask to the overlay
    frame_rest = np.multiply(frame, inverse_mask_3ch)  # Apply binary mask to other pixels

    overlayed_frame = frame_rest + frame_overlay  # Add the two parts together
    return overlayed_frame

def rotate_head(rotation: float, frame: np.ndarray, face_centroid: tuple[int, int], axis_scales: tuple[float, float], face_rotation: float, face_scale: float):
    """ Function that rotates the head in place"""
    
    # Creating a binary image of the original face location
    unwarped_binary = np.zeros_like(frame)  # Initiating array
    cv.ellipse(unwarped_binary, center=face_centroid,  # Drawing an ellipse of 255s around the face location
               axes=(int(axis_scales[0]*face_scale), int(axis_scales[1]*face_scale)),
               angle=math.degrees(face_rotation), startAngle=0, endAngle=360, color=(255, 255, 255), thickness=-1)
    
    masked_frame = np.multiply(frame, unwarped_binary // 255)

    C = np.array([  # Coordinate transfer matrix, shifts the coordinates of the frame to the face centroid
        [1, 0, -face_centroid[0]],
        [0, 1, -face_centroid[1]],
        [0, 0, 1]])
    
    S = np.array([  # Scaling matrix, scales the face in x and y direction
        [1.5, 0, 0],
        [0, 1.2, 0],
        [0, 0, 1]])
    
    req_rotation = rotation-face_rotation
    R = np.array([
        [math.cos(req_rotation), -math.sin(req_rotation), 0],
        [math.sin(req_rotation), math.cos(req_rotation), 0],
        [0, 0, 1]])
    
    reversed_C = np.array([  # Shifts the origin back to the top left corner
        [1, 0, face_centroid[0]],
        [0, 1, face_centroid[1]],
        [0, 0, 1]])

    M = reversed_C @ R @ S @ C
    M = M[:2, :]  # Only use Affine part
    
    # We warp the frame, and the binary mask
    warped_frame = cv.warpAffine(masked_frame, M=M, dsize=(frame.shape[1], frame.shape[0]))
    warped_binary = cv.warpAffine(unwarped_binary, M=M, dsize=(frame.shape[1], frame.shape[0]))

    background_binary = cv.bitwise_not(warped_binary)  # Background frame are all the pixels not in the warped part
    background_frame = np.multiply(frame, background_binary // 255)  # Apply binary map to background

    frame = background_frame + warped_frame # Add parts together
    return frame


stream = cv.VideoCapture(0)  # Get video feed from webcam

# Initializing classifiers and predictors
faceDetector = cv.CascadeClassifier(r'classifier_files\haarcascade_frontalface_default.xml')
landmarkPredictor = dlib.shape_predictor(r'classifier_files\shape_predictor_5_face_landmarks.dat')
mediaipeHands = mp.solutions.hands
handPredictor = mediaipeHands.Hands(max_num_hands=1, model_complexity=1, min_detection_confidence=0.8, min_tracking_confidence=0.5)

# Variables needed for loop logic
faceDetected = False  # Flag if a face is detected
faceCentroid = None  # Initializing variable, such that we can always check if it contains something
ROI = None # Initializing variable, such that we can always check if it contains something

# Constants
tSinceCascade = 0  # Number of timesteps since face is identified with Haar Cascade Clasiffier
tMaxSinceCascade = 20  # Maximum number of timesteps template matching is used instead of Haar Classifier
correlationThreshold = 0.92  # Minimum template match correlation to be detected as a face

# Scale factors for the face ellipse
xScaleFactor = 1.2  
yScaleFactor = 1.7

# Landmark IDs associated with the mediapipe hand landmark prediction
handLandmarkIDs = {
    'Wrist': 0,
    'Palm': [0, 1, 5, 9, 13, 17],
    'Fingers': {
        0: [1, 2, 3, 4],
        1: [5, 6, 7, 8],
        2: [9, 10, 11, 12],
        3: [13, 14, 15, 16],
        4: [17, 18, 19, 20]}}

# Images for overlays
overlayImages = {
    'Moustache': {
        'Frame': cv.imread(r'filter_overlays\moustache.png', cv.IMREAD_UNCHANGED),
        'Position': (0, 0.6),  # Relative position to the face centroid in size scale unit
        'Scale': 2},  # Width of the overlay image in size scale unit
    
    'Black Hat': {
        'Frame': cv.imread(r'filter_overlays\detective_hat_black.png', cv.IMREAD_UNCHANGED),
        'Position': (0, -1.5),  # Relative position to the face centroid in size scale unit
        'Scale': 4},  # Width of the overlay image in size scale unit
    
    'Joint': {
        'Frame': cv.imread(r'filter_overlays\joint.png', cv.IMREAD_UNCHANGED),
        'Position': (0.5, 0.8),  # Relative position to the face centroid in size scale unit
        'Scale': 1.5},  # Width of the overlay image in size scale unit
    
    'Thug Glasses': {
        'Frame': cv.imread(r'filter_overlays\thug_glasses.png', cv.IMREAD_UNCHANGED),
        'Position': (0, -0.2),  # Relative position to the face centroid in size scale unit
        'Scale': 2},  # Width of the overlay image in size scale unit
        }

# Extract height and with of the webcam video
width = int(stream.get(cv.CAP_PROP_FRAME_WIDTH)) 
height = int(stream.get(cv.CAP_PROP_FRAME_HEIGHT)) 

incrementingRotation = 0  # Incrementing rotation for the passive filter
if not stream.isOpened():
    print('Stream unavailable')
    exit()

while True:
    ret, frame = stream.read() # Get new frame from webcam 
    incrementingRotation += math.pi/180  # Increment passive filter rotation
    
    if not ret:  # Checking if getting a new frame is successful
        print('Stream unavailable')
        break

    """ FACIAL LANDMARK EXTRACTION """
    if faceDetected is False:
        faceDetected, ROI, roiTL, roiBR = cascade_ROI(frame=frame, face_detector=faceDetector) # Extracting an ROI using the Haar Cascade Classifier

        if faceDetected is True:  # If there is a face detection (and thus a ROI)
            tSinceCascade = 0  # Set time since cascade to 0
            roiGray = cv.cvtColor(ROI, cv.COLOR_BGR2GRAY)  # Convert ROI to grayscale, works better for landmark detection
            landmarkArray = get_facial_landmarks(gray_ROI=roiGray, landmark_predictor=landmarkPredictor)  # Extract 5 facial landmarks with the SVM + HOG classifier
            faceCentroid, sizeScale = get_face_features(facial_landmarks=landmarkArray, roi_origin=roiTL)  # Extract face centroid and face scale from the facial landmarks

    else:  # If a face has been detected earlier using Haar Cascade classifier
        tSinceCascade += 1  # Increment the time since Haar Cascade Classifier
        ROI, roiTL, corrVal = template_ROI(frame=frame, template=ROI)  # Use old ROI as a template for template matching to get updated ROI
        
        # This function will always return an ROI, however we do need to check its quality
        if tSinceCascade > tMaxSinceCascade or corrVal < correlationThreshold:
            faceDetected = False  # Set face detection flag to false again, which will switch the algorithm back to Haar Cascade Classification
        else:
            roiGray = cv.cvtColor(ROI, cv.COLOR_BGR2GRAY) # Convert ROI to grayscale, works better for landmark detection
            landmarkArray = get_facial_landmarks(gray_ROI=roiGray, landmark_predictor=landmarkPredictor) # Extract 5 facial landmarks with the SVM + HOG classifier
            faceCentroid, sizeScale = get_face_features(facial_landmarks=landmarkArray, roi_origin=roiTL) # Extract face centroid and face scale from the facial landmarks

    """ HAND POSITION EXTRACTION """
    fingersUp = 0  # Variable for storing how many fingers are held up

    # Making a prediction using the SVM + HOG hand landmark classifier
    results = handPredictor.process(frame) 
    handDetections = results.multi_hand_landmarks

    if handDetections:  # If a hand is detected
        singleHand = handDetections[0]  # Extract the first one
        handLandmarks = singleHand.landmark  # Extract the landmarks from that detection
        
        palmCentroid, wristCentroid = compute_hand_centroids(landmarks=handLandmarks, id_dict=handLandmarkIDs, w=width, h=height)  # Extract palm and wrist centroids
        
        # Computing hand rotation based on palm and wrist centroids
        dY = float(palmCentroid[1]) - float(wristCentroid[1])
        dX = float(palmCentroid[0]) - float(wristCentroid[0])
        handRotation = math.atan2(dX, -dY)

        # Looping over the fingers
        for fingerIndex, landmarkIDs in handLandmarkIDs['Fingers'].items():
            baseID, lowerMiddleID, upperMiddleID, tipID = landmarkIDs  # Getting the individual landmark IDs for the finger.
            
            # Extracting all the asssociated points of these IDs
            basePoint = np.array([handLandmarks[baseID].x*width, handLandmarks[baseID].y*height])
            lowerMiddlePoint = np.array([handLandmarks[lowerMiddleID].x*width, handLandmarks[lowerMiddleID].y*height])
            upperMiddlePoint = np.array([handLandmarks[upperMiddleID].x*width, handLandmarks[upperMiddleID].y*height])
            tipPoint = np.array([handLandmarks[tipID].x*width, handLandmarks[tipID].y*height])

            # Plot the finger
            plot_finger(frame=frame, points=[wristCentroid, basePoint, lowerMiddlePoint, upperMiddlePoint, tipPoint])

            # Checking if the finger is extended
            if is_finger_extended(palm_point=palmCentroid, base_point=basePoint, lower_middle_point=lowerMiddlePoint, upper_middle_point=upperMiddlePoint, tip_point=tipPoint):
                fingersUp += 1

    """ FILTER LOGIC """
    if faceCentroid:  # We can only interact with the filter if the face centroid is computed
        if fingersUp == 0:
            frame = rotate_head(rotation=incrementingRotation, frame=frame, face_centroid=faceCentroid, axis_scales=(xScaleFactor, yScaleFactor), face_rotation=0, face_scale=sizeScale)
        elif fingersUp == 1:
            frame = overlay_image(overlay_dict=overlayImages, overlay_id='Moustache', frame=frame, face_centroid=faceCentroid, face_scale=sizeScale)
        elif fingersUp == 2:
            frame = overlay_image(overlay_dict=overlayImages, overlay_id='Moustache', frame=frame, face_centroid=faceCentroid, face_scale=sizeScale)
            frame = overlay_image(overlay_dict=overlayImages, overlay_id='Black Hat', frame=frame, face_centroid=faceCentroid, face_scale=sizeScale)
        elif fingersUp == 3:
            frame = overlay_image(overlay_dict=overlayImages, overlay_id='Joint', frame=frame, face_centroid=faceCentroid, face_scale=sizeScale)
            frame = overlay_image(overlay_dict=overlayImages, overlay_id='Thug Glasses', frame=frame, face_centroid=faceCentroid, face_scale=sizeScale)
        elif fingersUp == 4:
            pass
        else:
            frame = rotate_head(rotation=handRotation, frame=frame, face_centroid=faceCentroid, axis_scales=(xScaleFactor, yScaleFactor), face_rotation=0, face_scale=sizeScale)
            
    cv.imshow('Image Processing and Computer Vision Visual Effects', frame)

    if cv.waitKey(1) == ord('q'):
        break

stream.release()
cv.destroyAllWindows()