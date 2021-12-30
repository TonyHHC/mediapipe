import cv2
import mediapipe as mp
import numpy as np

finger_angle_threshold = 160

finger_angle_thresholds = {
    'thumb' : 155,
    'index finger' : 160,
    'middle finger' : 160,
    'ring finger' : 160,
    'pinky' : 160,
}

finger_dict = {
    'thumb' : [2, 3, 4],
    'index finger' : [5, 6, 8],
    'middle finger' : [9, 10, 12],
    'ring finger' : [13, 14, 16],
    'pinky' : [17, 18, 20],
}

finger_status = {
    'thumb' : 0,
    'index finger' : 0,
    'middle finger' : 0,
    'ring finger' : 0,
    'pinky' : 0,
}

gesture_number = {
    'stone' : [0,0,0,0,0],
    'ok1' : [0,0,1,1,1],
    'ok2' : [1,0,1,1,1],
    '1' : [0,1,0,0,0],
    '2' : [0,1,1,0,0],
    '3' : [0,1,1,1,0],
    '4' : [0,1,1,1,1],
    '5' : [1,1,1,1,1],
    '6' : [1,0,0,0,1],
    '7' : [1,1,0,0,0],
    '8' : [1,1,1,0,0],
    '9' : [1,1,1,1,0]
}

def getAngle3D(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    radians = np.arccos(cosine_angle)
    
    angle = np.abs(radians*180.0/np.pi)
            
    if angle > 180.0:
        angle = 360-angle

    return angle

def getAngle2D(a, b, c):  
    radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
            
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def get_label(index, hand, results, width, height):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            
            # Process results
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))
            
            # Extract Coordinates
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                [width,height]).astype(int))
            
            output = text, coords
            
    return output

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        answer = ''
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        height, width, channels = image.shape
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # calculate every finger's angle
                vector = []
                for index, point in enumerate(hand_landmarks.landmark):
                    vector.append(np.array([point.x, point.y, point.z]))
                    #vector.append(np.array([point.x, point.y]))

                for finger_name, finger in finger_dict.items():
                    angle = getAngle3D(vector[finger[0]], vector[finger[1]], vector[finger[2]])
                    finger_status[finger_name] = 0
                    if angle >= finger_angle_thresholds[finger_name]:
                        finger_status[finger_name] = 1
                    #print('{} : {}'.format(finger_name, angle))
                   
                current_gesture = [finger_status['thumb'], finger_status['index finger'], finger_status['middle finger'], finger_status['ring finger'], finger_status['pinky']]
                for ans, gesture in gesture_number.items():
                    if gesture == current_gesture:
                        answer = ans
                    
                print(answer, current_gesture)
                

                # draw
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Render left or right detection
                if get_label(num, hand_landmarks, results, width, height):
                    text, coord = get_label(num, hand_landmarks, results, width, height)
                    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Flip the image horizontally for a selfie-view display.
        image = cv2.flip(image, 1)
        cv2.putText(image, str(answer), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()