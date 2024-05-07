import cv2
import mediapipe as mp
import numpy as np

# Global variable for knuckle allocation per finger
FINGER_KNUCKLES = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20]
}

# Function to draw landmarks on the hand including lines between knuckles
def draw_landmarks(image, hand_landmarks, straight_fingers):
    finger_number = 0
    for finger_name, finger_knuckles in FINGER_KNUCKLES.items():
        finger_landmarks = [hand_landmarks[i] for i in finger_knuckles]
        for i, landmark in enumerate(finger_landmarks):
            x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(image, str(finger_number), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            finger_number += 1

        if finger_name in straight_fingers:
            finger_knuckle_points = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in finger_landmarks]
            cv2.polylines(image, [np.array(finger_knuckle_points, dtype=int)], isClosed=False, color=(0, 0, 255), thickness=2)

# Function to calculate the direction vector between two landmarks
def calculate_direction(landmark1, landmark2):
    x1, y1 = landmark1.x, landmark1.y
    x2, y2 = landmark2.x, landmark2.y
    dx, dy = x2 - x1, y2 - y1
    return dx, dy

# Function to determine if a finger is straight
def is_straight_finger(finger_landmarks):
    # Calculate the direction vectors between adjacent knuckles
    directions = [calculate_direction(finger_landmarks[i], finger_landmarks[i+1]) for i in range(len(finger_landmarks)-1)]
    
    # Check if the directions are roughly aligned and each knuckle is further from the base knuckle
    threshold = 0.9  # Adjust this threshold as needed
    aligned = all(abs(np.dot(directions[i], directions[i+1])) / (np.linalg.norm(directions[i]) * np.linalg.norm(directions[i+1])) > threshold and
                  np.dot(directions[i], directions[i+1]) > 0 for i in range(len(directions)-1))
    
    # Calculate the base distance between the bottom two knuckles
    base_distance = np.linalg.norm(calculate_direction(finger_landmarks[0], finger_landmarks[1]))
    
    # Check if the distance between any other pair of consecutive knuckles is less than 20% of the base distance
    for i in range(1, len(finger_landmarks) - 1):
        knuckle_distance = np.linalg.norm(calculate_direction(finger_landmarks[i], finger_landmarks[i + 1]))
        if knuckle_distance < 0.2 * base_distance:
            return False
    
    return aligned


# Main function for hand tracking and finger counting
def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    # Initialize VideoCapture
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = hands.process(rgb_frame)

        # Count fingers in both hands if hands are detected
        straight_fingers = []  # Initialize list to store fingers being held up
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Count the number of straight fingers
                for finger_name, finger_knuckles in FINGER_KNUCKLES.items():
                    finger_landmarks = [hand_landmarks.landmark[i] for i in finger_knuckles]
                    if is_straight_finger(finger_landmarks):
                        straight_fingers.append(finger_name)

                # Draw landmarks and lines for fingers being held up
                draw_landmarks(frame, hand_landmarks.landmark, straight_fingers)

        # Display total finger count at the top of the frame
        cv2.putText(frame, f"Straight Fingers: {len(straight_fingers)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('Hand Tracking', frame)

        # Break the loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
