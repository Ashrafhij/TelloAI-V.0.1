import cv2
import numpy as np


def detect_aruco_codes(frame, aruco_dict, aruco_params):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create ArucoDetector object
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray)
    aruco_data = []

    if ids is not None:
        for i, aruco_id in enumerate(ids):
            pts = corners[i][0]  # corners[i] returns the corners of the i-th marker
            area = cv2.contourArea(pts)
            aruco_data.append(
                {
                    "id": aruco_id,
                    "points": pts.tolist(),  # Convert to list for dictionary
                    "area": area,
                    "center": np.mean(pts, axis=0),  # Calculate center of the marker
                }
            )

            # Draw ArUco marker and ID
            cv2.polylines(frame, [pts.astype(int)], True, (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID: {aruco_id}",
                tuple(pts[0].astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

    return aruco_data


# Function to calculate movements to center the marker and return message when centered
def calculate_movement(frame_center, marker_center):
    delta_x = marker_center[0] - frame_center[0]
    delta_y = marker_center[1] - frame_center[1]

    commands = []

    if abs(delta_x) > 10:  # Adjust this threshold as needed
        if delta_x > 0:
            commands.append("right")
        else:
            commands.append("left")

    if abs(delta_y) > 10:  # Adjust this threshold as needed
        if delta_y > 0:
            commands.append("down")
        else:
            commands.append("up")

    if abs(delta_x) <= 10 and abs(delta_y) <= 10:
        commands.append("centered")  # Add centered message

    return commands


# Load the target frame
target_frame = cv2.imread("target_frame.png")

# Ensure aruco module is available
if not hasattr(cv2, "aruco"):
    raise ImportError(
        "OpenCV does not have the 'aruco' module. Make sure you have the correct version of OpenCV installed."
    )

# ArUco dictionary and parameters
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# Detect target ArUco codes
target_arucos = detect_aruco_codes(target_frame, dictionary, parameters)

# Use DroidCam as the video capture device
cap = cv2.VideoCapture('http://192.168.35.149:4747/video')  # Replace with your DroidCam URL

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Get frame dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame dimensions.")
    exit()

frame_height, frame_width, _ = frame.shape
frame_center = (frame_width // 2, frame_height // 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_arucos = detect_aruco_codes(frame, dictionary, parameters)

    # Display the commands on the frame
    if current_arucos:
        for aruco in current_arucos:
            marker_center = aruco["center"]
            commands = calculate_movement(frame_center, marker_center)
            for i, command in enumerate(commands):
                if command == "centered":
                    cv2.putText(
                        frame,
                        "Centered",
                        (10, 30 * (i + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                else:
                    cv2.putText(
                        frame,
                        command,
                        (10, 30 * (i + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
    else:
        cv2.putText(
            frame,
            "No ArUco code detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    # Show the frame
    cv2.imshow("Live Video", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
