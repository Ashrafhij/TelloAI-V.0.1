Created by Nir Meir, Ashraf Hijazi, Abed Massarwa

# TelloAIv0.1 - ArUco Marker Detection for Indoor Autonomous Drones

## Project Overview

This project is part of the Course of Autonomous Robotics at Ariel University. The goal is to detect ArUco markers in video frames captured by Tello Drones, estimate their 3D position and orientation, and annotate the video with this information.

The project includes:
1. Detecting ArUco markers in each frame of a video.
2. Estimating the 3D position and orientation (distance, yaw, pitch, roll) of each detected marker.
3. Writing the results to a CSV file.
4. Annotating the video frames with the detected markers and their IDs.
5. Ensuring real-time processing performance.

## Requirements

- Python 3.7+
- OpenCV (opencv-contrib-python)
- NumPy
- qrcode[pil]
- pyzbar

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/TelloAIv0.1.git
    cd TelloAIv0.1
    ```

2. **Install required Python packages:**

    ```bash
    pip install opencv-contrib-python numpy qrcode[pil] pyzbar
    ```

## Usage

### Part A: ArUco Marker Detection

1. **Prepare your video file:**
   
   Ensure you have the video file (`TelloAIv0.0_video.mp4`) in the project directory.

2. **Run the script:**

    ```bash
    python TelloAI.py
    ```

3. **Output:**

    - `aruco_detection_results.csv`: A CSV file containing the frame ID, marker ID, 2D corner points, and 3D pose (distance, yaw, pitch, roll) for each detected marker.
    - `annotated_aruco_video.mp4`: A video file with annotated frames showing the detected markers and their IDs.

Image example of the detection:
![detects ArUco markers](https://github.com/nirmeir/TelloAI-V.0.1/assets/24902621/1d89e151-d8e2-4461-9925-b8c0c71dc57b)

### Code Explanation

#### detect_aruco_codes(frame, aruco_dict, aruco_params)

This function detects ArUco markers in a given video frame and estimates their 3D position and orientation.

```python
def detect_aruco_codes(frame, aruco_dict, aruco_params):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    aruco_data = []

    if ids is not None:
        for i in range(len(ids)):
            aruco_id = ids[i][0]
            pts = corners[i][0]

            # Estimate 3D position
            aruco_2d = pts
            aruco_3d = estimate_aruco_3d_position(pts, camera_matrix, dist_coeffs)

            aruco_data.append({
                'id': aruco_id,
                '2d_points': aruco_2d,
                '3d_info': aruco_3d
            })

    return aruco_data


```

### Part B: Live Video ArUco Code Detection and Movement Command Generation

### Usage

1. **Detect ArUco codes in live video:**

    ```python
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

    ```
 Video exammple of the detection and the movement commands:
![movement_commands test](screen_record.mp4)
(https://raw.githubusercontent.com/username/repository/branch/path/to/thumbnail.jpg)](https://raw.githubusercontent.com/username/repository/branch/path/to/screen_record.mp4)


