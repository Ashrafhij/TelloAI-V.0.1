import cv2
import requests

# Replace with your IP Webcam address
ip_webcam_url = 'http://10.22.109.178:8080/video'


# Create a VideoCapture object
cap = cv2.VideoCapture(ip_webcam_url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Display the frame
    cv2.imshow('IP Webcam Stream', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close windows
cap.release()
cv2.destroyAllWindows()