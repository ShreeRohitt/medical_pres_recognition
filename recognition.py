import cv2
import pytesseract
from pytesseract import Output
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Pre-trained EAST text detector model
net = cv2.dnn.readNet("frozen_east_text_detection.pb")

def detect_text(image):
    orig = image.copy()
    (H, W) = image.shape[:2]

    # Define the output layer names for the EAST detector model
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # Construct a blob from the image and perform a forward pass
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # Decode the predictions
    (rects, confidences) = decode_predictions(scores, geometry)
    
    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    return boxes

def recognize_text(image, boxes):
    results = []

    for (startX, startY, endX, endY) in boxes:
        # Extract the region of interest (ROI)
        roi = image[startY:endY, startX:endX]

        # OCR the ROI using Tesseract
        config = ("-l eng --oem 1 --psm 7")
        text = pytesseract.image_to_string(roi, config=config)

        # Append the bounding box coordinates and OCR'd text
        results.append(((startX, startY, endX, endY), text))

    return results

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to increase processing speed
    frame_resized = cv2.resize(frame, (640, 480))

    # Detect text in the frame
    boxes = detect_text(frame_resized)

    # Recognize text in the detected regions
    results = recognize_text(frame_resized, boxes)

    # Display the results
    for ((startX, startY, endX, endY), text) in results:
        # Draw the bounding box
        cv2.rectangle(frame_resized, (startX, startY), (endX, endY), (0, 255, 0), 2)
        # Put the recognized text
        cv2.putText(frame_resized, text, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the output frame
    cv2.imshow("Text Recognition", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
