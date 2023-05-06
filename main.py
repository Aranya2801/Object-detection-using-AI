import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('path/to/model')

# Define the class names and colors for visualization
class_names = ['person', 'car', 'bus', 'truck', 'motorcycle', 'bicycle', 'traffic light', 'stop sign', 'parking meter', 'bird', 'dog', 'cat', 'tree', 'building', 'airplane', 'boat', 'chair','table', 'book', 'phone', 'computer', 'keyboard', 'pizza', 'wine glass', 'scissors', 'toothbrush', 'hair drier', 'door', 'window', 'lamp', 'flower', 'cup', 'fork', 'spoon', 'knife']
colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 255, 255), (0, 165, 255), (255, 165, 0), (255, 255, 0), (0, 0, 0), (0, 128, 128), (128, 0, 128), (0, 128, 0), (128, 0, 0), (128, 128, 0), (0, 0, 128), (0, 255, 255), (255, 255, 0),(255, 0, 255), (0, 165, 255), (165, 0, 255), (0, 128, 255), (255, 128, 0), (255, 128, 128), (128, 255, 0), (128, 255, 128), (128, 128, 255), (0, 255, 0), (255, 0, 255), (255, 255, 0), (0, 128, 255), (0, 255, 128), (255, 128, 0), (192, 192, 192), (0, 255, 255), (255, 0, 0)]


# Define the minimum confidence threshold for object detection
min_confidence = 0.5

# Load the input video
cap = cv2.VideoCapture('path/to/video')

# Define the output video codec and dimensions
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# Define a dictionary to store the object counts
object_counts = {class_name: 0 for class_name in class_names}

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype('float32') / 255.0
    frame = np.expand_dims(frame, axis=0)

    # Perform object detection
    preds = model.predict(frame)

    # Visualize the results
    for i, pred in enumerate(preds[0]):
        if pred > min_confidence:
            class_name = class_names[i]
            confidence = pred * 100
            color = colors[i]
            label = f'{class_name}: {confidence:.2f}%'
            cv2.putText(frame, label, (10, (i+1)*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (0, i*20), (int(pred*100), (i+1)*20), color, -1)

            # Update the object counts
            object_counts[class_name] += 1

    # Write the frame to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow('Object Detection', frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# Print the object counts
print('Object Counts:')
for class_name, count in object_counts.items():
    print(f'{class_name}: {count}')

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()
