import cv2
import torch
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap = cv2.VideoCapture('FordGema.mp4')

# Se obtienen las dimensiones del video y la tasa de frames por segundo (FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter('test_1.mp4', fourcc, fps, (frame_width, frame_height))

interested_classes = [0, 2, 7]  # 0: 'person', 2: 'car', 7: 'truck'

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    for det in results.xyxy[0]:
        classId = int(det[5])
        score = det[4].item()
        if classId in interested_classes and score > 0.5:  # Solo si la confianza es mayor al 65%
            box = det[:4].cpu().numpy().astype(int)
            label = 'car' if classId in [2, 7] else model.names[classId]
            label = f'{label} {score:.2f}'
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)

    cv2.imshow('YOLOv5 - Ford Explorer - Filtered Detections', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release() 
cv2.destroyAllWindows()
