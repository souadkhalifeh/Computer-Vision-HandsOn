import cv2
import torch
from ultralytics import YOLO
import numpy as np
import time
import pygame
import threading
import os

class RealtimeYOLODetectionWithAlerts:
    def __init__(self, model_path='yolo26n.pt', target_object='person'):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(0)
        self.colors = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)
        self.target_object = target_object
        self.alert_active = False
        
        # Initialize pygame for sound
        pygame.mixer.init()
        self.sound_file = "alarm.wav"

        # Create alarm sound if it doesn't exist
        if not os.path.exists(self.sound_file):
            print("alarm.wav not found. Creating default alarm sound...")

            import wave
            import struct
            import math

            sample_rate = 44100
            duration = 1.0
            frequency = 440.0
            volume = 32767

            wav_file = wave.open(self.sound_file, 'w')
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)

            for i in range(int(sample_rate * duration)):
                value = int(volume * math.sin(2 * math.pi * frequency * i / sample_rate))
                data = struct.pack('<h', value)
                wav_file.writeframesraw(data)

            wav_file.close()

        self.alarm_sound = pygame.mixer.Sound(self.sound_file)
        
    def play_alarm(self):
        self.alarm_sound.play()
        time.sleep(2)  # Play for 2 seconds
        self.alarm_sound.stop()
        self.alert_active = False

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            start_time = time.time()
            
            # Perform inference
            results = self.model(frame)
            
            target_detected = False
            
            # Process results
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = f"{result.names[cls]} {conf:.2f}"
                    color = [int(c) for c in self.colors[cls]]
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    if result.names[cls].lower() == self.target_object.lower():
                        target_detected = True
                        # Highlight the target object
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

            # Trigger alert if target object is detected
            if target_detected and not self.alert_active:
                self.alert_active = True
                threading.Thread(target=self.play_alarm).start()
                
                # Visual alert
                cv2.putText(frame, f"{self.target_object.upper()} DETECTED!", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Calculate and display FPS
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow('Real-time YOLO Detection with Alerts', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

# Usage
detector = RealtimeYOLODetectionWithAlerts(target_object='person')
detector.run()