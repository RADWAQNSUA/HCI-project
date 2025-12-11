import cv2
import mediapipe as mp
import numpy as np
from collections import deque

class HandTracker:
    """Enhanced MediaPipe hands wrapper with better detection and performance."""
    
    def __init__(self, config=None):
        self.config = config
        
        # MediaPipe setup with better parameters
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Allow up to 2 hands for better stability
            min_detection_confidence=0.7,  # Increased from 0.6
            min_tracking_confidence=0.7,   # Increased from 0.6
            model_complexity=1
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Enhanced smoothing buffers
        self.smoothing_buffer_size = 5  # Increased for smoother tracking
        self.landmark_buffer = deque(maxlen=self.smoothing_buffer_size)
        self.position_buffer = deque(maxlen=self.smoothing_buffer_size)
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.last_time = 0
        
        # Detection state
        self.hand_detected = False
        self.detection_confidence = 0.0
        
        # Calibration reference
        self.calibration_reference = None
        self.hand_size_reference = 0.0
        
        # Tracking stability
        self.stability_threshold = 15
        self.hand_stability_counter = 0
        
    def calibrate(self, landmarks):
        """Set calibration reference based on current hand size"""
        if landmarks and len(landmarks) >= 21:
            # Calculate hand size as distance from wrist to middle finger MCP
            wrist = landmarks[0]
            middle_mcp = landmarks[9]
            dx = wrist[1] - middle_mcp[1]
            dy = wrist[2] - middle_mcp[2]
            self.hand_size_reference = np.sqrt(dx*dx + dy*dy)
            
    def process(self, frame, draw_landmarks=True, draw_connections=True):
        """Process BGR frame with enhanced detection"""
        if frame is None or frame.size == 0:
            self.hand_detected = False
            return None
            
        # Ensure frame is valid
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            self.hand_detected = False
            return None
            
        try:
            # Resize frame for faster processing while maintaining aspect
            h, w = frame.shape[:2]
            processing_size = 640  # Optimal size for MediaPipe
            scale = processing_size / w
            new_w = processing_size
            new_h = int(h * scale)
            
            frame_resized = cv2.resize(frame, (new_w, new_h))
            
            # Convert to RGB for MediaPipe
            img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            
            # Process with MediaPipe
            self.results = self.hands.process(img_rgb)
            img_rgb.flags.writeable = True
            
            # Check if hand is detected
            self.hand_detected = bool(self.results.multi_hand_landmarks)
            
            # Scale back to original dimensions for drawing
            if draw_landmarks and self.results.multi_hand_landmarks:
                # Scale landmarks back to original frame size
                scale_x = w / new_w
                scale_y = h / new_h
                
                for hand_landmarks in self.results.multi_hand_landmarks:
                    # Scale landmarks
                    for landmark in hand_landmarks.landmark:
                        landmark.x *= scale_x
                        landmark.y *= scale_y
                    
                    # Draw on original frame
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS if draw_connections else None,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Draw larger circles on fingertips for better visibility
                    for landmark_id in [4, 8, 12, 16, 20]:  # Fingertips
                        landmark = hand_landmarks.landmark[landmark_id]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                        cv2.circle(frame, (x, y), 10, (0, 0, 0), 2)
            
            return self.results
            
        except Exception as e:
            self.hand_detected = False
            print(f"Error in hand tracking: {e}")
            return None
    
    def get_landmarks_list(self, frame, hand_index=0):
        """Get smoothed landmarks for specified hand with enhanced processing"""
        if (not hasattr(self, 'results') or not self.results or 
            not self.results.multi_hand_landmarks or 
            hand_index >= len(self.results.multi_hand_landmarks)):
            self.hand_stability_counter = 0
            return []
        
        try:
            hand = self.results.multi_hand_landmarks[hand_index]
            h, w, _ = frame.shape
            
            # Extract current landmarks
            current_landmarks = []
            for i, lm in enumerate(hand.landmark):
                # Ensure landmark coordinates are within frame bounds
                x = max(0, min(int(lm.x * w), w - 1))
                y = max(0, min(int(lm.y * h), h - 1))
                current_landmarks.append((i, x, y))
            
            # Add to buffer
            self.landmark_buffer.append(current_landmarks)
            
            # Apply weighted temporal smoothing (more weight to recent frames)
            if len(self.landmark_buffer) >= 2:
                smoothed_landmarks = []
                weights = np.linspace(0.3, 1.0, len(self.landmark_buffer))
                weights = weights / weights.sum()
                
                for i in range(21):  # 21 landmarks per hand
                    avg_x = 0
                    avg_y = 0
                    
                    for j, landmarks in enumerate(self.landmark_buffer):
                        if i < len(landmarks):
                            weight = weights[j]
                            avg_x += landmarks[i][1] * weight
                            avg_y += landmarks[i][2] * weight
                    
                    avg_x = int(avg_x)
                    avg_y = int(avg_y)
                    
                    # Apply bounds checking
                    avg_x = max(0, min(w - 1, avg_x))
                    avg_y = max(0, min(h - 1, avg_y))
                    
                    smoothed_landmarks.append((i, avg_x, avg_y))
                
                # Check stability
                if len(self.landmark_buffer) >= 3:
                    recent_frames = list(self.landmark_buffer)[-3:]
                    stable = self.check_landmark_stability(recent_frames)
                    if stable:
                        self.hand_stability_counter += 1
                    else:
                        self.hand_stability_counter = max(0, self.hand_stability_counter - 1)
                
                return smoothed_landmarks
            
            return current_landmarks
            
        except Exception as e:
            print(f"Error getting landmarks: {e}")
            return []
    
    def check_landmark_stability(self, landmark_frames, threshold=10):
        """Check if landmarks are stable across frames"""
        if len(landmark_frames) < 2:
            return False
            
        # Compare first and last frame
        frame1 = landmark_frames[0]
        frame2 = landmark_frames[-1]
        
        if len(frame1) != len(frame2):
            return False
            
        distances = []
        for (i1, x1, y1), (i2, x2, y2) in zip(frame1, frame2):
            if i1 == i2:
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                distances.append(distance)
        
        if distances:
            avg_distance = np.mean(distances)
            return avg_distance < threshold
        
        return False
    
    def get_index_tip(self, frame, hand_index=0):
        """Get enhanced smoothed position of index finger tip"""
        landmarks = self.get_landmarks_list(frame, hand_index)
        
        if not landmarks or len(landmarks) < 9:
            return None
        
        try:
            # Get index tip (landmark 8)
            index_tip = next((p for p in landmarks if p[0] == 8), None)
            
            if not index_tip:
                return None
            
            current_pos = (index_tip[1], index_tip[2])
            
            # Add to position buffer
            self.position_buffer.append(current_pos)
            
            # Apply weighted moving average
            if len(self.position_buffer) >= 2:
                weights = np.linspace(0.3, 1.0, len(self.position_buffer))
                weights = weights / weights.sum()
                
                avg_x = 0
                avg_y = 0
                for i, (x, y) in enumerate(self.position_buffer):
                    avg_x += x * weights[i]
                    avg_y += y * weights[i]
                
                return (int(avg_x), int(avg_y))
            
            return current_pos
            
        except Exception as e:
            print(f"Error getting index tip: {e}")
            return None
    
    def get_hand_center(self, frame, hand_index=0):
        """Get center of palm for better stability"""
        landmarks = self.get_landmarks_list(frame, hand_index)
        
        if not landmarks or len(landmarks) < 9:
            return None
        
        try:
            # Use wrist (0) and middle MCP (9) for stable center
            wrist = landmarks[0]
            middle_mcp = landmarks[9]
            
            center_x = (wrist[1] + middle_mcp[1]) // 2
            center_y = (wrist[2] + middle_mcp[2]) // 2
            
            return (center_x, center_y)
            
        except Exception as e:
            print(f"Error getting hand center: {e}")
            return None
    
    def is_hand_stable(self, threshold=10):
        """Check if hand position is stable"""
        if len(self.position_buffer) < 3:
            return False
        
        # Calculate variance of recent positions
        positions = list(self.position_buffer)[-3:]
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        
        var_x = np.var(xs)
        var_y = np.var(ys)
        
        return var_x < threshold and var_y < threshold
    
    def get_hand_stability(self):
        """Get hand stability score (0-100)"""
        if self.hand_stability_counter == 0:
            return 0
        
        # Cap at 10 frames for stability measurement
        stability_frames = min(self.hand_stability_counter, 10)
        return (stability_frames / 10) * 100
    
    def reset_buffers(self):
        """Reset smoothing buffers"""
        self.landmark_buffer.clear()
        self.position_buffer.clear()
        self.hand_stability_counter = 0