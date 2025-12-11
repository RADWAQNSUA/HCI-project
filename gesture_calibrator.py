"""
Gesture Calibrator - Enhanced for PyQt5
"""
import cv2
import numpy as np
from collections import deque

class GestureCalibrator:
    def __init__(self):
        self.calibration_steps = [
            'open_hand',
            'fist',
            'pinch',
            'pointing',
            'victory'
        ]
        self.current_step = 0
        self.calibration_data = {}
        self.is_calibrating = False
        self.calibration_complete = False
        
    def start_calibration(self, frame_width, frame_height):
        self.calibration_data = {}
        self.current_step = 0
        self.is_calibrating = True
        self.calibration_complete = False
        return {
            'step': self.current_step + 1,
            'total_steps': len(self.calibration_steps),
            'gesture': self.calibration_steps[self.current_step],
            'message': f"Step 1/{len(self.calibration_steps)}: Show OPEN HAND"
        }
    
    def process_calibration_step(self, landmarks, frame=None):
        if not landmarks or not self.is_calibrating:
            return None
        
        step_name = self.calibration_steps[self.current_step]
        hand_size = self._calculate_hand_size(landmarks)
        
        # Store calibration data for this step
        if step_name not in self.calibration_data:
            self.calibration_data[step_name] = {
                'hand_size': hand_size,
                'landmarks': landmarks,
                'finger_states': self._get_finger_states(landmarks),
                'timestamp': cv2.getTickCount() / cv2.getTickFrequency()
            }
        
        # Add visual feedback to frame if provided
        if frame is not None:
            self._draw_calibration_ui(frame, step_name)
        
        return {
            'step': self.current_step + 1,
            'total_steps': len(self.calibration_steps),
            'gesture': step_name,
            'hand_size': hand_size,
            'progress': (self.current_step + 1) / len(self.calibration_steps)
        }
    
    def _draw_calibration_ui(self, frame, step_name):
        """Draw calibration UI on frame."""
        h, w = frame.shape[:2]
        
        # Draw instruction
        instruction = f"Step {self.current_step + 1}/{len(self.calibration_steps)}: {step_name.upper()}"
        cv2.putText(frame, instruction, (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw progress bar
        progress = (self.current_step + 1) / len(self.calibration_steps)
        bar_width = int(w * 0.8)
        bar_height = 20
        bar_x = (w - bar_width) // 2
        bar_y = h - 100
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        
        # Progress
        fill_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + fill_width, bar_y + bar_height), 
                     (0, 255, 0), -1)
        
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (255, 255, 255), 2)
        
    def next_step(self):
        if self.current_step < len(self.calibration_steps) - 1:
            self.current_step += 1
            return {
                'step': self.current_step + 1,
                'total_steps': len(self.calibration_steps),
                'gesture': self.calibration_steps[self.current_step],
                'message': f"Step {self.current_step + 1}/{len(self.calibration_steps)}: {self.calibration_steps[self.current_step].upper()}"
            }
        else:
            self.is_calibrating = False
            self.calibration_complete = True
            result = self._calculate_thresholds()
            return {
                'complete': True,
                'message': result,
                'thresholds': self.calibration_thresholds if hasattr(self, 'calibration_thresholds') else None,
                'base_hand_size': self.base_hand_size if hasattr(self, 'base_hand_size') else None
            }
    
    def _calculate_hand_size(self, landmarks):
        """Calculate hand size based on wrist to middle finger MCP distance"""
        if len(landmarks) < 10:
            return 100
        
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        
        dx = wrist[1] - middle_mcp[1]
        dy = wrist[2] - middle_mcp[2]
        
        return np.sqrt(dx*dx + dy*dy)
    
    def _get_finger_states(self, landmarks):
        """Get raw finger states without thresholds"""
        states = {}
        
        # Finger tip and PIP indices
        finger_data = [
            ('thumb', 4, 3),
            ('index', 8, 6),
            ('middle', 12, 10),
            ('ring', 16, 14),
            ('pinky', 20, 18)
        ]
        
        for name, tip_idx, pip_idx in finger_data:
            if tip_idx < len(landmarks) and pip_idx < len(landmarks):
                tip_y = landmarks[tip_idx][2]
                pip_y = landmarks[pip_idx][2]
                states[name] = tip_y < pip_y  # Tip above PIP = extended
        
        return states
    
    def _calculate_thresholds(self):
        """Calculate dynamic thresholds from calibration data"""
        if 'open_hand' not in self.calibration_data:
            return "Calibration failed: Missing open hand data"
        
        open_hand_data = self.calibration_data['open_hand']
        base_hand_size = open_hand_data['hand_size']
        
        # Calculate finger extension thresholds
        thresholds = {}
        
        if 'open_hand' in self.calibration_data and 'fist' in self.calibration_data:
            open_states = self.calibration_data['open_hand']['finger_states']
            fist_states = self.calibration_data['fist']['finger_states']
            
            for finger in open_states:
                if finger in fist_states:
                    # Store the difference as threshold
                    thresholds[finger] = base_hand_size * 0.12
        
        # Calculate pinch threshold
        if 'pinch' in self.calibration_data:
            pinch_data = self.calibration_data['pinch']
            if len(pinch_data['landmarks']) >= 9:
                thumb_tip = pinch_data['landmarks'][4]
                index_tip = pinch_data['landmarks'][8]
                dx = thumb_tip[1] - index_tip[1]
                dy = thumb_tip[2] - index_tip[2]
                pinch_distance = np.sqrt(dx*dx + dy*dy)
                thresholds['pinch'] = pinch_distance * 1.5
        
        self.calibration_thresholds = thresholds
        self.base_hand_size = base_hand_size
        
        return f"Calibration complete! Hand size: {base_hand_size:.1f}"
    
    def get_gesture_thresholds(self):
        return self.calibration_thresholds if hasattr(self, 'calibration_thresholds') else None
    
    def get_base_hand_size(self):
        return self.base_hand_size if hasattr(self, 'base_hand_size') else 100
    
    def reset(self):
        """Reset calibration data"""
        self.calibration_data = {}
        self.current_step = 0
        self.is_calibrating = False
        self.calibration_complete = False
        if hasattr(self, 'calibration_thresholds'):
            del self.calibration_thresholds
        if hasattr(self, 'base_hand_size'):
            del self.base_hand_size