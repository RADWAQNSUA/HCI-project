#!/usr/bin/env python3
"""
Virtual Drawing Studio Pro - Enhanced Edition
Main Entry Point
"""
import sys
import os

# Set HighDPI attributes BEFORE creating QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

def main():
    # Enable high DPI scaling - MUST be set before QApplication is created
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Create application AFTER setting attributes
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Set application name
    app.setApplicationName("Virtual Drawing Studio Pro - Enhanced")
    app.setApplicationDisplayName("Virtual Drawing Studio Pro - Enhanced")
    
    # Ensure directories exist
    for directory in ['auto_saves', 'exports', 'config']:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    
    try:
        # Import the enhanced main window
        from main_window import MainWindow
        # Create and show main window
        window = MainWindow()
        window.show()
        
        # Start application
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    main()