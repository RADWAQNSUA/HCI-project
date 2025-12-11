import tkinter as tk
from tkinter import ttk, colorchooser, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk

class ToolButton(tk.Button):
    def __init__(self, parent, text, command, icon=None, **kwargs):
        super().__init__(parent, text=text, command=command, **kwargs)
        self.icon = icon
        self.configure(
            relief=tk.FLAT,
            bg='#34495e',
            fg='white',
            activebackground='#2c3e50',
            activeforeground='white',
            font=("Arial", 10),
            height=2,
            width=15
        )
        
        if icon:
            self.configure(image=icon, compound=tk.LEFT)

class ColorButton(tk.Button):
    def __init__(self, parent, color_bgr, name, command):
        hex_color = self._bgr_to_hex(color_bgr)
        super().__init__(parent, 
                        bg=hex_color,
                        activebackground=hex_color,
                        relief=tk.RAISED,
                        command=lambda: command(color_bgr))
        self.color_bgr = color_bgr
        self.name = name
        
        # Add tooltip
        self.tooltip = tk.Label(parent, text=name, bg='black', fg='white')
        self.tooltip.place_forget()
        
        self.bind("<Enter>", self.show_tooltip)
        self.bind("<Leave>", self.hide_tooltip)
    
    def _bgr_to_hex(self, bgr):
        b, g, r = bgr
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def show_tooltip(self, event):
        self.tooltip.place(x=event.x_root - self.winfo_rootx() + 20, 
                          y=event.y_root - self.winfo_rooty() + 10)
    
    def hide_tooltip(self, event):
        self.tooltip.place_forget()

class StatusBar(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(bg='#2c3e50', height=30)
        
        # Status message
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(self, 
                                    textvariable=self.status_var,
                                    bg='#2c3e50', 
                                    fg='white',
                                    font=("Arial", 10),
                                    anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        # FPS counter
        self.fps_var = tk.StringVar(value="FPS: --")
        self.fps_label = tk.Label(self,
                                 textvariable=self.fps_var,
                                 bg='#2c3e50',
                                 fg='#95a5a6',
                                 font=("Arial", 9))
        self.fps_label.pack(side=tk.RIGHT, padx=10)
        
        # Gesture info
        self.gesture_var = tk.StringVar(value="Gesture: --")
        self.gesture_label = tk.Label(self,
                                     textvariable=self.gesture_var,
                                     bg='#2c3e50',
                                     fg='#95a5a6',
                                     font=("Arial", 9))
        self.gesture_label.pack(side=tk.RIGHT, padx=10)
    
    def set_status(self, message):
        self.status_var.set(message)
    
    def set_fps(self, fps):
        self.fps_var.set(f"FPS: {fps}")
    
    def set_gesture(self, gesture, confidence=0.0):
        if confidence > 0.7:
            self.gesture_var.set(f"Gesture: {gesture} ({confidence:.0%})")
        else:
            self.gesture_var.set(f"Gesture: {gesture}")

class ToolPanel(ttk.Frame):
    def __init__(self, parent, title, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Title
        title_frame = tk.Frame(self, bg='#34495e')
        title_frame.pack(fill=tk.X, padx=5, pady=(5, 0))
        
        title_label = tk.Label(title_frame, 
                              text=title,
                              bg='#34495e',
                              fg='white',
                              font=("Arial", 11, "bold"))
        title_label.pack(anchor=tk.W, padx=10, pady=5)
        
        # Separator
        separator = ttk.Separator(self, orient='horizontal')
        separator.pack(fill=tk.X, padx=10, pady=5)
        
        # Content frame
        self.content_frame = tk.Frame(self, bg='#34495e')
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    def add_widget(self, widget, **pack_args):
        widget.pack(in_=self.content_frame, **pack_args)

class ZoomSlider(ttk.Scale):
    def __init__(self, parent, command, **kwargs):
        super().__init__(parent, 
                        from_=10, 
                        to=500, 
                        orient=tk.HORIZONTAL,
                        command=command,
                        **kwargs)
        self.set(100)  # 100% zoom
        self.configure(length=200)

class CanvasDisplay(tk.Frame):
    def __init__(self, parent, width, height, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Canvas container with scrollbars
        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas
        self.canvas = tk.Canvas(self.canvas_frame, 
                               width=width, 
                               height=height,
                               bg='white',  # Changed from black to white
                               cursor="cross")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbars
        self.v_scrollbar = ttk.Scrollbar(self.canvas_frame, 
                                        orient=tk.VERTICAL,
                                        command=self.canvas.yview)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.h_scrollbar = ttk.Scrollbar(self, 
                                        orient=tk.HORIZONTAL,
                                        command=self.canvas.xview)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Configure canvas scrolling
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set,
                             xscrollcommand=self.h_scrollbar.set)
        
        # Bind mouse wheel for zoom
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Control-MouseWheel>", self.on_ctrl_mouse_wheel)
        
        # Image display
        self.image_id = None
        self.photo = None  # Keep reference to prevent garbage collection
    
    def on_mouse_wheel(self, event):
        # Vertical scrolling
        self.canvas.yview_scroll(-1 * (event.delta // 120), "units")
    
    def on_ctrl_mouse_wheel(self, event):
        # Horizontal scrolling
        self.canvas.xview_scroll(-1 * (event.delta // 120), "units")
    
    def display_image(self, pil_image):
        """Display PIL Image on canvas."""
        try:
            if pil_image is None:
                return
                
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update or create image
            if self.image_id is None:
                self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            else:
                self.canvas.itemconfig(self.image_id, image=photo)
            
            # Keep reference to prevent garbage collection
            self.photo = photo
            
            # Update scroll region
            self.canvas.configure(scrollregion=self.canvas.bbox(tk.ALL))
        except Exception as e:
            print(f"Error displaying image on canvas: {e}")
    
    def clear(self):
        """Clear canvas display."""
        try:
            if self.image_id:
                self.canvas.delete(self.image_id)
                self.image_id = None
            self.photo = None
        except Exception as e:
            print(f"Error clearing canvas: {e}")
    
    def get_canvas_coords(self, event):
        """Get canvas coordinates from event."""
        try:
            # Get canvas coordinates
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            return canvas_x, canvas_y
        except:
            return event.x, event.y

class ColorPickerDialog:
    def __init__(self, parent, current_color=(0, 0, 0)):
        self.parent = parent
        self.current_color = current_color
        self.result = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Color Picker")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center dialog
        self.dialog.geometry("400x500")
        x = parent.winfo_rootx() + (parent.winfo_width() - 400) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - 500) // 2
        self.dialog.geometry(f"+{x}+{y}")
        
        self.build_ui()
    
    def build_ui(self):
        # Current color display
        current_frame = tk.Frame(self.dialog)
        current_frame.pack(pady=10)
        
        hex_color = self._bgr_to_hex(self.current_color)
        self.current_display = tk.Label(current_frame,
                                       bg=hex_color,
                                       width=20,
                                       height=3,
                                       relief=tk.SUNKEN)
        self.current_display.pack()
        
        # Color presets
        presets_frame = tk.Frame(self.dialog)
        presets_frame.pack(pady=10)
        
        presets = [
            ("Black", (0, 0, 0)),
            ("White", (255, 255, 255)),
            ("Red", (0, 0, 255)),
            ("Green", (0, 255, 0)),
            ("Blue", (255, 0, 0)),
            ("Yellow", (0, 255, 255)),
            ("Magenta", (255, 0, 255)),
            ("Cyan", (255, 255, 0))
        ]
        
        for i, (name, color) in enumerate(presets):
            btn = ColorButton(presets_frame, color, name, self.on_preset_selected)
            btn.grid(row=i//4, column=i%4, padx=5, pady=5)
        
        # RGB sliders
        sliders_frame = tk.Frame(self.dialog)
        sliders_frame.pack(pady=10)
        
        self.red_var = tk.IntVar(value=self.current_color[2])
        self.green_var = tk.IntVar(value=self.current_color[1])
        self.blue_var = tk.IntVar(value=self.current_color[0])
        
        tk.Label(sliders_frame, text="Red:").grid(row=0, column=0, sticky=tk.W)
        red_slider = tk.Scale(sliders_frame, from_=0, to=255, 
                             variable=self.red_var, orient=tk.HORIZONTAL,
                             command=self.on_slider_change)
        red_slider.grid(row=0, column=1, padx=5)
        
        tk.Label(sliders_frame, text="Green:").grid(row=1, column=0, sticky=tk.W)
        green_slider = tk.Scale(sliders_frame, from_=0, to=255, 
                               variable=self.green_var, orient=tk.HORIZONTAL,
                               command=self.on_slider_change)
        green_slider.grid(row=1, column=1, padx=5)
        
        tk.Label(sliders_frame, text="Blue:").grid(row=2, column=0, sticky=tk.W)
        blue_slider = tk.Scale(sliders_frame, from_=0, to=255, 
                              variable=self.blue_var, orient=tk.HORIZONTAL,
                              command=self.on_slider_change)
        blue_slider.grid(row=2, column=1, padx=5)
        
        # Buttons
        buttons_frame = tk.Frame(self.dialog)
        buttons_frame.pack(pady=10)
        
        tk.Button(buttons_frame, text="OK", command=self.on_ok, 
                 width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(buttons_frame, text="Cancel", command=self.on_cancel,
                 width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(buttons_frame, text="System Picker", 
                 command=self.on_system_picker,
                 width=15).pack(side=tk.LEFT, padx=5)
    
    def _bgr_to_hex(self, bgr):
        b, g, r = bgr
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def on_preset_selected(self, color):
        self.current_color = color
        self.red_var.set(color[2])
        self.green_var.set(color[1])
        self.blue_var.set(color[0])
        self.update_display()
    
    def on_slider_change(self, *args):
        self.current_color = (self.blue_var.get(),
                             self.green_var.get(),
                             self.red_var.get())
        self.update_display()
    
    def update_display(self):
        hex_color = self._bgr_to_hex(self.current_color)
        self.current_display.configure(bg=hex_color)
    
    def on_system_picker(self):
        color = colorchooser.askcolor(title="Choose color")
        if color and color[0]:
            r, g, b = map(int, color[0])
            self.current_color = (b, g, r)
            self.red_var.set(r)
            self.green_var.set(g)
            self.blue_var.set(b)
            self.update_display()
    
    def on_ok(self):
        self.result = self.current_color
        self.dialog.destroy()
    
    def on_cancel(self):
        self.result = None
        self.dialog.destroy()
    
    def show(self):
        self.dialog.wait_window()
        return self.result