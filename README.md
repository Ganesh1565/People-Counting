# People Counter & Tracker

A real-time people counting and tracking system that uses YOLO object detection and SORT (Simple Online and Realtime Tracking) to monitor and count people moving in different directions across video streams.

##  Features

- **Real-time People Detection**: Uses YOLO11n model for fast and accurate person detection
- **Multi-Object Tracking**: Implements SORT algorithm for robust object tracking across frames
- **Directional Counting**: Tracks people moving left-to-right and right-to-left across a center line
- **Live Visualization**: Real-time display with bounding boxes, tracking IDs, and count displays
- **Video Processing**: Supports both video files and webcam input
- **Performance Optimization**: Configurable resolution scaling for faster processing

##  How It Works

The system works by:

1. **Detection**: YOLO model detects people in each video frame
2. **Tracking**: SORT algorithm assigns unique IDs to detected people and tracks them across frames
3. **Direction Analysis**: Monitors when tracked individuals cross a vertical center line
4. **Counting**: Maintains separate counts for left-to-right and right-to-left movements
5. **Visualization**: Displays real-time tracking boxes, IDs, and count information

##  Requirements

- Python 3.7+
- OpenCV (cv2)
- Ultralytics (YOLO)
- NumPy
- FilterPy (for SORT algorithm)

##  Installation

1. **Clone or download this repository**

2. **Install required packages:**
   ```bash
   pip install ultralytics opencv-python numpy filterpy scikit-image matplotlib
   ```

3. **Download YOLO model:**
   The project uses `yolo11n.pt` which should be placed in the project directory. You can download it from the Ultralytics repository or use your own YOLO model.

##  Project Structure

```
people/
├── app.py              # Main application file
├── sort.py             # SORT tracking algorithm implementation
├── yolo11n.pt          # YOLO model weights
├── README.md           # This file
└── *.mp4               # Sample video files
```

##  Usage

### Basic Usage

Run the main application:
```bash
python app.py
```

### Configuration Options

You can modify these parameters in `app.py`:

- **Video Source**: Change the video file path or use `0` for webcam
- **Resolution**: Adjust `resize_width` and `resize_height` for performance
- **Detection Confidence**: Modify the confidence threshold (currently 0.5)
- **Tracking Parameters**: Adjust SORT tracker parameters in the tracker initialization

### Controls

- **Q**: Quit the application
- **Real-time Display**: Shows tracking boxes, IDs, and counts

## 🔧 Customization

### Changing Video Source

```python
# For webcam
cap = cv2.VideoCapture(0)

# For different video file
cap = cv2.VideoCapture("your_video.mp4")
```

### Adjusting Detection Sensitivity

```python
# Change confidence threshold
if label.lower() == "person" and conf > 0.3:  # Lower threshold = more detections
```

### Modifying Tracking Parameters

```python
# SORT tracker parameters
tracker = Sort(max_age=30, min_hits=5, iou_threshold=0.4)
```

##  Output

The system provides:

- **Real-time Counts**: Left-to-right and right-to-left movement counts
- **Total Count**: Combined count of all tracked movements
- **Visual Tracking**: Green bounding boxes with unique tracking IDs
- **Center Line**: Red vertical line showing the counting boundary

##  Sample Videos

The project includes several sample videos for testing:
- `5330829-hd_1920_1080_30fps.mp4` - HD video (default)
- `3206583-uhd_3840_2160_30fps.mp4` - 4K video
- `4787385-uhd_2560_1440_30fps.mp4` - 2K video

##  Technical Details

### YOLO Model
- Uses YOLO11n for fast inference
- Optimized for person detection
- Configurable confidence thresholds

### SORT Algorithm
- Kalman filter-based tracking
- Hungarian algorithm for data association
- Handles occlusions and temporary disappearances

### Performance Considerations
- Resolution scaling for faster processing
- Configurable tracking parameters
- Efficient frame processing pipeline

##  Troubleshooting

### Common Issues

1. **Model not found**: Ensure `yolo11n.pt` is in the project directory
2. **Performance issues**: Reduce `resize_width` for faster processing
3. **Tracking errors**: Adjust SORT parameters (`max_age`, `min_hits`, `iou_threshold`)

### Dependencies Issues

If you encounter import errors:
```bash
pip install --upgrade ultralytics opencv-python
```

##  License

This project uses the SORT algorithm which is licensed under GNU General Public License v3.0. The main application code is provided as-is for educational and research purposes.

##  Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project.

##  References

- [YOLO: Real-Time Object Detection](https://github.com/ultralytics/ultralytics)
- [SORT: Simple Online and Realtime Tracking](https://github.com/abewley/sort)
- [OpenCV Documentation](https://docs.opencv.org/)

##  Support

For questions or issues, please check the troubleshooting section above or create an issue in the project repository.
