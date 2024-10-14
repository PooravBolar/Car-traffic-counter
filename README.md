# 🚗 Vehicle Detection and Tracking

This project leverages the YOLO (You Only Look Once) algorithm to detect and track vehicles in a video stream. It counts the number of vehicles that pass through a predefined line in the video.

---

## 🧰 Requirements:

Before running the code, ensure you have installed the following Python libraries:

```bash
pip install ultralytics
pip install cvzone
pip install opencv-python
pip install numpy
pip install sort
```

---

## ✨ Features:
- Real-time Vehicle Detection: Detects various vehicle types (cars, buses, motorbikes, etc.) using YOLO.
- Vehicle Tracking: Keeps track of detected vehicles and assigns unique IDs to them.
- Count Vehicles: Counts the number of vehicles that cross a specified line in the frame.
- Customizable Detection Mask: Use a mask to focus on specific regions of the video.

---

## 🚀 How to Run:
1) Clone the Repository:

```bash
git clone https://github.com/YourUsername/Vehicle-Detection-Tracking
cd Vehicle-Detection-Tracking
```

2) Download YOLO Model: Download the YOLO model weights yolov8n.pt from the official YOLO repository and place it in the project directory.

3) Prepare Video: Place your video file (e.g., 15928377-hd_1280_720_30fps.mp4) in the project directory.

4) Run the Application:

```bash
python vehicle_detection_tracking.py
```

5) View Output: The output video will show detected vehicles with bounding boxes and their corresponding IDs.

6) Exit:

- Press the q key to quit the application.

---

## 🧑‍💻 Contribution:
Feel free to fork this repository, create a new branch, and submit a pull request. Suggestions or improvements are always welcome!

---

## 📧 Contact:
For any queries or feedback, reach out at:

📩 Email: [pooravbolar3@gmail.com]
