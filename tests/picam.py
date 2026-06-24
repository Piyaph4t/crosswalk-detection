from picamera2 import Picamera2
import cv2

picam2 = Picamera2()

# Configure for a BGR-ish format OpenCV expects (3-channel)
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

frame = picam2.capture_array()  # returns a numpy array (H, W, 3), but in RGB order

# IMPORTANT: picamera2's "RGB888" is actually ordered BGR in memory on most builds,
# but don't trust that blindly — verify with a quick visual check (see note below).
# If colors look swapped in OpenCV, convert:
frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

cv2.imshow("frame", frame_bgr)
cv2.waitKey(0)

picam2.stop()
