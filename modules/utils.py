import cv2

def annotate_frame(frame, bbox, name):
    """
    Draw bounding box + label on frame
    """
    x1, y1, x2, y2 = bbox
    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, name, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame
