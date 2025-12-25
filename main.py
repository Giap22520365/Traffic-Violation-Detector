import multiprocessing
from ultralytics import YOLO
import cv2
import numpy as np
from fast_alpr import ALPR
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime

# ------------------------
# MongoDB
# ------------------------

def connect():
    global database, collection
    uri = "mongodb+srv://22520365:Iamalive04@tfdb.zarmm6y.mongodb.net/?appName=TFdb"

    client = MongoClient(uri, server_api=ServerApi('1'))
    try:
        client.admin.command('ping')
        print("Connected to MongoDB")
        database = client["TFdb"]
        collection = database["example_collection"]
    except Exception as e:
        print("MongoDB error:", e)

def save_to_db(lp):
    now = datetime.now()

    results = collection.find_one({"License Plate": str(lp)})

    if results is None:
        collection.insert_one({
            "License Plate": str(lp),
            "count": "1",
            "day": str(now.date()),
            "hour": str(now.hour),
            "minute": str(now.minute),
            "second": str(now.second)
        })
    else:
        prev_time = (
            int(results["hour"]) * 3600 +
            int(results["minute"]) * 60 +
            int(results["second"])
        )
        curr_time = now.hour * 3600 + now.minute * 60 + now.second

        if curr_time - prev_time >= 15:
            collection.update_one(
                {"License Plate": str(lp)},
                {"$set": {
                    "count": str(int(results["count"]) + 1),
                    "hour": str(now.hour),
                    "minute": str(now.minute),
                    "second": str(now.second)
                }}
            )

# ------------------------
# Polygon utilities
# ------------------------

def draw_polygon(event, x, y, flags, param):
    global polygon_points, final_polygon

    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))

    elif event == cv2.EVENT_RBUTTONDOWN:
        if polygon_points:
            polygon_points.pop()
            final_polygon = False

def point_in_polygon(x, y, polygon):
    poly = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(poly, (x, y), False) >= 0

# ------------------------
# Main
# ------------------------

def main():
    global polygon_points, final_polygon, colour

    # ---------- MODELS ----------
    lp_model = YOLO("best_lp.pt")
    vehicle_model = YOLO("vehicle_orientation.pt")
    traffic_lights = YOLO("best_traffic_nano_yolo.pt")

    alpr = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-xs-v1-global-model"
    )

    connect()

    # ---------- VIDEO ----------
    video_path = "output.mp4"  # or 0 for webcam
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Cannot open video")
        return

    polygon_points = []
    final_polygon = False
    colour = "green"

    cv2.namedWindow("traffic")
    cv2.setMouseCallback("traffic", draw_polygon)

    # ---------- LOOP ----------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (960, 540))
        display = frame.copy()

        # Draw polygon points
        for p in polygon_points:
            cv2.circle(display, p, 4, (0, 255, 0), -1)

        if len(polygon_points) > 1:
            cv2.polylines(display, [np.array(polygon_points)], False, (255, 0, 0), 2)

        if final_polygon:
            cv2.polylines(display, [np.array(polygon_points)], True, (0, 0, 255), 2)

            # ---------- TRAFFIC LIGHT ----------
            tl_results = traffic_lights(display)
            for r in tl_results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    colour = traffic_lights.names[cls]

            # ---------- VEHICLE ----------
            v_results = vehicle_model(display)
            for r in v_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    name = vehicle_model.names[cls]

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    if (
                        point_in_polygon(cx, cy, polygon_points)
                        and "side" not in name
                        and colour in ["red", "yellow"]
                    ):
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(display, (cx, cy), 4, (0, 255, 0), -1)
                        vehicle_crop = frame[y1:y2, x1:x2]

                        # ---------- LICENSE PLATE ----------
                        lp_results = lp_model(vehicle_crop)
                        for lp in lp_results:
                            for box in lp.boxes.xyxy:
                                x1, y1, x2, y2 = map(int, box.tolist())
                                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)

                                texts = alpr.predict(display)
                                if texts:
                                    text = texts[0].ocr.text
                                    print("Detected LP:", text)
                                    save_to_db(text)

        cv2.imshow("traffic", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == 13 and len(polygon_points) >= 3:  # ENTER
            final_polygon = True
            print("Polygon finalized")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
