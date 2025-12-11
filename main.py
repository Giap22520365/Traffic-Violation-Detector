import multiprocessing
from ultralytics import YOLO
import cv2
import numpy as np
from fast_alpr import ALPR
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus
from datetime import datetime

def connect():
	global database, collection
	uri = "mongodb+srv://22520365:Iamalive04@tfdb.zarmm6y.mongodb.net/?appName=TFdb"

	# Create a new client and connect to the server
	client = MongoClient(uri, server_api=ServerApi('1'))

	# Send a ping to confirm a successful connection
	try:
		client.admin.command('ping')
		print("Pinged your deployment. You successfully connected to MongoDB!")

		database = client["TFdb"]
		#database.create_collection("example_collection")
		collection = database["example_collection"]
	except Exception as e:
		print("reason for failure: ",e)

def save_to_db(lp):
	results = collection.find_one({ "License Plate" : str(lp) })
	if results is None:
		now = datetime.now()
		today = now.date()
		current_hour = now.hour
		current_minute = now.minute
		current_second = now.second
		
		save = collection.insert_one({"License Plate" : str(lp), "count" : "1", "day" : str(today), "hour": str(current_hour), "minute":str(current_minute), "second":str(current_second)})
	else:
		current_hour = now.hour
		current_minute = now.minute
		current_second = now.second
		
		if current_hour != int(results["hour"]) or current_minute != int(results["minute"]) or (int(results["second"]) - current_second >= 15):
			prev_count = int(results["count"])
			current_count = int(results["count"]) + 1
			
			query_filter = {"License Plate":str(lp), "count" : str(prev_count) }
			update_operation = { "$set" :  {"License Plate":str(lp), "count" : str(current_count) }}
			save = collection.update_one(query_filter, update_operation)

# ------------------------
# Polygon drawing callback
# ------------------------

def draw_polygon(event, x, y, flags, param):
	global polygon_points, final_polygon

	# Left click → add new point
	if event == cv2.EVENT_LBUTTONDOWN:
		polygon_points.append((x, y))

	# Right click → undo last point
	elif event == cv2.EVENT_RBUTTONDOWN:
		if polygon_points:
			final_polygon = False
			polygon_points.pop()

def point_in_polygon(x, y, polygon):
	"""Return True if point (x,y) is inside polygon."""
	poly = np.array(polygon, dtype=np.int32)
	return cv2.pointPolygonTest(poly, (x, y), False) >= 0

def save_plate(plate_text):
	with open("detected_plates.txt", "a") as f:
		f.write(plate_text + "\n")
	print("Saved LP:", plate_text)

img_lp = cv2.imread("C:/UIT-Courses/CE410.Q11 - Computer System Engineering/New folder/rotatequandoi13.jpg")
#img_lp = cv2.imread("C:/UIT-Courses/CE410.Q11 - Computer System Engineering/New folder/LP_detection/images/val/rotatequandoi97.jpg")

img_lp = cv2.resize(img_lp, (960, 540))

cv2.namedWindow('traffic')
cv2.setMouseCallback('traffic', draw_polygon)

def main():
	model = YOLO("best_lp.pt")
	vehicles = YOLO("vehicle.pt")
	#traffic_lights = YOLO("traffic_lights_custom.pt")
	#traffic_lights = YOLO("traffic-lights.pt")
	traffic_lights = YOLO("best_traffic_nano_yolo.pt")
	
	alpr = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-xs-v1-global-model"
	)
    
	global polygon_points, final_polygon, detected_lp_set, colour, collection, database
	polygon_points = []
	final_polygon = False
	detected_lp_set = set()
	colour = 'green'
	connect()
    # -------------------------------------------------------------------
    # Main loop → draw polygon repeatedly and wait for Enter / ESC
    # -------------------------------------------------------------------
	while True:
		
		display = img_lp.copy()

        # Draw points
		for p in polygon_points:
			cv2.circle(display, p, 4, (0, 255, 0), -1)

        # Draw edges
		if len(polygon_points) > 1:
			cv2.polylines(display, [np.array(polygon_points)], False, (255, 0, 0), 2)

        # If polygon finalized
		if final_polygon:
			cv2.polylines(display, [np.array(polygon_points)], True, (0, 0, 255), 2)
			# --- TRAFFIC LIGHTS ---
			tl_results = traffic_lights(display)
			for r in tl_results:
				print(r.names)
				for box in r.boxes:
					x1, y1, x2, y2 = box.xyxy[0].tolist()
					cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
					cls = int(box.cls[0])
					colour = traffic_lights.names[cls]
					print ("Colour: ", colour)
		    # --- VEHICLE DETECTION ---
			v_results = vehicles(display)

			for r in v_results:
				for box in r.boxes:
					x1, y1, x2, y2 = box.xyxy[0].tolist()
					cls = int(box.cls[0])
					name = vehicles.names[cls]

					# Compute object center
					cx = int((x1 + x2) / 2)
					cy = int((y1 + y2) / 2)

					# Check if center is inside polygon
					inside = point_in_polygon(cx, cy, polygon_points)

					if inside:
						# Draw the bounding box
						cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
						cv2.putText(display, name, (int(x1), int(y1) - 5),
							cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

						# Draw center point
						cv2.circle(display, (cx, cy), 4, (0, 255, 0), -1)
						# --- LICENSE PLATE DETECTION ---
						lp_results = model(display)
						for result in lp_results:
							xyxy = result.boxes.xyxy
							for box in xyxy:
								x1, y1, x2, y2 = map(int, box.tolist())
								cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 225), 2)
								texts = alpr.predict(display)
								if texts:
									text = texts[0].ocr.text
									print("Detected LP:", text)
									if text not in detected_lp_set:
										detected_lp_set.add(text)
										print("New LP detected:", text)
										save_plate(text)
										save_to_db(text, )
									else:
										print("Duplicate LP ignored:", text)

		cv2.imshow('traffic', display)

		k = cv2.waitKey(1) & 0xFF
        
		if k == 27:  # ESC → exit
			break
		if k == 13:  # Enter → finalize polygon
			enter = True
			if len(polygon_points) >= 3:
				final_polygon = True
				print("Polygon finalized:", polygon_points)

	cv2.destroyAllWindows()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
