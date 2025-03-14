
app.py
DetailsActivity
#### Library Imports ####
# MavSDK Related
import asyncio
import nest_asyncio
import subprocess
# Map Related
from shapely.geometry import Point, Polygon as
ShapelyPolygon
# Other Libraries
import json
import math
import os
import time
import signal
import logging
import tkinter as tk
from tkinter import filedialog
import threading
from datetime import datetime
from geopy.distance import geodesic
from multiprocessing import Pool
from typing import List, Tuple
from shapely.geometry import LineString, Polygon, Point
from geopy.distance import geodesic
from flask import Flask, render_template, request, jsonify,
current_app
from flask_cors import CORS
from flask_socketio import SocketIO, emit
# Custom Libraries
from drone import Drone
from observer import Observer
import subprocess
from net_rid_server import NetRidServer
from flight_plan_server import FlightPlanServer
from flight_plan import FlightPlan
from red_area import RedArea
from red_areas_server import RedAreasServer
from gpt_eng import GPT_ENG as GPT
import queue
#### Global Variables & Conditional Imports ####
simulation_process = None
file_queue = queue.Queue()
launched_instances = False
programmed_uavs = False
sim_run_before = False
net_rid_server = NetRidServer()
flight_plan_server = FlightPlanServer()
red_areas_server = RedAreasServer()
nest_asyncio.apply()
SIMULATION_SPEED_FACTOR = 10 # Default = 1, Real-World Speed
LLM_TRIGGER_PERIODICITY = 30 # Seconds
UPDATE_LOCATION_INTERVAL = 5 # Seconds
RUN_INTERNAL_CALC = False
RUN_LLM_PROMPTS = True # Turn On or Off LLM
REINJECT_PAST_LLM_DATA = True
HEADLESS_SIM = 1 # Run the simulation without the jMavSim
GUI, runs quicker (1/0)
PRESET_FILE = 'presets'
if (RUN_LLM_PROMPTS):
from llm_data_formatter import llm_to_html, llm_error
from llm_player import play_llm_message_tts
# Paths to Installations
EXP_NAME = "SEMI-COMPLIANT_2"
# Paths to Installations inside the container
PX4_PATH = "/opt/PX4-Autopilot"
MAVSDK_PATH = "/opt/MAVSDK"
initial_flight_alt = 50 # Meters, Future Consideration for
3D Red Zones (Height Dependent Restrictions)
internal_calc_timer = datetime.now()
llm_call_number = 1
uav_id = -1
observer_id = -1
red_area_id = -1
trials = 0
trig_time = None
abu_dhabi = 24.448440, 54.395426
abu_dhabi_coords = {
"latitude": 24.448440,
"longitude": 54.395426
}
observers = []
current_observer = 0
observers_data = [] # Keeps Track of Chat History,
Reinjection, and First Run
class OB_LLM_DAT:
def __init__(self, reinject):
self.first_run = True
self.reinject = reinject
self.chat_history = []
red_areas = []
uavs = []
processes = []
llm_response = ""
# Configure Logging to Debug While Flask Runs
logging.basicConfig(level=logging.INFO, format='%(asctime)s
- %(levelname)s - %(message)s')
##########################################################
###############################################
##########################################################
###############################################
##########################################################
###############################################
##########################################################
###############################################
#### Helper Functions and Classes ####
def instantiate_uavs_observers_red_areas(data: str):
global uavs, observers, red_areas, red_areas_server
# Load Data into Global Variables
observers = [Observer(id=ob['id'], lat=ob['lat'],
long=ob['lon'], range_radius=ob['radius'],
vis_on=ob['vis_on']) for ob in data.get('observers', [])]
red_areas = [RedArea(id=ra['id'],
description=ra['description'], polygon=ra['polygon']) for
ra in data.get('red_areas', [])]
red_areas_server.red_areas = red_areas
uavs = [Drone(id=uav['id'], port_indent=uav['id'],
actual_coords=uav['actual'],
submitted_coords=uav['submitted'],
flightplan_public=uav['auth'],
flightplan_separate=uav['auth'],
net_rid_enabled=uav['net_rid'],
b_rid_enabled=uav['b_rid'], drone_type= uav['uav_type'],
flying_altitude=int(uav['altitude']), auth=uav['auth'],
launch_delay=uav['launch_delay']) for uav in
data.get('uavs', [])]
for uav in uavs:
fp = None
if(uav.flightplan_separate):
inverted_coords = []
for p in uav.drone_public_path:
inverted_coords.append([p['lng'], p['lat']])
# Flight Plan Objects Use Inverted Coords
fp = FlightPlan(uav.id, uav.flying_altitude,
inverted_coords)
else:
inverted_coords = []
for p in uav.drone_path:
inverted_coords.append([p['lng'], p['lat']])
# Flight Plan Objects Use Inverted Coords
fp = FlightPlan(uav.id, uav.flying_altitude,
inverted_coords)
# Sets Flight Plan
uav.set_flight_plan(fp)
# If Public/Submitted, Send Flight Plan to Flight Plan
Server
if(uav.flightplan_public):
flight_plan_server.add_flight_plan(fp)
def open_file_dialog():
global file_queue
root = tk.Tk()
root.withdraw()
file_path =
filedialog.askopenfilename(initialdir="/presets",
title="Select Preset File", filetypes=(("JSON files",
"*.json"), ("all files", "*.*")))
file_queue.put(file_path)
def is_point_in_polygon(lat, lon, polygon_coords): # Backend
Helper Function
point = Point(lon, lat)
polygon = ShapelyPolygon([(lon, lat) for lat, lon in
polygon_coords])
return polygon.contains(point)
def err_cb(e):
logging.info(f"Error Pinging LLM: {str(e)}")
# Function to calculate geodesic distance between two points
def geodesic_distance(p1, p2):
return geodesic((p1.y, p1.x), (p2.y, p2.x)).meters
# Socket Function to Send Current Drone Status
def update_frontend_uavs():
global uavs
uav_data = {}
for drone in uavs:
uav_data[drone.id] = drone.get_location()
socketio.emit('update_uavs_state', uav_data)
def make_observers():
for ob in observers:
rid_drones_in_range = []
vis_drones_in_range = []
drone_ids_range = []
for drone in uavs:
if ((drone.lat is not None) and (drone.long is not
None)):
if (ob.check_drone_in_range((drone.lat,
drone.long))):
if (ob.vision_on):
vis_drones_in_range.append(drone.get_vis_id_message())
if (drone.b_rid_enabled):
rid_drones_in_range.append(drone.get_rid_message())
drone_ids_range.append(drone.id)
ob.set_vis_drones_in_range(vis_drones_in_range)
ob.set_brid_drones_in_range(rid_drones_in_range)
def run_tests(uav) -> Tuple[float, bool, float]:
global red_areas_server # Directly Access Red Areas Server
global flight_plan_server # Directly Access Flight Plan
Server
# Define UAV Location
point_coords = (uav.long, uav.lat)
point = Point(point_coords)
print(f"\n\n{flight_plan_server.get_flights_json()}\n\n")
# Simulate Grabbing from a Flight Plan Server
index = uav.id
fp = flight_plan_server.get_flight_plans()[index]
deviation = None
if (fp is not None):
# Define flight plan
line_points = []
for i in fp.drone_path:
line_points.append((i[1], i[0]))
line = LineString(line_points)
# Find the nearest point on the path to the point
nearest_point_on_line =
line.interpolate(line.project(point))
# Calculate the distance between the uav and the nearest
point on the path
deviation = geodesic_distance(point,
nearest_point_on_line)
else:
print("\n\n\n\nFLIGHT PLAN NOT VISIBLE\n\n\n\n")
is_inside_no_fly = None
min_distance_no_fly = math.inf
# Simulate Grabbing from a Red Area Server
# List of Red Areas
red_areas = red_areas_server.get_red_areas()
for red_area in red_areas:
polygon_coords = []
for coords in red_area.polygon:
for coord in coords:
polygon_coords.append((coord['lng'], coord['lat']))
polygon = Polygon(polygon_coords)
if polygon.contains(point):
print("\n\n\n\nin no fly\n\n\n\n")
is_inside_no_fly = True
min_distance_no_fly = 0
break
else:
for i in range(len(polygon_coords)):
p1 = polygon_coords[i]
p2 = polygon_coords[(i + 1) % len(polygon_coords)]
distance_to_start = geodesic(point_coords,
p1).meters
distance_to_end = geodesic(point_coords, p2).meters
# Update the minimum distance
min_distance_no_fly = min(min_distance_no_fly,
distance_to_start, distance_to_end)
logging.info(f"{uav.id} is inside a no fly zone:
{is_inside_no_fly}")
logging.info(f"{uav.id}'s Distance to No Fly Zone
{min_distance_no_fly} meters")
return (deviation, is_inside_no_fly, min_distance_no_fly)
# Upon Pressing of Start Button
async def start_full_simulation():
global sim_run_before, uavs, observers, observers_data,
net_rid_server, red_areas_server, processes, trials,
internal_calc_timer, flight_plan_server, llm_call_number,
current_observer, trig_time
# Instantiate Observers Data list
for observer in observers:
observers_data.append(OB_LLM_DAT(reinject=REINJECT_PAST_LL
M_DATA))
trials += 1
update_count = 0
make_observers() # Initialize Observers with Respect to
Drones
socketio.emit("start_timer", {"sim_speed_factor":
SIMULATION_SPEED_FACTOR})
await start_uav_missions()
# Infinite Loop
while (True):
# List to Keep of Whether all Drones Have Finished
Flight, Check if List is "All True"
# Resets on Every Iteration
end_of_path = []
for i, drone in enumerate(uavs):
await drone.update_location()
fin = drone.check_drone_end()
end_of_path.append(fin)
if (drone.net_rid_enabled):
net_rid_server.send_net_rid_message(drone.get_rid_message(
))
# Divide Updating Frequency by 100x
if update_count % 100 == 0:
update_frontend_uavs()
update_count = 0
update_count += 1
# Exit Condition
# Kill all LLM-Interaction Instances
if(all(end_of_path)):
for process in processes:
os.killpg(os.getpgid(process.pid),
signal.SIGTERM)
processes.clear()
break
for count, ob in enumerate(observers):
rid_drones_in_range = []
vis_drones_in_range = []
drone_ids_range = []
# Set Current Observer to be used in ping_llm
callback handler function
current_observer = ob.id
for drone in uavs:
if ((drone.lat is not None) and (drone.long is not
None)):
if(ob.check_drone_in_range((drone.lat,
drone.long))):
if (ob.vision_on):
vis_drones_in_range.append(drone.get_vis_id_message())
if (drone.b_rid_enabled):
rid_drones_in_range.append(drone.get_rid_message())
drone_ids_range.append(drone.id)
ob.set_vis_drones_in_range(vis_drones_in_range)
ob.set_brid_drones_in_range(rid_drones_in_range)
if RUN_INTERNAL_CALC:
t1 = datetime.now()
last_ping_diff = (t1 -
internal_calc_timer).total_seconds()
# print(f"T1: {t1} =?= {internal_calc_timer} Timer
! PING:_DIFF: {last_ping_diff}")
if (last_ping_diff > 3):
logging.info("Running Tests")
internal_calc_timer = t1
for uav in uavs:
# RUN TESTS ACCESSES FLIGHT PLAN SERVER DIRECTLY
# if len(flight_plan_server.flight_plans) > 0:
# fp =
flight_plan_server.flight_plans[0].drone_path
# else:
# fp = None
(deviation, is_inside_no_fly,
min_distance_no_fly) = run_tests(uav)
if RUN_LLM_PROMPTS:
for count, ob in enumerate(observers):
mes = {}
if(len(ob.drones_vis_in_range) > 0 or
len(ob.drones_brid_in_range) > 0): # if drones in range
t1 = datetime.now() # Take note of start time
(to calculate total time taken)
last_ping_diff = (t1 -
ob.last_llm_ping).total_seconds()
if(last_ping_diff > LLM_TRIGGER_PERIODICITY): #
If the LLM hasn't been called in at least
LLM_TRIGGER_PERIODICITY time, then we call it again
ob.last_llm_ping = t1
internal_calc_timer = t1
# Iterates over all UAVs in UAvs
for uav in uavs:
# RUN TESTS ACCESSES FLIGHT PLAN SERVER
DIRECTLY
# if len(flight_plan_server.flight_plans) >
0:
# fp =
flight_plan_server.flight_plans[0].drone_path
# else:
# fp = None
(deviation, is_inside_no_fly,
min_distance_no_fly) = run_tests(uav)
# Determines the Following for the LLM:
# deviation: Distance from the UAV to the
nearest point on the flight path.
# is_inside_no_fly: Boolean indicating if
the UAV is inside the no-fly zone.
# min_distance_no_fly: Minimum distance
from the UAV to the no-fly zone boundary.
net_rid_server_mes =
net_rid_server.get_current_net_rid_message_json(ob.lat,
ob.long, ob.range_radius)
flight_plan_server_mes =
flight_plan_server.get_flights_json()
observer_observations_mes =
ob.get_drones_json()
no_fly_zones_mes =
red_areas_server.get_red_areas_json()
mes["Network_RID_Server_Data"] =
net_rid_server_mes
mes["FlightPlan_Server_Data"] =
flight_plan_server_mes
mes["Observer_Observations"] =
observer_observations_mes
mes["No_Fly_Zones"] = no_fly_zones_mes
mes["UAV_Deviation_From_Flight_Plan_metres"] =
round(deviation) if(deviation is not None) else "Data Not
Available"
mes["UAV_Is_Within_No_Fly_Zone"] =
is_inside_no_fly if(is_inside_no_fly is not None) else "Data
Not Available"
mes["UAV_Deviation_From_No_Fly_Zone_metres"] =
round(min_distance_no_fly) if(min_distance_no_fly is not
math.inf) else "Data Not Available"
mes_ready = json.dumps(mes)
involved_uavs = [] # Prepare ALL drones/uavs
info (drone id, distance) for LLM
for drn in uavs:
distance = geodesic((drn.lat, drn.long),
(ob.lat, ob.long)).meters
res = distance < ob.range_radius
if(res):
involved_uavs.append(drn.id)
else:
logging.info(f"UAV Not in Range")
pool = Pool(processes=1) # Start a worker
process.
trig_time = datetime.now()
print(f"\n\n1: Observer ID: {ob.id} -\n2.
Trigger Time: {trig_time} -\n3. Involved UAVs:
{involved_uavs} -\n4. Observer Object: {ob}\n\n")
print(f"\n\n\nMessage to Print:
{mes_ready}\n\n\n")
pool.apply_async(ping_llm, args=[ob,
mes_ready, involved_uavs, observers_data[ob.id]],
callback=ping_llm_callback, error_callback=err_cb)
time.sleep(UPDATE_LOCATION_INTERVAL) # sleep for variable
time, then run next loop iteration
socketio.emit("simulation_done")
logging.info("Done with Simulation")
#### Purely Backend ####
def ping_llm(observer: Observer, ob_data: str,
involved_uavs: List[Drone], obs_data: OB_LLM_DAT) ->
Tuple[Observer, str, list, bool, list]:
global trials
llm = GPT(observer=observer,
reinject=REINJECT_PAST_LLM_DATA)
(response, first_run, llm_history) =
llm.drid_query(query=ob_data, iteration=trials,
first_run=obs_data.first_run,
reinject=REINJECT_PAST_LLM_DATA,
llm_hist=obs_data.chat_history)
logging.info(f"\n\nLLM Response @ping_llm:
{response}\n\n")
return [observer.id, response, involved_uavs, first_run,
llm_history]
# Removed "obd" parameter
def ping_llm_callback(result):
global llm_reports_content, uavs, observers_data,
llm_call_number, observers, current_observer, llm_response,
trig_time
uav_response_list = []
# result contains: (observer, llm_response, involved_uavs,
first_run, llm_history)
observer = result[0]
llm_response = result[1]
involved_uavs = result[2]
first_run = result[3]
llm_history = result[4]
# Set First Run to False for Selected Observer
if first_run:
observers_data[observer].first_run = False
# Update the LLM Chat History for Observer X (system,
assistant, user)
observers_data[observer].chat_history = llm_history
# does this need to go in a try catch?
print(f"\n\nLLM Response: {llm_response}\n\n")
temp = llm_response
if(not "ERROR" in temp):
end_time = datetime.now()
time_taken = (end_time - trig_time).total_seconds()
logging.info(f"LLM request took {time_taken}")
# Play Audio Script File Generated by LLM
# play_llm_message_tts(EXP_NAME, llm_call_number,
llm_response)
# llm_call_number += 1
# Unpack LLM Response JSON into uav_response_list for
each UAV
llm_response_json = json.loads(llm_response)
for sub_response in llm_response_json.get("UAVs", []):
print(f"\n\nSub Response: {sub_response}\n\n")
uav_response_list.append(
json.dumps
(
{
'flight_id': sub_response.get("Flight_id"),
'distance_from_no_fly_zones':
sub_response.get("Distance_from_no_fly_zones"),
'is_in_fly_zone':
sub_response.get("Is_in_fly_zone"),
'deviation_distance_from_approved_path':
sub_response.get("Deviation_distance_from_approved_path"),
'risk_assessment':
sub_response.get("Risk_assessment"),
'time_taken': time_taken
}
)
)
print(f"\n\nUAV Response List:
{uav_response_list}\n\nType: {type(uav_response_list)}")
# Emit the message to the frontend
socketio.emit('llm_response', uav_response_list)
async def start_uav_missions():
global uavs
async def launch_uav_with_delay(uav, delay):
await asyncio.sleep(delay/SIMULATION_SPEED_FACTOR)
logging.info(f"===========================================
== Starting Drone {uav.id}
=============================================")
logging.info("------------------------ Arming Drone ---
---------------------")
await uav.arm()
logging.info("------------------------ Armed Drone ----
--------------------")
logging.info("------------------------ Starting Mission
------------------------")
await uav.start_mission()
logging.info("------------------------ Started Mission
------------------------")
tasks = []
for uav in uavs:
tasks.append(launch_uav_with_delay(uav,
uav.launch_delay))
await asyncio.gather(*tasks)
def launch_instances():
global processes, launched_instances, uavs, observers
command = f"/opt/PX4-
Autopilot/Tools/simulation/sitl_multiple_run.sh
{len(uavs)}"
logging.info(f"Simulator Command: {command}")
process = subprocess.Popen(command, shell=True,
text=True, start_new_session=True)
processes.append(process)
for count, uav in enumerate(uavs):
process = uav.start_simulator(SIMULATION_SPEED_FACTOR,
PX4_PATH, HEADLESS_SIM)
processes.append(process)
process = uav.start_mavsdk_server(MAVSDK_PATH)
processes.append(process)
logging.info(f"###########################################
##########################################################
#######")
logging.info(f"###########################################
###### ALL DONE
#################################################")
logging.info(f"###########################################
##########################################################
#######")
launched_instances = True # Mark Instances as Ready to be
Deployed
async def upload_to_uavs():
global programmed_uavs, uavs
for count, drn in enumerate(uavs):
logging.info(f"======================= Preparing Drone
{count+1}/{len(uavs)} ========================")
logging.info("------------------------ Connecting to
System ------------------------")
await drn.connect_to_drone_system()
logging.info("------------------------ Connected to
System ------------------------")
logging.info("------------------------ Uploading
Flight-Plan ------------------------")
await drn.upload_flight_plan()
programmed_uavs = True # UAVs are Programmed/Ready to Go
def save_preset(object_list: str, preset_filename:
str=PRESET_FILE):
'''Saves Flight Plan, Red Zones, and Observers via JSON
Dump'''
with open(preset_filename, 'w') as f:
json.dump(object_list, f)
def load_presets(preset_filename: str=PRESET_FILE) -> str:
'''Loads Flight Plan, Red Zones, and Observers via JSON
Load'''
with open(preset_filename, 'r') as f:
presets = json.load(f)
return presets
def handle_load_preset():
root = tk.Tk()
root.withdraw()
def task():
file_path =
filedialog.askopenfilename(initialdir="/presets",
title="Select Preset File", filetypes=(("JSON files",
"*.json"), ("all files", "*.*")))
if file_path:
data = load_presets(file_path)
socketio.emit('loaded_preset', data)
root.quit()
root.after(0, task)
root.mainloop()
##########################################################
###############################################
##########################################################
###############################################
##########################################################
###############################################
##########################################################
###############################################
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
@app.route('/')
def index():
return render_template('index.html')
# Route to Save Presets (saving UAVs, observers, and red
areas)
@socketio.on('save_preset')
def save_preset_button(request):
data = json.loads(request)
preset_filename = "presets/" + data.get('filename',
PRESET_FILE) + ".json"
data.pop('filename', None)
save_preset(data, preset_filename)
# Route to Load Presets
@socketio.on('load_presets')
def load_preset():
global uavs, observers, red_areas
def open_file_dialog():
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
initialdir="presets",
title="Select Preset File",
filetypes=(("JSON files", "*.json"), ("all
files", "*.*"))
)
root.destroy()
return file_path
# Run the file dialog in a separate thread
def load_preset_thread():
file_path = open_file_dialog()
if not file_path:
logging.warning("No preset file selected.")
return
try:
with open(file_path, 'r') as file:
data = json.load(file)
socketio.emit('loaded_preset', data)
logging.info("Preset loaded successfully.")
except Exception as e:
logging.error(f"Error loading preset: {e}")
threading.Thread(target=load_preset_thread).start()
# Handling Start Simulation Button Press
@socketio.on('start_simulation')
def start_simulation(request):
global initial_flight_alt, simulation_process, uavs,
observers, red_areas
# Clean Slate
end_simulation()
time.sleep(5)
data = json.loads(request)
logging.info("Simulation starting with the following
data:")
logging.info("Observers: %s, Type: %s",
data.get('observers', []), type(data.get('observers', [])))
logging.info("Red Areas: %s, Type: %s",
data.get('red_areas', []), type(data.get('red_areas', [])))
logging.info("UAVs: %s, Type: %s", data.get('uavs', []),
type(data.get('uavs', [])))
instantiate_uavs_observers_red_areas(data)
launch_instances()
loop = asyncio.get_event_loop()
result = loop.run_until_complete(upload_to_uavs())
time.sleep(6)
asyncio.run(start_full_simulation())
return jsonify({'status': 'success', 'message':
'Simulation Started Successfully'})
@socketio.on('end_simulation')
#### Communicate with Frontend (Backend-Flask Hybrid
Functions) ####
def end_simulation(): # INVOLVES FLASK
global uavs, observers, red_areas, net_rid_server,
flight_plan_server, red_areas_server, llm_reports_content
subprocess.run(["./clear_drones.sh"])
net_rid_server = NetRidServer()
flight_plan_server = FlightPlanServer()
red_areas_server = RedAreasServer()
uavs = []
observers = []
red_areas = []
uavs.clear()
observers.clear()
red_areas.clear()
for process in processes:
os.killpg(os.getpgid(process.pid), signal.SIGTERM)
processes.clear()
llm_reports_content = ""
# TODO: IF LLM KEEPS PREVIOUS INPUTS THEN RESET HERE
logging.info("Restarting Simulation")
@app.route('/filter', methods=['GET'])
def filter_uavs():
severity_level = request.args.get('severity',
default=0, type=int)
# Ensure `uavs` exists and contains data
matching_uavs = [uav for uav in uavs if
uav.get('compliance_status_int', -1) >= severity_level]
# Emit filtered UAVs via Socket.IO
socketio.emit('filtered_uavs', {"matching":
matching_uavs})
return jsonify({"status": "success"})
@socketio.on('stop_simulation')
def stop_simulation():
global processes
for process in processes:
os.killpg(os.getpgid(process.pid), signal.SIGTERM)
processes.clear()
return jsonify({'status': 'success', 'message':
'Simulation Stopped Successfully'})
if __name__ == '__main__':
socketio.run(app, host='0.0.0.0',
allow_unsafe_werkzeug=True)# sdp-LLM
