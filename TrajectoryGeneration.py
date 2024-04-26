import os
import sys
import time
import math
import pandas as pd
import numpy as np
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI


#######################################################
#"""
# Set and Test the xArm IP
# """
if len(sys.argv) >= 2:
     ip = sys.argv[1]
else:
    try:
         from configparser import ConfigParser
         parser = ConfigParser()
         parser.read('../robot.conf')
         ip = parser.get('xArm', 'ip')
    except:
        ip = "192.168.1.230"
        #ip = input('Please input the xArm ip address:')
        if not ip:
            print('input error, exit')
            sys.exit(1)
########################################################

# import the initial data csv file
dt = 0.01 # 100Hz
df = pd.read_csv('init_data_spt.csv')
df['t'] = np.linspace(0, dt*df.shape[0], df.shape[0])


command_lines =[]
for idx, t in enumerate(df[f'q_0']):
    command_line = [df[f'q_0'][idx],df[f'q_1'][idx],
                    df[f'q_2'][idx],df[f'q_3'][idx],
                    df[f'q_4'][idx],df[f'q_5'][idx],df[f'q_6'][idx]]
    command_lines.append(command_line)


#command_lines = np.asarray(command_lines) * (360.01/(2.0*np.pi))
#print(command_lines)


        

# for i in range(7):
#     q = q.append(df[f'q_{i}'])
#     t = df['t']
#     dq = np.gradient(q, t)
#     df[f'dq_{i}'] = dq
#     df[f'ddq_{i}'] = np.gradient(dq, t)

# print(q)

# raise ValueError("111")


# set up the robot
arm = XArmAPI(ip)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

arm.reset(wait=True)

# set the speed
speed = 30


# interpolate the data between these angles to get more data readings from the robot
# angles = [
#   [0, 14, -25, 13, 12.6, 0, 0],
#   [-14, 40, -73, 20, 33.4, -13.8, 0],
#   [21.9, 50, -80, 30, 36, 29, 0]]



#angles_new = []
#for angle in angles:
#   angle_line = [item*0.1 for item in angle]
#   angles_new.append(angle_line)

command_lines = np.rad2deg(command_lines)

# angles = command_lines
# angles = []
# print(command_lines[0][6])
# for i in range(len(command_lines)-1):
#     print(i)
#     angles[i] = [math.degrees(command_lines[i][0]), math.degrees(command_lines[i][1]), math.degrees(command_lines[i][2]), math.degrees(command_lines[i][3]), math.degrees(command_lines[i][4]), math.degrees(command_lines[i][5]), math.degrees(command_lines[i][6])]
# print(angles)

arm.set_pause_time(1)


record_positions = []
record_velocity = []
record_torque = []

code = arm.reset()
code = arm.set_servo_angle(angle=command_lines[0], speed=speed, radius=20, wait=False)

#arm.set_pause_time(5)
#arm.set_pause_time(5)
 
time.sleep(10)


q = []
dq = []
torque = []

for angle in command_lines:
    print(angle)
    code = arm.set_servo_angle(angle=angle, speed=speed, radius=20, wait=False)

    #print(angle)

    #sleep function for 0.01 s
    time.sleep(0.01)

    # function 
    code,[q,dq,torque]  = arm.get_joint_states()
    print("code: {0}".format(code))
    record_positions.append(np.rad2deg(q))
    record_velocity.append(dq)
    record_torque.append(torque)# how to add data found in one instance to whole csv?

#print(record_positions)
# csv file save

# writing measurements to file
print ("Writing measurements to file.")

output_path = "\XArmMeasurements_spt.csv"
with open(output_path, "a",newline = '') as csvfile:
    dict_writer = csv.DictWriter(
        csvfile,
        fieldnames = [f"q_{i}" for i in range (7)] 
        + [f"tau_{i}" for i in range(7)]
        )
    
    dict_writer.writeheader()

    for i in range(len(record_positions)):
        combined_dict = {}
        for j in range(7):
            combined_dict[f"q_{j}"] = record_positions[i][j]
            combined_dict[f"tau_{j}"] = record_torque[i][j]
        dict_writer.writerow(combined_dict)


 
    # combined_dict = {}
    # combined_dict.update({f"q_{i}": qi for i, qi in enumerate(q)})
    # combined_dict.update({f"dq_{i}": dqi for i, dqi in enumerate(dq)})
    # combined_dict.update({f"tau_{i}": tau_i for i, tau_i in enumerate(torque)})
    # dict_writer.writerow(combined_dict)

q.clear()
torque.clear()

print(f"Measurements saved to {output_path}")



arm.reset(wait=True)
arm.disconnect()