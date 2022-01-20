from re import M
import airsim
import cv2
import numpy as np
import os
import pprint
import setup_path 
import tempfile
import argparse
import math

client = airsim.MultirotorClient()
client.confirmConnection()

parser = argparse.ArgumentParser(description='Arguments for the flocking simulation.')
parser.add_argument('Number of drones', metavar='N', type=int)
parser.add_argument('X Arrival', metavar='X', type=int)
parser.add_argument('Y Arrival', metavar='Y', type=int)

args = vars(parser.parse_args())

# Initialisation
drones = {}
N = args["Number of drones"]
X = args["X Arrival"]
Y = args["Y Arrival"]
centroid_x, centroid_y = 0, 0
m_vel_x, m_vel_y = 0, 0

for i in range(0, N):
    name = "Drone" + str(i)
    client.enableApiControl(True, name)
    client.armDisarm(True, name)
    f = client.takeoffAsync(vehicle_name=name)
    f.join()
    state = client.getMultirotorState(vehicle_name=name)



    # Vérifier si centroid_x, centroid_y récupère bien les coordonnées et quel format
    centroid_x += state.gps_location.latitude
    centroid_y += state.gps_location.longitude

    print(state)

    v = [X-state.gps_location.latitude, Y-state.gps_location.longitude]
    norm = math.sqrt(v[0]**2 + v[1]**2)
    v_norm = [
            v[0] / norm,
            v[1] / norm
            ]

    m_vel_x += v_norm[0]
    m_vel_y += v_norm[1]

    drones[name] = [state, v_norm]

centroid = [centroid_x / N, centroid_y / N]
m_vel = [m_vel_x / N, m_vel_y / N]

print("################################################################")
print("################################################################")
print("################################################################")
print("Centroid : ", (centroid_x, centroid_y))


print("################################################################")
print("################################################################")
print("################################################################")
print("Mean Velocity : ", (m_vel[0], m_vel[1]))

k1 = 0.5
k2 = 0.3
k3 = 0.3

# check 2 si c'est suffisant et vérifier unité de mesure distance cm / m
c_iter = 0
#while (centroid[0] > X + 2 or centroid[0] < X - 2) and (centroid[1] > Y + 2 or centroid[1] < Y - 2):
while c_iter < 2:
    #print(c_iter)
    c_iter = c_iter + 1
    for name, val in drones.items():
        # Define forces
        state, vel = val[0], val[1]
        F1 = [k1*(centroid[0] - state.gps_location.latitude), k1*(centroid[1] - state.gps_location.longitude)] 
        F2 = [k2 * m_vel[0], k2 * m_vel[1]]

        print("################################################################")
        print("################################################################")
        print("################################################################")
        print("State : ", (state.gps_location.latitude, state.gps_location.longitude))
        
        print("################################################################")
        print("################################################################")
        print("################################################################")
        print("F1 : ", F1)

        print("################################################################")
        print("################################################################")
        print("################################################################")
        print("F1 : ", F2)

        rep = []
        rep_x, rep_y = 0, 0
        for i in range(0, N):
            if i != int(name[-1]):
                # Calculate distance between pairs of drones
                droneI = "Drone" + str(i)
                r = [
                    state.gps_location.latitude - drones[droneI][0].gps_location.latitude,
                    state.gps_location.longitude - drones[droneI][0].gps_location.longitude
                    ]

                print("################################################################")
                print("################################################################")
                print("################################################################")
                print("r : ", r)
                
                norm = math.sqrt(r[0]**2 + r[1]**2)                
                r_norm = [r[0] / norm, r[1] / norm]

                print("################################################################")
                print("################################################################")
                print("################################################################")
                print("r_norm : ", r_norm)
                
                F = [r[0] / r_norm[0]**3, r[1] / r_norm[1]**3]

                print("################################################################")
                print("################################################################")
                print("################################################################")
                print("F : ", F)
                
                rep.append(F)

                rep_x += F[0]
                rep_y += F[1]
        
        # Vérifier taille de rep (49 ou 50 élément)
        F3 = [k3 * rep_x / N, k3 * rep_y / N]

        print("################################################################")
        print("################################################################")
        print("################################################################")
        print("F3 : ", F3)
                

        new_coords = [
                        state.gps_location.latitude + F1[0] + F2[0] + F3[0],
                        state.gps_location.longitude + F1[1] + F2[1] + F3[1]
        ]

        # A voir ici avec vel, petit doute
        print("################################################################")
        print("################################################################")
        print("################################################################")
        print((new_coords[0], new_coords[1]))
        #f = client.moveToPositionAsync(new_coords[0], new_coords[1], state.gps_location.altitude, vel, vehicle_name=name)
        f = client.moveToPositionAsync(-5, 5, -10, 5, vehicle_name=name)
        f.join()







