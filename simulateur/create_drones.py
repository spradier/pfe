import argparse
import json

parser = argparse.ArgumentParser(description='Arguments for the generation.')

parser.add_argument('Number of drones', metavar='N', type=int)
parser.add_argument('--Attribute', metavar='A', type=str)
parser.add_argument("--Coordinates", metavar="C", type=str)

args = vars(parser.parse_args())

vehicles = {}
N = args["Number of drones"]
C = [0, 0, 0]
for i in range(N):
    vehicles["Drone"+str(i)] = {"VehicleType" : "SimpleFlight", 
                                "X" : C[0], 
                                "Y" : C[1], 
                                "Z" : C[2]
                                }

final_dict = {"SettingsVersion": 1.2,
            "SimMode": "Multirotor",
            "Vehicles": vehicles
            }

with open('data.json', 'w') as outfile:
    json.dump(final_dict, outfile, indent=4)