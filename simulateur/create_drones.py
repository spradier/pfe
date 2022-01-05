import argparse
import json

parser = argparse.ArgumentParser(description='Arguments for the generation.')

parser.add_argument('Number of drones', metavar='N', type=int)
parser.add_argument('--Attribute', metavar='A', type=str)
parser.add_argument("--Coordinates", metavar="C", type=str)

args = vars(parser.parse_args())

vehicles = {}
sensors = {}
N = args["Number of drones"]
C = [0, 0, 0]
for i in range(N):
    vehicles["Drone"+str(i)] = {"VehicleType" : "SimpleFlight", 
                                "X" : C[0], 
                                "Y" : C[1], 
                                "Z" : C[2],
                                "Cameras": {},
                                }




cameraDefault = {"CaptureSettings": [
        {
            "Width": 640,
            "Height": 480,
            "FOV_Degrees": 360
        }
    ],
    "NoiseSettings":[
        {
            "Enabled": "true",
            "HorzDistortionContrib": 1.0,
            "HorzDistortionStrength": 0.002
        }
    ],
}

final_dict = {"SettingsVersion": 1.2,
            "SimMode": "Multirotor",
            "CameraDirector": {
                "X": 1, "Y": 1, "Z": 1 
            },
            "Vehicles": vehicles,
            "CameraDefaults": cameraDefault
            }


with open('data.json', 'w') as outfile:
    json.dump(final_dict, outfile, indent=4)
