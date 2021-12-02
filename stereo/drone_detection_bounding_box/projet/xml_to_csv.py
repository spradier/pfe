from os import walk
import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join
import csv

mypath = 'dataset/annotation/'
filenames = next(walk(mypath), (None, None, []))[2]  # [] if no file

# open the file in the write mode
f = open('dataset/annotation.csv', 'w', encoding='UTF8', newline='')

# create the csv writer
writer = csv.writer(f)

for file in filenames:
    filename = mypath + file 
    tree = ET.parse(filename)
    root = tree.getroot()
    name = root.find("filename").text
    labels = [root[6][4][0].text, root[6][4][1].text, root[6][4][2].text, root[6][4][3].text]

    cell = name + "," + labels[0] + "," + labels[1] + "," + labels[2] + "," + labels[3]  
    #write a row to the csv file
    print(cell)
    val = [cell]
    writer.writerow(val)    

# close the file
f.close()
