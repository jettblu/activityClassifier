import csv
from pathlib import Path
from tkinter import filedialog
import numpy as np
from visuals import *


class Samples:
    samples = dict()

    def __init__(self, name, times, sampleRate, totalTime, roll, pitch, yaw, accelX, accelY, accelZ):
        self.name = name
        self.times = times
        self.sampleRate = sampleRate
        self.newSampleRate = None
        self.totaltime = totalTime
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.accelX = accelX
        self.accelY = accelY
        self.accelZ = accelZ
        BikingSamples.samples[name] = self

    def __repr__(self):
        return self.name


class BikingSamples(Samples):
    pass


class SquatSamples(Samples):
    pass


class RunningSamples(Samples):
    pass


class WalkingSamples(Samples):
    pass


class UpStairsSamples(Samples):
    pass


class DownStairsSamples(Samples):
    pass


class StandingSamples(Samples):
    pass


def getPaths():
    fileNames = []
    files = filedialog.askopenfilenames(title = "Choose CSV(s)",filetypes = (("CSV Files","*.csv"),))
    dirName = filedialog.askdirectory()
    for file in files:
        fileName = Path(file).stem
        fileNames.append(fileName)
    return files, fileNames, dirName


# transforms androsensor csv into clean output csv
def transformRawCsv(filePath, fileName, dirName):
    with open(filePath, newline='') as csvfile:
        dataReader = csv.reader(csvfile, delimiter=";")
        next(dataReader)
        # create a list of category labels
        categories = [cat for cat in next(dataReader)]
        # for cat in categories:
        #     print(cat)
        accelX = []
        accelY = []
        accelZ = []
        gyrX = []
        gyrY = []
        gyrZ = []
        times = []
        for row in dataReader:
            for i, value in enumerate(row):
                # print(f"{categories[i]}: {value}")
                if categories[i] == 'ACCELEROMETER X (m/sÂ²)':
                    accelX.append(value)
                if categories[i] == 'ACCELEROMETER Y (m/sÂ²)':
                    accelY.append(value)
                if categories[i] == 'ACCELEROMETER Z (m/sÂ²)':
                    accelZ.append(value)
                if categories[i] == "GYROSCOPE X (rad/s)":
                    gyrX.append(value)
                if categories[i] == "GYROSCOPE Y (rad/s)":
                    gyrY.append(value)
                if categories[i] == "GYROSCOPE Z (rad/s)":
                    gyrZ.append(value)
                if categories[i] == 'Time since start in ms ':
                    times.append(float(value))
    fields = ['time in milliseconds', 'acceleration_x', 'acceleration_y', 'acceleration_z', 'Roll', 'Pitch', 'Yaw']
    rows = [[times[i], accelX[i], accelY[i], accelZ[i], gyrZ[i], gyrY[i], gyrX[i]] for i in range(len(times))]
    transformedPath = dirName + f'/{fileName}.csv'
    # writing to csv file
    with open(transformedPath, 'w') as csvfile:
        # creating csv writer object
        csvwriter = csv.writer(csvfile)
        # writing fields
        csvwriter.writerow(fields)
        # writing data rows
        csvwriter.writerows(rows)


# extracts accel. and gyr. values from raw androsensor csv files
# outputs new csv files to specified folder
def csvTransformer():
    filePaths, fileNames, folderPath = getPaths()
    for i, path in enumerate(filePaths):
        print(f'Transforming {fileNames[i]}...')
        transformRawCsv(path, fileName=fileNames[i], dirName=folderPath)
    print('Transformation finished!')


# creates sample object from clean csv w/ optional visualizations
def readData(filePath, displayRaw=False, displayFFT=False):
    name = Path(filePath).stem
    print(f"Reading {name}...")
    with open(filePath, newline='') as csvfile:
        dataReader = csv.reader(csvfile, delimiter=",")
        # create a list of category labels
        categories = [cat for cat in next(dataReader)]
        # for cat in categories:
        #     print(cat)
        accelX = []
        accelY = []
        accelZ = []
        roll = []  # z value
        pitch = []  # y value
        yaw = []  # x value
        times = []
        for row in dataReader:
            for i, value in enumerate(row):
                # print(f"{categories[i]}: {value}")
                if categories[i] == 'acceleration_x':
                    accelX.append(value)
                if categories[i] == 'acceleration_y':
                    accelY.append(value)
                if categories[i] == 'acceleration_z':
                    accelZ.append(value)
                if categories[i] == "Yaw":
                    yaw.append(value)
                if categories[i] == "Pitch":
                    pitch.append(value)
                if categories[i] == "Roll":
                    roll.append(value)
                if categories[i] == 'time in milliseconds':
                    times.append(float(value))
        # total record time in seconds
        totalTime = (times[len(times)-1] - times[0])/1000
        sampleRate = len(times)/totalTime

    # converts lists to np array
    roll, pitch, yaw, accelX, accelY, accelZ = np.array(roll), np.array(pitch), np.array(yaw), \
                                               np.array(accelX), np.array(accelY), np.array(accelZ)

    # create objects based on sample type
    if 'biking' in name:
        sampleObj = BikingSamples(name, times, sampleRate, totalTime, roll, pitch, yaw, accelX, accelY, accelZ)
    if 'squats' in name:
        sampleObj = SquatSamples(name, times, sampleRate, totalTime, roll, pitch, yaw, accelX, accelY, accelZ)
    if 'running' in name:
        sampleObj = RunningSamples(name, times, sampleRate, totalTime, roll, pitch, yaw, accelX, accelY, accelZ)
    if 'standing' in name:
        sampleObj = StandingSamples(name, times, sampleRate, totalTime, roll, pitch, yaw, accelX, accelY, accelZ)
    if 'stairs down' in name:
        sampleObj = DownStairsSamples(name, times, sampleRate, totalTime, roll, pitch, yaw, accelX, accelY, accelZ)
    if 'stairs up' in name:
        sampleObj = UpStairsSamples(name, times, sampleRate, totalTime, roll, pitch, yaw, accelX, accelY, accelZ)
    if 'squats' in name:
        sampleObj = SquatSamples(name, times, sampleRate, totalTime, roll, pitch, yaw, accelX, accelY, accelZ)
    if 'walking' in name:
        sampleObj = WalkingSamples(name, times, sampleRate, totalTime, roll, pitch, yaw, accelX, accelY, accelZ)

    if displayRaw:
        print(f"{name} read. Creating raw visuals.")
        try:
            displayRawSample(sampleObj)
        except:
            print("Unable to produce raw visuals. Check sample object configuration.")

    if displayFFT:
        print(f"{name} read. Creating FFT visuals.")
        try:
            displayRawSample(sampleObj)
        except:
            print("Unable to produce FFT visuals. Check sample object configuration.")

    return sampleObj
