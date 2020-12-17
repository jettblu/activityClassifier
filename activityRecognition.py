import loadCSV
# system libraries
import joblib
from os import listdir
from os.path import isfile, join
from collections import Counter
# data packages
from scipy.stats import kurtosis, skew, pearsonr
import numpy as np
import plotly.graph_objects as go

# signal processing libraries
from sklearn import decomposition
from sklearn.preprocessing import *
from scipy import signal
# visualization libraries
from plotmesh import plot_decision_boundaries
import seaborn as sns
import matplotlib.pyplot as plt
# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
# imbalanced learn libraries
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold


# trims the beginning and end of sample to avoid pre/post activity noise
# trimAmount is in seconds
def trimData(sampleObj, trimAmount):
    # converts trim amount to milliseconds
    trimAmount = trimAmount
    sampleRate = sampleObj.sampleRate
    # number of points to trim from beginning and end
    numTrimPoints = int(trimAmount*sampleRate)

    adjustedYawLength = len(sampleObj.yaw)-numTrimPoints
    sampleObj.times = sampleObj.times[numTrimPoints:adjustedYawLength]
    sampleObj.accelX = sampleObj.accelX[numTrimPoints:len(sampleObj.yaw)-numTrimPoints]
    sampleObj.accelY = sampleObj.accelY[numTrimPoints:len(sampleObj.yaw)-numTrimPoints]
    sampleObj.accelZ = sampleObj.accelZ[numTrimPoints:len(sampleObj.yaw)-numTrimPoints]
    sampleObj.roll = sampleObj.roll[numTrimPoints:len(sampleObj.yaw)-numTrimPoints]
    sampleObj.pitch = sampleObj.pitch[numTrimPoints:len(sampleObj.yaw) - numTrimPoints]
    sampleObj.yaw = sampleObj.yaw[numTrimPoints:len(sampleObj.yaw)-numTrimPoints]
    sampleObj.totalTime = (sampleObj.times[len(sampleObj.times) - 1] - sampleObj.times[0]) / 1000


def resampleData(activityObject, newSampleRate):
    newSampleRate = newSampleRate
    activityObject.newSampleRate = newSampleRate
    # change from sample rate to number of data points to sample
    newSampleRate *= int(activityObject.totaltime)
    activityObject.accelX = signal.resample(activityObject.accelX, newSampleRate)
    activityObject.accelY = signal.resample(activityObject.accelY, newSampleRate)
    activityObject.accelZ = signal.resample(activityObject.accelZ, newSampleRate)
    activityObject.roll = signal.resample(activityObject.roll, newSampleRate)
    activityObject.pitch = signal.resample(activityObject.pitch, newSampleRate)
    activityObject.yaw = signal.resample(activityObject.yaw, newSampleRate)


# splits single audio sample into multiple samples of window size
def windowData(rawData,  windowSize):
    return [rawData[x:x + windowSize] for x in range(0, len(rawData), windowSize)]


def preprocess(activityObject, trimAmount, newSampleRate):
    resampleData(activityObject=activityObject, newSampleRate=newSampleRate)
    trimData(activityObject, trimAmount)

    activityObject.accelX = windowData(activityObject.accelX, activityObject.newSampleRate)
    activityObject.accelY = windowData(activityObject.accelY, activityObject.newSampleRate)
    activityObject.accelZ = windowData(activityObject.accelZ, activityObject.newSampleRate)
    activityObject.pitch = windowData(activityObject.pitch, activityObject.newSampleRate)
    activityObject.roll = windowData(activityObject.roll, activityObject.newSampleRate)
    activityObject.yaw = windowData(activityObject.yaw, activityObject.newSampleRate)


# returns feature vector for activity object
def getFeatureVector(activityObject):
    preprocess(activityObject, 3, 50)
    # container for windowed features
    splitFeatures = []
    for i in range(len(activityObject.accelX)):
        accelX = activityObject.accelX[i]
        accelY = activityObject.accelY[i]
        accelZ = activityObject.accelZ[i]

        yaw = activityObject.yaw[i]
        pitch = activityObject.pitch[i]
        roll = activityObject.roll[i]

        componenets = [accelX, accelY, accelZ, yaw, pitch, roll]

        featList1 = []

        fnList1 = [
            np.std,
            np.mean,
            kurtosis,
            skew
        ]

        # applies function from func list to each accel/ gyr. component of data
        for funct in fnList1:
            for component in componenets:
                featList1.append(funct(component))

        corrList = [
            pearsonr(accelX, accelZ)[1],
            pearsonr(pitch, roll)[1]
        ]

        featVec = featList1 + corrList
        splitFeatures.append(featVec)

    return splitFeatures


def loadBikingData(isPredict=False):
    sampleObjs = []
    bikingPath = 'biking/'
    # creates list of all files in directory
    bikingFiles = [f for f in listdir(bikingPath) if isfile(join(bikingPath, f))]
    fv = []
    for file in bikingFiles:
        # return object with sample attributes
        sampleObj = loadCSV.readData(bikingPath + file)
        # returns data from sample object split into windowed features
        fVec = getFeatureVector(sampleObj)
        sampleObjs.append(sampleObj)
        fv.append(fVec)
    if isPredict:
        return fv, sampleObjs
    return fv


def loadRunningData(isPredict=False):
    sampleObjs = []
    runningPath = 'running/'
    # creates list of all files in directory
    runningFiles = [f for f in listdir(runningPath) if isfile(join(runningPath, f))]
    fv = []
    for file in runningFiles:
        sampleObj = loadCSV.readData(runningPath + file)
        fVec = getFeatureVector(sampleObj)
        sampleObjs.append(sampleObj)
        fv.append(fVec)
    if isPredict:
        return fv, sampleObjs
    return fv


def loadStairsUpData(isPredict=False):
    sampleObjs = []
    stairUpPath = 'stairs up/'
    # creates list of all files in directory
    stairUpFiles = [f for f in listdir(stairUpPath) if isfile(join(stairUpPath, f))]
    fv = []
    for file in stairUpFiles:
        sampleObj = loadCSV.readData(stairUpPath + file)
        fVec = getFeatureVector(sampleObj)
        sampleObjs.append(sampleObj)
        fv.append(fVec)
    if isPredict:
        return fv, sampleObjs
    return fv


def loadStairsDownData(isPredict=False):
    sampleObjs = []
    stairsDownPath = 'stairs down/'
    # creates list of all files in directory
    stairUpFiles = [f for f in listdir(stairsDownPath) if isfile(join(stairsDownPath, f))]
    fv = []
    for file in stairUpFiles:
        sampleObj = loadCSV.readData(stairsDownPath + file)
        fVec = getFeatureVector(sampleObj)
        sampleObjs.append(sampleObj)
        fv.append(fVec)
    if isPredict:
        return fv, sampleObjs
    return fv


def loadStandingData(isPredict=False):
    sampleObjs = []
    standingPath = 'standing/'
    # creates list of all files in directory
    standingFiles = [f for f in listdir(standingPath) if isfile(join(standingPath, f))]
    fv = []
    for file in standingFiles:
        # return object with sample attributes
        sampleObj = loadCSV.readData(standingPath + file)
        # returns data from sample object split into windowed features
        fVec = getFeatureVector(sampleObj)
        sampleObjs.append(sampleObj)
        fv.append(fVec)
    if isPredict:
        return fv, sampleObjs
    return fv


def loadSquatsData(isPredict=False):
    sampleObjs = []
    squatsPath = 'squats/'
    # creates list of all files in directory
    squatsFiles = [f for f in listdir(squatsPath) if isfile(join(squatsPath, f))]
    fv = []
    for file in squatsFiles:
        sampleObj = loadCSV.readData(squatsPath + file)
        sampleObjs.append(sampleObj)
        fVec = getFeatureVector(sampleObj)
        fv.append(fVec)
    if isPredict:
        return fv, sampleObjs
    return fv


def loadWalkingData(isPredict=False):
    sampleObjs = []
    walkingPath = 'walking/'
    # creates list of all files in directory
    walkingFiles = [f for f in listdir(walkingPath) if isfile(join(walkingPath, f))]
    fv = []
    for file in walkingFiles:
        sampleObj = loadCSV.readData(walkingPath + file)
        sampleObjs.append(sampleObj)
        fVec = getFeatureVector(sampleObj)
        fv.append(fVec)
    if isPredict:
        return fv, sampleObjs
    return fv


# establish labels and combined data set of all activities
def setLabelsAndData(useStored=False, store=True, predict=False):
    # use stored features if specified
    if useStored:
        data = np.load('data.npy', allow_pickle=True)
        labels = np.load('labels.npy', allow_pickle=True)
        return data, labels
    # load raw audio data
    biking = loadBikingData()
    running = loadRunningData()
    squats = loadSquatsData()
    stairsDown = loadStairsDownData()
    stairsUp = loadStairsUpData()
    walking = loadWalkingData()
    standing = loadStandingData()
    activities = [biking, running, squats, standing, stairsDown, stairsUp, walking]
    labels = []
    data = []
    # store features for quicker testing if specified
    for i, activity in enumerate(activities):
        for session in activity:
            labels.extend([i]*len(session))
            # add each sample from of each sound to collective data set
            for sample in session:
                data.append(sample)
    if store:
        np.save('data.npy', np.array(data))
        np.save('labels.npy', np.array(labels))
    return np.array(data), np.array(labels)


# scales and reduces dimensionality of feature vectors
def normalizeFeatures(data, visualize=False, isPredict=False):
    if isPredict:
        maxAbs = joblib.load('normalizers/maxAbs.pkl')
        pca = joblib.load('normalizers/pca.pkl')
        data = maxAbs.transform(data)
        data = pca.transform(data)
        return data
    # scales data
    maxAbs = MaxAbsScaler().fit(data)
    joblib.dump(maxAbs, 'normalizers/maxAbs.pkl')
    data = maxAbs.transform(data)
    # applies principle component analysis
    pca = decomposition.PCA(n_components=5)
    pca.fit(data)
    joblib.dump(pca, f'normalizers/pca.pkl')
    data = pca.transform(data)
    # visualizes scaled feature spread
    if visualize:
        for i in range(data.shape[1]):
            sns.kdeplot(data[:, i])
        plt.show()
    return data


# calculates the mean accuracy for a given classifier over a number of trials
def getAccuracy(classifier, data, labels):
    print(f'Testing {classifier}')
    testScores = []
    # split data using stratified K fold which accounts for class imbalances
    cv = StratifiedKFold(n_splits=10, random_state=65, shuffle=True)
    # make predictions
    for train_index, test_index in cv.split(data, labels):
        dataTrain, dataTest, labelsTrain, labelsTest = data[train_index], data[test_index], labels[train_index], labels[test_index]
        print(Counter(labelsTrain))
        sm = SMOTE(random_state=2)
        dataTrain, labelsTrain = sm.fit_sample(dataTrain, labelsTrain)
        print(Counter(labelsTrain))
        # under = RandomUnderSampler(sampling_strategy=.5)
        # dataTrain, labelsTrain = under.fit_sample(dataTrain, labelsTrain)
        classifier.fit(dataTrain, labelsTrain)
        # create confusion matrix
        testScores.append(classifier.score(dataTest, labelsTest))
    joblib.dump(classifier, f'classifiers/{str(classifier)}.pkl')
    return np.mean(testScores)


# returns the accuracy for a series of classifiers
def classify(useStored=True, store=True, visualize=False):
    data, labels = setLabelsAndData(useStored=useStored, store=store)
    data = normalizeFeatures(data, visualize=True)
    print('normalized')
    clfs = [RandomForestClassifier(n_estimators=300), KNeighborsClassifier(n_neighbors=10), SVC(kernel="rbf"),
            GradientBoostingClassifier(n_estimators=150)]
    # initializes dictionary that will contain classifier as a key and accuracy as a value
    accuracies = dict()
    if visualize:
        meshPlots(data, labels)
    # retrieves accuracy of each classifier
    for clf in clfs:
        accuracy = getAccuracy(clf, data, labels)
        accuracies[str(clf)] = accuracy
    cumulative = np.mean(list(accuracies.values()))
    print(f"\tFINAL ACCURACY\nAchieved using ensemble of algorithms\nMean Accuracy: "
          f"{cumulative}\n"
          f"You'll find accuracies produced from classifier tests below\n\t------------")

    return accuracies


def meshPlots(data, labels):
    # create mesh plots for first two features with given classifiers
    plt.figure()
    plt.title("random Forest")
    plot_decision_boundaries(data, labels, RandomForestClassifier, n_estimators=300)
    plt.show()
    plt.figure()
    plt.title("SVC")
    plot_decision_boundaries(data, labels, SVC, kernel="rbf")
    plt.show()
    plt.figure()
    plt.title("Nearest Neighbors")
    plot_decision_boundaries(data, labels, KNeighborsClassifier, n_neighbors=2)
    plt.show()


def makePredictions(data):
    data = normalizeFeatures(data, visualize=False, isPredict=True)
    classes = {0: 'biking', 1: 'running', 2: 'squats', 3: 'standing', 4: 'stairsDown', 5: 'stairsUp', 6: 'walking'}
    classifiers = []
    for fileName in listdir('classifiers'):
        classifiers.append(joblib.load(f'classifiers/{fileName}'))
    predictions = []
    for i, timePoint in enumerate(data):
        print(f'Predicting second {i} of sample.')
        for classifier in classifiers:
            predictions.append(classifier.predict([timePoint])[0])

    votingCount = {}
    # record votes
    for label in predictions:
        votingCount[label] = votingCount.get(label, 0)+1
    # tally votes
    prediction = classes[max(votingCount, key=votingCount.get)]
    # visualizePrediction(predictions=predictions, prediction=prediction)
    return predictions, prediction


'''Intended to create a more comprehensive visual with distinctions between classes, but was in a time crunch.'''


def visualizePrediction(predictions, prediction):
    # establishes times where time length corresponds to window
    times = [i for i in range(len(predictions))]
    plt.title(f'{prediction} Decisions')
    patches = []
    plt.plot(predictions, 'ro')
    traces = []
    # for i, label in enumerate(diarizedDict):
    #     color = randomColor()
    #     patches.append(mpatches.Patch(color=color, label=f'Spaker {i}'))
    #     for start, end in times:
    #         plt.axvspan(start, end + 1, color=color, alpha=0.5)
    #     plt.legend(handles=patches)
    plt.show()


for test in loadWalkingData():
    print(makePredictions(test))


print(classify(useStored=True, store=False, visualize=True))
