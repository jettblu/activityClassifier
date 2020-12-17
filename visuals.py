import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.fft import fft


def displayFFT(sampleObj):
    times = sampleObj.times
    accelX = sampleObj.accelX
    accelY = sampleObj.accelY
    accelZ = sampleObj.accelZ

    yaw = sampleObj.yaw
    pitch = sampleObj.pitch
    roll = sampleObj.roll

    # create FFts
    roll, pitch, yaw, accelX, accelY, accelZ = np.abs(fft(roll)), np.abs(fft(pitch)), \
                                               np.abs(fft(yaw)), np.abs(fft(accelX)), \
                                               np.abs(fft(accelY)), np.abs(fft(accelZ))

    fig = make_subplots(rows=3, cols=2)

    fig.append_trace(go.Scatter(
        x=times,
        y=accelX,
        name="Accel. X",
        xaxis="Time (milliseconds)",
        yaxis="Frequency (HZ)",
    ), row=1, col=1)

    fig.append_trace(go.Scatter(
        x=times,
        y=accelY,
        name="Accel. Y",
        xaxis="Time (milliseconds)",
        yaxis="Frequency (HZ)",
    ), row=2, col=1)

    fig.append_trace(go.Scatter(
        x=times,
        y=accelY,
        name="Accel. Y FFT",
        xaxis="Time (milliseconds)",
        yaxis="Frequency (HZ)",
    ), row=2, col=1)

    fig.append_trace(go.Scatter(
        x=times,
        y=accelZ,
        name="Accel. Z",
        xaxis="Time (milliseconds)",
        yaxis="Frequency (HZ)",
    ), row=3, col=1)

    fig.append_trace(go.Scatter(
        x=times,
        y=yaw,
        xaxis="Time (milliseconds)",
        yaxis="Frequency (HZ)",
        name="Gyr. X"
    ), row=1, col=2)

    fig.append_trace(go.Scatter(
        x=times,
        y=pitch,
        xaxis="Time (milliseconds)",
        yaxis="Frequency (HZ)",
        name="Gyr. Y"
    ), row=2, col=2)

    fig.append_trace(go.Scatter(
        x=times,
        y=roll,
        name="Gyr. Z",
        xaxis="Time (milliseconds)",
        yaxis="Frequency (HZ)",
    ), row=3, col=2)

    fig.update_layout(height=600, width=600, title_text=f"{sampleObj.name} FFTs")
    fig.show()


def displayRawSample(sampleObj):
    fig = make_subplots(rows=3, cols=2)
    sampleName = sampleObj.name
    times = sampleObj.times
    accelX = sampleObj.accelX
    accelY = sampleObj.accelY
    accelZ = sampleObj.accelZ
    roll = sampleObj.roll
    yaw = sampleObj.yaw
    pitch = sampleObj.pitch


    fig.append_trace(go.Scatter(
        x=times,
        y=accelX,
        name="Accel. X"
    ), row=1, col=1)

    fig.append_trace(go.Scatter(
        x=times,
        y=accelY,
        name="Accel. Y"
    ), row=2, col=1)

    fig.append_trace(go.Scatter(
        x=times,
        y=accelY,
        name="Accel. Y FFT"
    ), row=2, col=1)

    fig.append_trace(go.Scatter(
        x=times,
        y=accelZ,
        name="Accel. Z"
    ), row=3, col=1)

    fig.append_trace(go.Scatter(
        x=times,
        y=yaw,
        name="Gyr. X"
    ), row=1, col=2)

    fig.append_trace(go.Scatter(
        x=times,
        y=pitch,
        name="Gyr. Y"
    ), row=2, col=2)

    fig.append_trace(go.Scatter(
        x=times,
        y=roll,
        name="Gyr. Z"
    ), row=3, col=2)

    fig.update_layout(height=600, width=600, title_text=sampleName)
    fig.show()