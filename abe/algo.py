import pystoi

from scipy.io import wavfile
from IPython.display import Audio, display

# import osclea
import scipy.signal as signal
from scipy.signal import decimate
import numpy as np
import tensorflow as tf
from tensorflow import keras

# import shutil
import librosa
import sklearn
from scipy.signal import butter, filtfilt

# from pesq import pesq
from pystoi import stoi
from sklearn.model_selection import train_test_split


def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 8000
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = filtfilt(b, a, data)
    return y


def bwe(atekhz):

    cutoff = 4100
    order = 6
    fs = 16000
    frame_size = 512  # 2048  #512
    hop_length = frame_size // 2  # 512 #256
    mfccs = 100
    audio_framesx = []

    model = tf.keras.models.load_model("my_model")
    x, f = librosa.load(atekhz, sr=None)
    print("signal length: ", len(x))
    print("frequency is: ", f)
    Xm = []
    Xs = []
    lowed_signal = butter_lowpass_filter(x, cutoff, fs, order)
    mf = librosa.feature.mfcc(
        y=lowed_signal,
        sr=f,
        n_fft=frame_size,
        win_length=frame_size,
        hop_length=hop_length,
        n_mfcc=mfccs,
        window="hann",
        pad_mode="constant",
    )

    xstft = np.abs(
        librosa.stft(
            y=lowed_signal,
            n_fft=frame_size,
            hop_length=hop_length,
            win_length=frame_size,
            window="hann",
            pad_mode="constant",
        )
    )
    print("shape of mfcc matrix ", (mf.shape))
    print("shape of stft matrix ", (xstft.shape))

    xframewise = []

    for j in range(len(mf[0])):
        z = []
        for i in range(len(mf)):
            z.append(mf[i][j])
        Xm.append(z)

    for j in range(len(xstft[0])):
        z = []
        for i in range(len(xstft)):
            z.append(xstft[i][j])
        Xs.append(z)

    print(np.array(Xm).shape)
    print(np.array(Xs).shape)

    y_pred = model.predict([np.array(Xm), np.array(Xs)])
    yFormat = []

    for j in range(len(y_pred[0])):
        z = []
        for i in range(len(y_pred)):
            z.append(y_pred[i][j])
        yFormat.append(z)
    toplay = []
    final_aud = librosa.griffinlim(np.array(yFormat), hop_length=hop_length)
    for i in final_aud:
        toplay.append(i)

    toplay = np.array(toplay)
    minlen = min(len(x), len(toplay))
    # score = pesq(fs, x[:minlen], toplay[:minlen], "wb")
    stoi_score = stoi(toplay[:minlen], x[:minlen], fs, extended=False)
    Audio(toplay, rate=fs)
    return toplay, stoi_score
    # return 0, 0
