import librosa
import librosa.display
import numpy as np
import os

def convert_to_spectrogram(filepath, filedest, filename):
    filepath = os.path.abspath(filepath)
    y, sr = librosa.load(filepath, sr=44100)
    librosa.feature.melspectrogram(y=y, sr=sr)

    D = np.abs(librosa.stft(y, hop_length = 300))**2
    S = librosa.feature.melspectrogram(S=D)

    # Passing through arguments to the Mel filters
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256,
                                        fmax=8000)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4) , frameon=False)
    librosa.display.specshow(librosa.power_to_db(S,
                                                ref=np.max),
                            y_axis='mel', fmax=8000,
                            x_axis='time')

    # IDK how but removes extra padding and whitespace from the plot
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    savePath = os.path.abspath(os.path.join(filedest, filename.rsplit('.')[0]))
    plt.savefig(savePath)
    return savePath