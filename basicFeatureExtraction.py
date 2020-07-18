import matplotlib as matplotlib
from pathlib import Path
import numpy, scipy, matplotlib.pyplot as plt, sklearn, urllib, IPython.display as ipd
import librosa, librosa.display
import stanford_mir; stanford_mir.init()


piano_signals = [
    librosa.load(p)[0] for p in Path().glob('fur_elise.mp3')
]

len(piano_signals)

plt.figure(figsize=(15, 6))
for i, x in enumerate(piano_signals):
    plt.subplot(2, 5, i+1)
    librosa.display.waveplot(x[:10000])
    plt.ylim(-1, 1)


