from psychopy import visual, core, monitors, event
from psychopy.sound import backend_ptb as sound
# from psychopy import sound
# from psychopy.sound.backend_sounddevice import SoundDeviceSound
from psychopy.hardware import keyboard
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

clock = core.Clock()

wrong_sound = sound.SoundPTB(volume=1.0)
# wrong_sound = SoundDeviceSound(value=r'.\incorrect.ogg', stereo=True)

# for i in range(1):
#     wrong_sound.play()
wrong_sound.play()