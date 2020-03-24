<<<<<<< HEAD
from psychopy import visual, core, monitors, event
from psychopy.hardware import keyboard
import pandas as pd
import numpy as np

# TEST AREA

# test = Experiment(2, useDisplay=True)
# print(test.startTrial('first', False))
# print(test.startExperiment(True, True))

# with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(test.startExperiment(False))


# por_prime_df = df[:150]
# eng_prime_df = df[150:].reset_index(drop=True)

# por_cor = 0
# eng_cor = 0

# for i in range(150):
#     if por_prime_df['correct'][i] == True:
#         por_cor += 1
#     if eng_prime_df['correct'][i] == True:
#         eng_cor += 1

# print(por_cor/150, eng_cor/150)
# print(df.columns)

# df.to_csv(r'.\trials_data\subject-2.csv')
=======
from psychopy import visual, core, event, monitors #import some libraries from PsychoPy

from psychopy.preferences import Preferences
# Set preferences
prefs = Preferences()
prefs.hardware['audioLib'] = ['PTB']

from psychopy.sound.backend_ptb import SoundPTB

mysound = SoundPTB('A')

mysound.play()

>>>>>>> sa_igdevelop

