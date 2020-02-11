from psychopy import visual, core, event, monitors #import some libraries from PsychoPy

from psychopy.preferences import Preferences
# Set preferences
prefs = Preferences()
prefs.hardware['audioLib'] = ['PTB']

from psychopy.sound.backend_ptb import SoundPTB

mysound = SoundPTB('A')

mysound.play()


