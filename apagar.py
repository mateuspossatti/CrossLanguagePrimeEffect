from psychopy import visual, core, monitors, event
from psychopy.hardware import keyboard
import pandas as pd
import numpy as np

# kb = keyboard.Keyboard()
# kb.getKeys(('z', 'm'), waitRelease=False)
mon = monitors.Monitor('SurfaceBook2')
mon.setSizePix((1000, 1500))

win = visual.Window(monitor=mon, units='cm', viewScale=None, fullscr=False)
ab = visual.Line(win, start=(1, 0), end=(2, 0))
fixation = visual.ShapeStim(win, 
                vertices=((0, -.5), (0, .5), (0,0), (-.5,0), (.5, 0)),
                lineWidth=5,
                closeShape=False,
                lineColor="white",
                units='cm'
            )


while True:
    fixation.draw()
    win.flip()
    k = input()
    if k:
        break

# target = visual.TextStim(win=win)
# target.text = 'a'
# kb.clock.reset()
# target.draw()
# event.waitKeys(keyList=('z', 'm'))

