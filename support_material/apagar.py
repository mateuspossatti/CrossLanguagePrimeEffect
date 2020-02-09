from psychopy import visual, core, event, monitors #import some libraries from PsychoPy


# #create a window
# mywin = visual.Window([800,600],monitor="testMonitor", units="deg")

# #create some stimuli
# grating = visual.GratingStim(win=mywin, mask='circle', size=3, pos=[-4,0], sf=3)
# fixation = visual.GratingStim(win=mywin, size=0.2, pos=[0,0], sf=0, rgb=-1)

# #draw the stimuli and update the window
# while True: #this creates a never-ending loop
#     grating.setPhase(0.05, '+')#advance phase by 0.05 of a cycle
#     grating.draw()
#     fixation.draw()
#     mywin.flip()

#     if len(event.getKeys())>0:
#         break
#     event.clearEvents()

# #cleanup
# mywin.close()
# core.quit()

# mon = monitors.Monitor('SurfaceBook2')
import json

with open(r'.\support_material\monitor_settings.json', 'r') as monSet:
    monDict = json.load(monSet)
    print(monDict)

def set_monitor():
    # Load monitor settings
    name = monDict['monitor_name']
    width = monDict['monitor_width']
    resol = monDict['monitor_resolution']
    mon = monitors.Monitor(name=name, width=width)
    mon.setSizePix(resol)
    return mon

mon = set_monitor()

# win = visual.Window([1500, 1000], monitor=mon, fullscreen=True)

# text = visual.TextStim(win, text='a', units='norm', height=3)

text.autoDraw = True

win.flip()
text.draw()
core.wait(3)


