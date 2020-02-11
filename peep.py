import subprocess as sb

try:
    from psychopy import visual, core, monitors, event, clock
    from matplotlib import pyplot as plt
    from playsound import playsound
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import json
except ModuleNotFoundError:
    sb.run('pip3 install -r requirements.txt', shell=True)

import keyboard_mod as keyboard

class Experiment(object):
    def __init__(self, n=None, mask_case='upper', pairs_n=50, fullcross=True, conditions_n=3, mask_size=8, onelanguageorder=None,
    fullscreen=False, timeparadigm=None, kb_keys=None, save=None, practiceLeng=50):
        """:Parameters:
        fullcross: will ditermine if the effect will be studied in the two ways.
        n: the number of the subject, will ditermine the sequence of trial and the language of intructions.
        mask_case: The case of the mask's letters
        pairs_n: The number of prime-target pairs to use in the study, it have to be a even number.
        conditions_n: The number of conditions that the subject will be tested
        mask_size: The number of letter that the masks will have
        onelanguageorder: Only if fullcross will be False
        timeparadigm: The correct is to be set to None, but you can use a dictionary of 4 key-values pairs.
        practiceLeng: The number of trials in the practice.
        """

# LOAD THE JSON FILE WITH THE MONITOR SETTINGS
        monitorSets = open(r'.\support_material\monitor_settings.json', 'r')
        self.monDict = json.load(monitorSets)

        # If there is no settings, a function will be executed that will question the user to set the settings
        if self.monDict['monitor_name'] is None:
            name, width, resol, freq = self.define_mon_settings()

            self.monDict['monitor_name'] = name
            self.monDict['monitor_width'] = width
            self.monDict['monitor_resolution'] = resol
            self.monDict['monitor_frequency'] = freq

            with open(r'.\support_material\monitor_settings.json', 'w') as monitorSets:
                json.dump(self.monDict, monitorSets)

# QUESTION THE USER ABOUT THE VOLUNTEER NUMBER IF IT ISN'T ALREADY DECLARE
        if n is None:
            while True:
                try:
                    n = int(input('Please enter the number of the volunteer: '))
                    break

                except ValueError:
                    print("Oops!  That was no valid number.  Try again...")

# DEFINE ATTRIBUTES BASED ON CODITIONAL STATEMENTS
        if mask_case == 'upper':
            self.mask_char = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        elif mask_case == 'lower':
            self.mask_char = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        if fullcross == True:
            self.totaltrials_n = pairs_n * conditions_n * 2
        elif fullcross == False:
            self.totaltrials_n = pairs_n * conditions_n
        if onelanguageorder != None:
            self.onelanguageorder = onelanguageorder
        if timeparadigm == None:
            self.timeparadigm = {'fixation' : 700, 'back_mask' : 100, 'prime' : 50, 'forward_mask' : 50}
        else:
            self.timeparadigm = timeparadigm
        if kb_keys == None:
            self.kb_keys = ('z', 'm')
        else:
            self.kb_keys = kb_keys

# DEFINE THE OTHER ATTRIBUTES
        self.conditions = ['congruent', 'incongruent', 'control']
        self.subject_n = n
        self.pairs_n = pairs_n
        self.language_n = pairs_n * conditions_n
        self.fullcross = fullcross
        self.mask_size = mask_size
        self.fullscreen = fullscreen
        self.screen_hz = self.monDict['monitor_frequency']
        self.practiceLeng = practiceLeng

# CREATE A GLOBAL KEY EVENT TO QUIT THE PROGRAM
        # Determine key and modifires
        key = 'q'
        modifiers = ['ctrl']

        # Create global key event
        event.globalKeys.add(key=key, modifiers=modifiers, func=core.quit)

# DETERMINE LANGUAGE ORDER FOR THE ACTUAL SUBJECT
        def subject_experiment_order():
            left, right = self.kb_keys
            if fullcross == True:
                if n % 2 == 0:
                    language_order = 'PorEng-EngPor'
                    kb_key_response = {'concrete' : right, 'abstract' : left}
                else:
                    language_order = 'EngPor-PorEng'
                    kb_key_response = {'concrete' : left, 'abstract' : right}

            elif fullcross == False:
                language_order = self.onelanguageorder
                if n % 2 == 0:
                    kb_key_response = {'concrete' : left, 'abstract' : right}
                else:
                    kb_key_response = {'concrete' : right, 'abstract' : left}

            return language_order, kb_key_response

        self.language_order, self.kb_key_response = subject_experiment_order() 

        self.firstLang = self.language_order.split('-')[0]
        self.secondLang = self.language_order.split('-')[1]

# DETERMINE FRAME DURATION:
        def frame_duration():
            ms_paradigm = self.timeparadigm
            screen_hz = self.screen_hz
            ms_frame = 1000 / screen_hz
            frame_paradigm = {'fixation' : np.round((ms_paradigm['fixation'] / ms_frame)), 'back_mask' : np.round((ms_paradigm['back_mask'] / ms_frame)), 
            'prime' : np.round((ms_paradigm['prime'] / ms_frame)), 'forward_mask' : np.round((ms_paradigm['forward_mask'] / ms_frame))}
            return frame_paradigm

        self.frame_paradigm = frame_duration()

# GENERATE CLOCK, KEYBOARD AND MONITOR
        def clock_generator():
            monitorclock = clock.Clock()
            return monitorclock

        self.monitorclock = clock_generator()

        def hardware_generator():
            kb = keyboard.Keyboard(waitForStart=True)

            return kb

        self.kb = hardware_generator()

        self.mon = self.set_monitor()

# DEFINE WORDS SEQUENCE
        def words_sequence():
            # LOAD CSV
            words_df = pd.read_csv(r'.\support_material\words.csv').columns
            col_names = {}
            for col in words_df:
                col_names[col] = str
            words_df = pd.read_csv(r'.\support_material\words.csv', dtype=col_names)

            # CREATE KEY RESPONSE LIST
            obj, nobj = tuple(self.kb_key_response.values())
            key_response_sequence = []
            for _ in range(conditions_n):
                for _ in range(int(self.pairs_n / 2)):
                    key_response_sequence.append(obj)
                for _ in range(int(self.pairs_n / 2)):
                    key_response_sequence.append(nobj)

            # CREATE INDEX
            index_f = list(np.random.choice(np.arange(self.language_n), replace=False, size=self.language_n))
            index_s = list(np.random.choice(np.arange(self.language_n), replace=False, size=self.language_n))

            # CREATE CLASSES
            class_list = []
            for _ in range(self.pairs_n):
                class_list.append(self.conditions[0])
            for _ in range(self.pairs_n):
                class_list.append(self.conditions[1])
            for _ in range(self.pairs_n):
                class_list.append(self.conditions[2])

            # CREATE DATAFRAMES
            def create_incong_df():
                eng_por = words_df.loc[:, ['Portuguese', 'English']]
                end_a, end_b = np.array((int(self.pairs_n / 2), self.pairs_n)) - 1
                new_incong_ob = [eng_por['English'][end_a]] + list(eng_por['English'][:end_a].values) + \
                    [eng_por['English'][end_b]] + list(eng_por['English'][(end_a + 1):end_b])
                eng_por['English'] = new_incong_ob

                return eng_por 

            cong_df = words_df[['Portuguese', 'English']]
            incong_df = create_incong_df()
            control_df = words_df

            if fullcross:
                # LANGUAGE ORDER
                lo = self.language_order
                if lo == 'PorEng-EngPor':
                    first = ['Portuguese', 'English', 'PseudoPor']
                    second = ['English', 'Portuguese', 'PseudoEng']
                else:
                    first = ['English', 'Portuguese', 'PseudoEng']
                    second = ['Portuguese', 'English', 'PseudoPor']

                # FIRST TRIAL
                # PRIME SEQUENCE
                prime_f = cong_df[first[0]].append(incong_df[first[0]].append(control_df[first[2]])).reset_index(drop=True)

                # Put the prime in UPPER CASE
                prime_f_UC = []
                for i in range(prime_f.shape[0]):
                    word = prime_f[i].upper()
                    prime_f_UC.append(word)

                prime_f = prime_f_UC

                # TARGET SEQUENCE
                target_f = cong_df[first[1]].append(incong_df[first[1]].append(control_df[first[1]])).reset_index(drop=True)

                # PRIME-TARGET DATA FRAME
                # print(key_response_sequence, class_list)
                prime_target_first = pd.DataFrame(data={
                    'prime_{}'.format(first[0][:3].lower()) : prime_f,
                    'target_{}'.format(first[1][:3].lower()) : target_f,
                    'correct_response' : key_response_sequence,
                    'class' : class_list,
                    'original_index' : [i for i in range(self.language_n)]
                    })

                # SECOND TRIAL
                # PRIME SEQUENCE
                prime_s = cong_df[second[0]].append(incong_df[second[0]].append(control_df[second[2]])).reset_index(drop=True)

                # Put the prime in UPPER CASE
                prime_s_UC = []
                for i in range(prime_s.shape[0]):
                    word = prime_s[i].upper()
                    prime_s_UC.append(word)

                prime_s = prime_s_UC

                # TARGET SEQUENCE
                target_s = cong_df[second[1]].append(incong_df[second[1]].append(control_df[second[1]])).reset_index(drop=True)

                # PRIME-TARGET DATA FRAME
                prime_target_second = pd.DataFrame(data={
                    'prime_{}'.format(second[0][:3].lower()) : prime_s,
                    'target_{}'.format(second[1][:3].lower()) : target_s,
                    'correct_response' : key_response_sequence,
                    'class' : class_list,
                    'original_index' : [i for i in range(self.language_n)]
                    })

                # prime_target_second = None

                return prime_target_first.reindex(index_f).reset_index(drop=True), prime_target_second.reindex(index_s).reset_index(drop=True)

            else:
                lo = self.onelanguageorder
                if lo == 'PorEng':
                    order = ['Portuguese', 'English', 'PseudoPor']
                elif lo == 'EngPor':
                    order = ['English', 'Portuguese', 'PseudoEng']
                else:
                    raise Exception('The language format used is not supported. Please do the correction one the "onelanguageorder" argument. \n onelanguageorder = {}'.format(self.onelanguageorder))

                # TRIAL
                # PRIME SEQUENCE
                prime = list(cong_df[order[0]].append(incong_df[order[0]].append(control_df[order[2]])))

                # TARGET SEQUENCE
                target = list(cong_df[order[1]].append(incong_df[order[1]].append(control_df[order[1]])))

                # PRIME-TARGET DATA FRAME
                prime_target = pd.DataFrame(data={
                    'prime_{}'.format(order[0][:3].lower()) : prime,
                    'target_{}'.format(order[1][:3].lower()) : target,
                    'correct_response' : key_response_sequence,
                    'class' : class_list,
                    'index' : index_f
                    }, index=index_f).reset_index(drop=True)

                return prime_target, None

        self.first_sequence, self.second_sequence = words_sequence()

        self.startPractice(False)

# QUESTION THE USER IF HIS WANT TO START THE EXPERIMENT
        while True:
            startexp = str(input('Do you want to begin the expriment?\n(y/n): ')).lower()
            if startexp != 'y' and startexp != 'n':
                print('The command typed ("{}") is invalid, please type "y" to begin the experiment\nor "n" to continue without begin the experiment.'.format(startexp))
            
            # THE USER WANT TO BEGIN THE EXPERIMENT
            elif startexp == 'y':

                # QUESTION THE USER ABOUT HIS FULLSCREEN PREFERENCE
                if fullscreen is None:
                    while True:
                        fs = str(input('Do you want to run the experiment in fullscreen?\n(y/n): ')).lower()
                        if fs != 'y' and fs != 'n':
                            print('The command typed ("{}") is invalid, please type "y" to make the experiment fullscreen\nor "n" to not make the experiment fullscreen'.format(fs))
                            pass
                        elif fs == 'y':
                            self.fullscreen = True
                            break
                        else:
                            self.fullscreen = False
                            break

                self.data_trial_final = self.startExperiment(save=save)
                return

            # THE USER DON'T WANT TO BEGIN THE EXPERIMENT
            else:
                try:
                    self.win.close()
                    return
                except AttributeError:
                    return

############# END OF __INIT__() #################

    def define_mon_settings(self):
        print("""Unfortunately, you don't have set your monitor settings yet.\nPlease, answer the next 4 questions to configure your monitor correctly""")
        # Question the name of the monitor
        def question_mon_name():
            # Question about the name of the monitor:
            name = str(input('What name do you want to give to your monitor? ')) 

            # Confirm with the user if he really want to use the name typed:
            while True:
                confirm_name = str(input('Are you certain that you want to use "{name}" as the name of your monitor?\n (y/n): '.format(name=name))).lower()

                # Command typed is invalid:
                if confirm_name != 'y' and confirm_name != 'n':
                    print("""The command typed "{confirm_name}" is invalid, please type "y" to use "{name}" as your monitor's name\nor type "n" to choose a new name.""".format(name=name, confirm_name=confirm_name))
                
                # The user want to choose another name
                elif confirm_name == 'n':
                    name = question_mon_name()
                    break
                
                # The user confirmed the name
                else:
                    break

            return name

        # Question width of the monitor:
        def question_mon_width():
            # Question the width of the display
            while True:
                try:
                    width = float(input('Please, insert the width of your display in "cm": '))
                    break

                except ValueError:
                    print("""Oops!  That was no valid number format.\nPlease, verify if you separated the decimals with a "." and not a ",". You typed "{width}".""".format(width=width))

                # Confirm with the user if the width is correct
            while True:
                confirm_vert_pix = str(input('Please, confirm if you typed the correct value. Width = {width} cm.\n (y/n): '.format(width=width))).lower()

                # Command typed is invalid:
                if confirm_vert_pix != 'y' and confirm_vert_pix != 'n':
                    print("""The command typed "{confirm_vert_pix}" is invalid, please type "y" to confirm that the display's width is {width} cm.\nor type "n" to choose a new value for the display's width.""".format(width=width, confirm_vert_pix=confirm_vert_pix))

                # The user want to choose another value
                elif confirm_vert_pix == 'n':
                    width = question_mon_width()
                    break

                # The user confirmed the display's width
                else:
                    break

            return width

        # Question monitor's resolution
        def question_mon_resol():
            print("""Please, read the brief introduction bellow to answer the next 2 questions about the resolution of your monitor.\nNormally, the monitor's resolution is in (x, y) format, where the "x" represents the quantity of pixels in the horizontal, and "y" represents the number of pixels in the vertical.\nFor example, a monitor with the resolution 1920 x 1080 have 1920 pixels in the horizontal length and 1080 in the vertical length.""")

            def question_vert_pix():
                # Question the width of the display
                while True:
                    try:
                        vert_pix = int(input('How many pixels in the VERTICAL have your monitor? '))
                        break

                    except ValueError:
                        print("Oops!  That was no valid number.  Try again...")

                    # Confirm with the user if the vert_pix is correct
                while True:
                    confirm_vert_pix = str(input('Please, confirm if you typed the correct value for the VERTICAL number of pixels. Your answer was {vert_pix}\n (y/n): '.format(vert_pix=vert_pix))).lower()

                    # Command typed is invalid:
                    if confirm_vert_pix != 'y' and confirm_vert_pix != 'n':
                        print("""The command typed "{confirm_vert_pix}" is invalid, please type "y" to confirm that your monitor have {vert_pix} pixels in the HORIZONTAL.\nor type "n" to choose a new value to the number of HORIZONTAL pixels in your monitor.""".format(vert_pix=horz_pix, confirm_vert_pix=confirm_vert_pix))

                    # The user want to choose another value
                    elif confirm_vert_pix == 'n':
                        vert_pix = question_vert_pix()
                        break

                    # The user confirmed the display's vert_pix
                    else:
                        break

                return vert_pix

            def question_horz_pix():
                # Question the width of the display
                while True:
                    try:
                        horz_pix = int(input('How many pixels in the HORIZONTAL have your monitor? '))
                        break

                    except ValueError:
                        print("Oops!  That was no valid number.  Try again...")

                    # Confirm with the user if the horz_pix is correct
                while True:
                    confirm_horz_pix = str(input('Please, confirm if you typed the correct value for the HORIZONTAL number of pixels. Your answer was {horz_pix}\n (y/n): '.format(horz_pix=horz_pix))).lower()

                    # Command typed is invalid:
                    if confirm_horz_pix != 'y' and confirm_horz_pix != 'n':
                        print("""The command typed "{confirm_horz_pix}" is invalid, please type "y" to confirm that your monitor have {horz_pix} pixels in the HORIZONTAL.\nor type "n" to choose a new value to the number of HORIZONTAL pixels in your monitor.""".format(horz_pix=horz_pix, confirm_horz_pix=confirm_horz_pix))

                    # The user want to choose another value
                    elif confirm_horz_pix == 'n':
                        horz_pix = question_horz_pix()
                        break

                    # The user confirmed the display's vert_pix
                    else:
                        break

                return horz_pix

            horz_pix = question_horz_pix()
            vert_pix = question_vert_pix()

            return (horz_pix, vert_pix)

        # Question the monitor's frequency
        def question_mon_freq():
            print("""All monitors have a frame rate value that says how many "images" the monitor can display per second.\nThe most commom frame rate value is 60 Hz.""")
            while True:
                try:
                    freq = int(input('Please, insert the frame rate of your monitor in "hz" (hertz): '))
                    break

                except ValueError:
                    print("Oops!  That was no valid number.  Try again...")

                # Confirm with the user if the freq is correct
            while True:
                confirm_mon_freq = str(input('Please, confirm if you typed the correct value. Your answer was {freq} Hz.\n (y/n): '.format(freq=freq))).lower()

                # Command typed is invalid:
                if confirm_mon_freq != 'y' and confirm_mon_freq != 'n':
                    print("""The command typed "{confirm_mon_freq}" is invalid, please type "y" to confirm that the monitor's frame rate is {freq} Hz.\nor type "n" to choose a new value for the monitor's frame rate.""".format(freq=freq, confirm_mon_freq=confirm_mon_freq))

                # The user want to choose another value
                elif confirm_mon_freq == 'n':
                    freq = question_mon_freq()
                    break

                # The user confirmed the display's freq
                else:
                    break

            return freq

        name = question_mon_name()
        width = question_mon_width()
        resolution = question_mon_resol()
        freq = question_mon_freq()

        return name, width, resolution, freq

    def set_window(self):
        # If there's no monitor object, create it
        try:
            self.mon
        except AttributeError:
            self.mon = self.set_monitor()

        # Load monitor frequency
        freq = self.monDict['monitor_frequency']

        if self.fullscreen:
            win = visual.Window(monitorFramePeriod=freq, monitor=self.mon, fullscr=True, units=['cm', 'norm'], color=(1, 1, 1))
        else:
            win = visual.Window(size=[1200, 800], monitorFramePeriod=freq, monitor=self.mon, units=['cm', 'norm'], color=(1, 1, 1))
        return win

    def set_monitor(self):
        # Load monitor settings
        name = self.monDict['monitor_name']
        width = self.monDict['monitor_width']
        resol = self.monDict['monitor_resolution']

        mon = monitors.Monitor(name=name, width=width)
        mon.setSizePix(resol)

        return mon

    def confirmDisplaySet(self):
        try:
            self.win
        except AttributeError:
            self.win = self.set_window(monDict=self.monDict)

        # DETERMINE THE LENGTH OF THE LINES
        horz_leng = np.random.choice(np.arange(1, 10, 0.5)) / 2
        virt_leng = np.random.choice(np.arange(1, 10, 0.5)) / 2

        # CREATE LINE OBJECT
        horz_line = visual.Line(self.win, start=(-horz_leng, 0), end=(horz_leng, 0), units='cm', lineColor='black', lineWidth=3)
        virt_line = visual.Line(self.win, start=(0, -virt_leng), end=(0, virt_leng), units='cm', lineColor='black', lineWidth=3)

        # DISPLAY HORIZONTAL LINE
        horz_line.draw()
        self.win.flip()
        # INPUT
        while True:
            try:
                value_horz = float(input('Please type the size (in "cm") of the horizontal line displayed: '))
                break
            except ValueError():
                print('The format used to describe the size is wrong,\nplease type the correct size in "cm": ')

        # DISPLAY VERTICAL LINE
        virt_line.draw()
        self.win.flip()
        # INPUT
        while True:
            try:
                value_virt = float(input('Please type the size (in "cm") of the vertical line displayed: '))
                break
            except ValueError():
                print('The format used to describe the size is wrong,\nplease type the correct size in "cm": ')

        # COMPARE ANSWERS
        # MONITOR CONFIG IS CORRECT. PRINT OUT A CONFIRMATION MESSAGE AND CLOSE THE WINDOW
        if value_horz == horz_leng * 2 and value_virt == virt_leng * 2:
            self.win.close()
            print('Monitor configuration is CORRECT, you can proceed with the trials.')
        # MONITOR CONFIG IS NOT CORRECT. RAISE EXCEPTION AND CLOSE THE WINDOW
        else:
            self.win.close()
            raise Exception('Monitor configuration is WRONG, please stop the trials until corrected.')

    def stimulus_generator(self):
        fixation = visual.ShapeStim(self.win, 
            vertices=((0, -0.5), (0, 0.5), (0,0), (-0.5,0), (0.5, 0)),
            lineWidth=5,
            closeShape=False,
            lineColor="black",
            units='cm'
        )
        back_mask = visual.TextStim(self.win, text='', units='cm', height=3, alignHoriz='center', alignVert='center', color=(-1, -1, -1))
        prime = visual.TextStim(self.win, text='', units='cm', height=3, alignHoriz='center', alignVert='center', color=(-1, -1, -1))
        forward_mask = visual.TextStim(self.win, text='', units='cm', height=3, alignHoriz='center', alignVert='center', color=(-1, -1, -1))
        target = visual.TextStim(self.win, text='', units='cm', height=3, alignHoriz='center', alignVert='center', color=(-1, -1, -1))

        return fixation, back_mask, prime, forward_mask, target

    def mask_generator(self, mask_char=None, total=None, mask_size=None):
        if mask_char == None:
            mask_char = self.mask_char
        if total == None:
            total = self.totaltrials_n
        if mask_size == None:
            mask_size = self.mask_size
        mask_list = np.random.choice(mask_char, replace=True, size=(total, mask_size))
        mask_list = ["".join(mask_list[i]) for i in range(total)]
        mask_list = pd.DataFrame({
            'mask' : mask_list
        })

        return mask_list

    def endFrames(self):
        # Calculate the total duration
        total_duration_f = sum(list(self.frame_paradigm.values()))

        # Calculate frame end
        fixation_end = self.frame_paradigm['fixation']
        back_mask_end = fixation_end + self.frame_paradigm['back_mask']
        prime_end = back_mask_end + self.frame_paradigm['prime']
        forward_mask_end = prime_end + self.frame_paradigm['forward_mask']

        # Create Dictionary
        end_frames = {'fixation_end' : fixation_end, 'back_mask_end' : back_mask_end, 'prime_end' : prime_end, 'forward_mask_end' : forward_mask_end}

        return end_frames, total_duration_f

    def instructions(self, full):
        # Verify if a window is already created
        try:
            self.win
        except AttributeError:
            self.win = self.set_window()

        # Key response load
        key_response = self.kb_key_response
        concrete_key = key_response['concrete']
        abstract_key = key_response['abstract']

        # Title
        studyTitle = 'Cross-Language Associative Priming Effect Study'

        # Load the instructions according to language choice
        textPor = open(r'.\support_material\introductionTextPor.txt', 'r', encoding='utf8').read().format(concrete_key=concrete_key, abstract_key=abstract_key)

        textEng = """
        It's not ready yet.
        """

        # Verify what is the correct idiom of the instructions
        textLang = self.language_order[:3]

        # Condition statement to choose the text's verion
        if textLang == 'Por':
            _intText = textPor

        # Change elif para a textEng
        elif textLang == 'Eng':
            _intText = textPor

        title = visual.TextStim(self.win, text=studyTitle, units='norm', pos=(0, 0.8), color=(-1, -1, -1), wrapWidth=1.75)
        intText = visual.TextStim(self.win, text=_intText, units='norm', alignText='left', height=0.07, pos=(0.0, -0.12), wrapWidth=1.75, color=(-1, -1, -1))

        # Create local kb
        intKb = keyboard.Keyboard()

        # Loop to re-draw the text
        while True:
            # Draw the introduction with the title
            title.draw(), intText.draw()
            self.win.flip()

            # Record key:
            key = intKb.getKeys(keyList=('return', 'backspace'))

            if key:
                if key[0].name == 'return':
                    if not full:
                        self.win.close()
                        return True
                    else:
                        return True
                elif key[0].name == 'backspace':
                    if not full:
                        self.win.close()
                        return False
                    else:
                        return False

    def startPractice(self, full):
        if not full:
            self.win = self.set_window()

        # Try window:
        try:
            self.win
        except AttributeError:
            self.win = self.set_window()

        # Load the targets for practice
        trialsN = self.practiceLeng

        if trialsN % 2 != 0:
            raise Exception('The length of practice is not even, please change the length.')
 
        if self.subject_n % 2 == 0:
            _EngTarg = self.first_sequence
            _PorTarg = self.second_sequence
            EngTarg = _EngTarg[_EngTarg['class'] == 'control'][:int(trialsN / 2)]
            PorTarg = _PorTarg[_PorTarg['class'] == 'control'][:int(trialsN / 2)]
            EngTarg.set_index(EngTarg['original_index'], drop=True, inplace=True)
            PorTarg.set_index(PorTarg['original_index'], drop=True, inplace=True)
            EngTarg.sort_index(axis=0, ignore_index=True, inplace=True)
            PorTarg.sort_index(axis=0, ignore_index=True, inplace=True)
            firstLang = 'Português'
            secondLang = 'English'
        else:
            _PorTarg = self.first_sequence
            _EngTarg = self.second_sequence
            EngTarg = _EngTarg[_EngTarg['class'] == 'control'][:int(trialsN / 2)]
            PorTarg = _PorTarg[_PorTarg['class'] == 'control'][:int(trialsN / 2)]
            EngTarg.set_index(EngTarg['original_index'], drop=True, inplace=True)
            PorTarg.set_index(PorTarg['original_index'], drop=True, inplace=True)
            EngTarg.sort_index(axis=0, ignore_index=True, inplace=True)
            PorTarg.sort_index(axis=0, ignore_index=True, inplace=True)
            firstLang = 'English'
            secondLang = 'Português'

        # Create random prime
        def createRandomPrime():
            randomPrimeList = []
            for i in range(50):
                # Choose the number of letters on the prime
                prime_leng = int(np.random.choice(range(3, 9), size=1))
                randomPrime = "".join(np.random.choice(self.mask_char, size=prime_leng))
                randomPrimeList.append(randomPrime)

            return randomPrimeList

        randomPrime = createRandomPrime()

        # Create mask
        mask_df = self.mask_generator(total=50)

        # Create prime-target data frame
        def target_df(PorTarg=PorTarg, EngTarg=EngTarg):
            if self.subject_n % 2 == 0:
                PorTarg = PorTarg[['target_por', 'correct_response', 'pair_index']]
                PorTarg.rename(columns={'target_por' : 'target'}, inplace=True)
                EngTarg = EngTarg[['target_eng', 'correct_response', 'pair_index']]
                EngTarg.rename(columns={'target_eng' : 'target'}, inplace=True)

                target_df = PorTarg.append(EngTarg).reset_index(drop=True)

                return target_df

            else:
                PorTarg = PorTarg[['target_por', 'correct_response']]
                PorTarg.rename(columns={'target_por' : 'target'}, inplace=True)
                EngTarg = EngTarg[['target_eng', 'correct_response']]
                EngTarg.rename(columns={'target_eng' : 'target'}, inplace=True)

                target_df = EngTarg.append(PorTarg).reset_index(drop=True)

                return target_df

        target_df = target_df() 

        # Create text objects
        self.fixation, self.back_mask, self.prime, self.forward_mask, self.target = self.stimulus_generator()

        end_frames, total_f = self.endFrames()

        # Create practice KB
        practiceKb = keyboard.Keyboard()

        # Create show language TextStim
        LangTextF = visual.TextStim(self.win, text=firstLang, units='norm', height=.2, color=(-1, -1, -1))
        LangTextS = visual.TextStim(self.win, text=secondLang, units='norm', height=.2, color=(-1, -1, -1))

        # Display language of the trial:
        def showLangTrial(StimText):
            StimText.draw()
            self.win.flip()
            core.wait(2)

        # EXPERIMENT LOOP
        for trialN in np.arange(trialsN):
            # Display the language of the trail 
            if trialN == 0:
                showLangTrial(LangTextF)
            elif trialN == (int(trialsN / 2) - 1):
                showLangTrial(LangTextS)

            # Show fixation cross
            self.fixation.setAutoDraw(True)
            self.win.flip()
            self.fixation.draw()

            # Break to prepare the stimulus while the fixation cross is draw
            stimPrep = core.StaticPeriod(screenHz=self.monDict['monitor_frequency'], win=self.win, name='Stimulus Preparation Interval')
            stimPrep.start((self.timeparadigm['fixation'] / 1000))

            # STIMULUS PREPARATION
            self.back_mask.text = mask_df['mask'][trialN]
            self.prime.text = randomPrime[trialN]
            self.forward_mask.text = mask_df['mask'][trialN]
            self.target.text = target_df['target'][trialN]

            # Complete the preparation period with a frame remaning to finish the time.
            stimPrep.complete(), self.fixation.setAutoDraw(False)

            for frameN in np.arange(end_frames['fixation_end'] - 1, total_f + 1):
                # FIXATION DRAW, the stimuli will be draw for one frame to finish the preparation interval
                if frameN < end_frames['fixation_end']:
                    self.fixation.draw()

                # BACK MASK DRAW
                elif frameN < end_frames['back_mask_end']:
                    self.back_mask.draw()

                # PRIME DRAW                  
                elif frameN < end_frames['prime_end']:
                    self.prime.draw()

                # FORWARD MASK DRAW
                elif frameN < end_frames['forward_mask_end']:
                    self.forward_mask.draw()

                else:
                    # DRAW TARGET
                    self.target.draw()

                    # REDRAW TARGET LOOP AND WAIT FOR KEY
                    key = practiceKb.waitKeys(keyList=('z', 'm'), stimDraw=self.target)

                    # If the response was incorrect play error sound
                    if key.name != target_df['correct_response'][trialN]:
                        playsound(r'.\support_material\incorrect.mp3')

                self.win.flip()

        if not full:
            self.win.stop()
            return True

        else:
            return True

    def endPractice(self, full):
        if not full:
            self.win = self.set_window()

        try:
            self.win
        except AttributeError:
            self.win = self.set_window()

        # Load texts:
        if self.language_order[:3] == 'Por':
            _endPracText = open(r'.\support_material\endPracticePor.txt', 'r', encoding='utf8').read()
            _endText = """Por favor, pressione a tecla "Enter" no teclado para começar o período de avaliação."""
        elif self.language_order[:3] == 'Eng':
            _endText = """Por favor, pressione a tecla "Enter" no teclado para começar o período de avaliação."""
            _endPracText = open(r'.\support_material\endPracticePor.txt', 'r', encoding='utf8').read()

        titleText = visual.TextStim(self.win, text='End of Practice', color=(-1, -1, -1), units='norm', wrapWidth=1.8, alignText='center', height=0.15, pos=(0, 0.8))

        endPracText = visual.TextStim(self.win, text=_endPracText, color=(-1, -1, -1), units='norm', wrapWidth=1.8, alignText='left', height=0.1, pos=(0, 0.2))

        countdownText = visual.TextStim(self.win, text='', color=(-1, -1, -1), units='norm', wrapWidth=1.8, alignText='center', height=0.3, pos=(0, -0.5))

        countdown = core.CountdownTimer(60)

        def display_countdown():
            countdown.reset()
            self.kb.start()
            titleText.autoDraw = True
            endPracText.autoDraw = True
            stop = False
            for i in range(1, -1, -1):
                if not stop: 
                    countdown.reset()
                    while countdown.getTime() > 0.0:
                        sec = int(np.around(countdown.getTime(), 0))
                        if sec > 59:
                            time = '{}:00'.format(i+1)
                        elif sec < 10:
                            time = '{}:0{}'.format(i, sec)
                        else:
                            time = '{}:{}'.format(i, sec)

                        countdownText.text = time
                        countdownText.draw()
                        self.win.flip()

                        key = self.kb.getKeys(keyList=('return'))

                        if key:
                            if key[0].name == 'return':
                                stop = True
                                break

                else:
                    titleText.autoDraw = False
                    endPracText.autoDraw = False
                    pass

            if not stop:
                endText = visual.TextStim(self.win, text=_endText, color=(-1, -1, -1), units='norm', wrapWidth=1.8, alignText='center', height=0.1)

                endText.draw()
                self.win.flip()
                self.kb.waitKeys(stimDraw=endText, keyList=('return'))
                self.kb.stop()

        display_countdown()

        if not full:
            self.win.close()
            return True

        else:
            return True

    def interLanguageBreak(self, full):
        if not full:
            self.win = self.set_window()

        try:
            self.win
        except AttributeError:
            self.win = self.set_window()

        # Load texts:
        if self.language_order[:3] == 'Por':
            _interText = open(r'.\support_material\interLanguageTextPor.txt', 'r', encoding='utf8').read()
            _endText = """Por favor, pressione a tecla "Enter" no teclado para começar o período de avaliação da próxima língua."""
        elif self.language_order[:3] == 'Eng':
            _interText = open(r'.\support_material\interLanguageTextPor.txt', 'r', encoding='utf8').read()
            _endText = """Por favor, pressione a tecla "Enter" no teclado para começar o período de avaliação da próxima lingua."""

        titleText = visual.TextStim(self.win, text='End of the First Language Evaluation', color=(-1, -1, -1), units='norm', wrapWidth=1.8, alignText='center', height=0.15, pos=(0, 0.8))

        endPracText = visual.TextStim(self.win, text=_interText, color=(-1, -1, -1), units='norm', wrapWidth=1.8, alignText='left', height=0.1, pos=(0, 0.2))

        countdownText = visual.TextStim(self.win, text='', color=(-1, -1, -1), units='norm', wrapWidth=1.8, alignText='center', height=0.3, pos=(0, -0.5))

        countdown = core.CountdownTimer(60)

        def display_countdown():
            countdown.reset()
            self.kb.start()
            titleText.autoDraw = True
            endPracText.autoDraw = True
            stop = False
            for i in range(4, -1, -1):
                if not stop: 
                    countdown.reset()
                    while countdown.getTime() > 0.0:
                        sec = int(np.around(countdown.getTime(), 0))
                        if sec > 59:
                            time = '{}:00'.format(i+1)
                        elif sec < 10:
                            time = '{}:0{}'.format(i, sec)
                        else:
                            time = '{}:{}'.format(i, sec)

                        countdownText.text = time
                        countdownText.draw()
                        self.win.flip()

                        key = self.kb.getKeys(keyList=('return'))

                        if key:
                            if key[0].name == 'return':
                                stop = True
                                break

                else:
                    titleText.autoDraw = False
                    endPracText.autoDraw = False
                    pass

            if not stop:
                endText = visual.TextStim(self.win, text=_endText, color=(-1, -1, -1), units='norm', wrapWidth=1.8, alignText='center', height=0.1)

                endText.draw()
                self.win.flip()
                self.kb.waitKeys(stimDraw=endText, keyList=('return'))
                self.kb.stop()

        display_countdown()

        if not full:
            self.win.close()
            return True

        else:
            return True

    def experimentEnd(self, full):
        if not full:
            self.win = self.set_window()

        try:
            self.win
        except AttributeError:
            self.win = self.set_window()

        # Load texts:
        if self.language_order[:3] == 'Por':
            # _endPracText = open(r'.\support_material\endPracticePor.txt', 'r', encoding='utf8').read()
            _endText = open(r'.\support_material\endExpTextPor.txt', 'r', encoding='utf8').read() 
        elif self.language_order[:3] == 'Eng':
            _endText = open(r'.\support_material\endExpTextPor.txt', 'r', encoding='utf8').read()
            # _endPracText = open(r'.\support_material\endPracticePor.txt', 'r', encoding='utf8').read()

        title = visual.TextStim(self.win, text='End Of The Experiment', units='norm', pos=(0, 0.8), color=(-1, -1, -1), wrapWidth=1.75, height=0.15)
        endText = visual.TextStim(self.win, text=_endText, units='norm', alignText='left', height=0.1, pos=(0.0, 0), wrapWidth=1.75, color=(-1, -1, -1))

        title.autoDraw = True
        endText.autoDraw = True

        title.draw()
        endText.draw()
        self.win.flip()

        while True:
            self.kb.start()
            
            key = self.kb.getKeys(keyList=('return'))

            if key:
                if key[0].name == 'return':
                    break

        self.win.close()

        return True

    def startTrial(self, order, full):
        """:Parameters:
        order must be a string with the position of the trial.
        order = ['first', 'second', 'third']
        """

        # IF NOT FULL, CHOOSE THE NUMBER OF TRIALS
        if not full:
            while True:
                try:
                    limit = int(input('Please, insert the number of trials: '))
                    self.win = self.set_window()
                    break

                except ValueError:
                    print("Oops!  That was no valid number.  Try again...")
                    
        # TRY TO CALL THE SELF.WIN, EXCEPT CREATE SELF.WIN
        try:
            self.win
        except AttributeError:
            self.win = self.set_window()

        # CREATE STIMULUS OBJECT
        try:
            self.fixation, self.back_mask, self.prime, self.forward_mask, self.target
            if not full:
                self.fixation, self.back_mask, self.prime, self.forward_mask, self.target = self.stimulus_generator()

        except AttributeError:
            self.fixation, self.back_mask, self.prime, self.forward_mask, self.target = self.stimulus_generator()

        # CREATE TRIAL KEYBOARD:
        trial_kb = keyboard.Keyboard(waitForStart=True)

        # CREATE L1_L2 VARIABLE AND LOAD PRIME-TARGET SEQUENCE
        language_order = self.language_order.split('-')

        if self.fullcross:
            if order == 'first':
                prime_target_df = self.first_sequence
                l1_l2 = language_order[0]
            elif order == 'second':
                prime_target_df = self.second_sequence
                l1_l2 = language_order[1]
            else:
                raise Exception('The order used is either misspelled or out of range.\norder = {}'.format(order))
        else:
            if order == 'first':
                prime_target_df = self.first_sequence
                l1_l2 = self.onelanguageorder
            elif order == 'second':
                prime_target_df = self.second_sequence
                l1_l2 = self.onelanguageorder
            else:
                raise Exception('The order used is either misspelled or out of range.\norder = {}'.format(order))

        # PRIME-TARGET DATAFRAME COLUMNS
        columns_pt = list(prime_target_df.columns)

        # DETERMINE THE FRAME DURATION
        end_frames, total_f = self.endFrames()

        # LOAD MASK DATA FRAME
        mask_df = self.mask_generator()

        # CREATE TRIALS DATA FRAME
        trials_data = pd.DataFrame(columns=['prime', 'target', 'group', 'pair_index', 'mask', 'l1_l2',
        'key_name', 'correct', 'response_time', 'key_tDown',
        'fixation_dur', 'bm_dur', 'prime_dur', 'fm_dur', 'target_dur'])

        columns_trial = list(trials_data.columns)

        # Decide what text will show before the start of the evalutation
        if order == 'first':
            if self.language_order[:3] == 'Por':
                _prepText = 'English'
            elif self.language_order[:3] == 'Eng':
                _prepText = "Português"
        elif order == 'second':
            langOrder = self.language_order.split('-')[1]
            if langOrder[:3] == 'Por':
                _prepText = 'English'
            elif langOrder[:3] == 'Eng':
                _prepText = "Português"

        prepText = visual.TextStim(self.win, text=_prepText, units='norm', height=.3, color=(-1, -1, -1))

        def showLangTrial(StimText):
            StimText.draw()
            self.win.flip()
            core.wait(2)

        showLangTrial(prepText)

        # EXPERIMENT LOOP
        for trialN in np.arange(self.language_n): 
            # Show fixation cross
            self.fixation.setAutoDraw(True)
            self.fixation.draw()
            self.win.flip()

            # Break to prepare the stimulus while the fixation cross is draw
            stimPrep = core.StaticPeriod(screenHz=self.monDict['monitor_frequency'], win=self.win, name='Stimulus Preparation Interval')
            self.monitorclock.reset(), stimPrep.start((self.timeparadigm['fixation'] / 1000))

            # STIMULUS PREPARATION
            self.back_mask.text = mask_df['mask'][trialN]
            self.prime.text = prime_target_df[columns_pt[0]][trialN]
            self.forward_mask.text = mask_df['mask'][trialN]
            self.target.text = prime_target_df[columns_pt[1]][trialN]

            # DATA VARIABLES
            tClass, pair_index = prime_target_df[columns_pt[3]][trialN], prime_target_df[columns_pt[4]][trialN]

            frame_rate = self.win.getActualFrameRate(10, 40, 0, 1)

            # Complete the preparation period with a frame remaning to finish the time.
            stimPrep.complete(), self.fixation.setAutoDraw(False)

            for frameN in np.arange(end_frames['fixation_end'] - 1, total_f + 1):
                # FIXATION DRAW, the stimuli will be draw for one frame to finish the preparation interval
                if frameN < end_frames['fixation_end']:
                    self.fixation.draw()

                # BACK MASK DRAW
                elif frameN == end_frames['fixation_end']:
                    back_mask_onset = self.monitorclock.getTime()
                    self.back_mask.draw()
                elif frameN < end_frames['back_mask_end']:
                    self.back_mask.draw()

                # PRIME DRAW
                elif frameN == end_frames['back_mask_end']:
                    prime_onset = self.monitorclock.getTime()
                    self.prime.draw()                    
                elif frameN < end_frames['prime_end']:
                    self.prime.draw()

                # FORWARD MASK DRAW
                elif frameN == end_frames['prime_end']:
                    forward_mask_onset = self.monitorclock.getTime()
                    self.forward_mask.draw()
                elif frameN < end_frames['forward_mask_end']:
                    self.forward_mask.draw()

                else:
                    # DRAW TARGET
                    self.target.draw()

                    # RESET KB CLOCK AND DEFINE TARGET ONSET
                    target_onset = self.monitorclock.getTime()

                    # REDRAW TARGET LOOP AND WAIT FOR KEY
                    key = trial_kb.waitKeys(keyList=('z', 'm'), stimDraw=self.target)
                    target_time_end = self.monitorclock.getTime()

                    # VERIFY THE CONTENT OF KEY
                    if key is None:
                        print("The key variable is equal to None")
                        key.name, key.rt, key.tDown = (None, None, None)

                    # Play incorrect sound
                    if key.name != prime_target_df['correct_response'][trialN]:
                        playsound(r'.\support_material\incorrect.mp3')

                    # COLLECT TRIAL DATA
                    time_data = {
                        'fixation_dur' : back_mask_onset,
                        'back_mask_dur' : prime_onset - back_mask_onset,
                        'prime_dur'

                         : forward_mask_onset - prime_onset,
                        'forward_mask_dur' : target_onset - forward_mask_onset,
                        'target_dur' : target_time_end - target_onset
                    }

                    # UPDATE TRIALS DATA FRAME
                    trials_data = trials_data.append({
                        columns_trial[0] : self.prime.text,
                        columns_trial[1] : self.target.text,
                        columns_trial[2] : tClass,
                        columns_trial[3] : pair_index,
                        columns_trial[4] : self.back_mask.text,
                        columns_trial[5] : l1_l2,
                        columns_trial[6] : key.name,
                        columns_trial[7] : None,
                        columns_trial[8] : key.rt,
                        columns_trial[9] : key.tDown, 
                        columns_trial[10] : time_data['fixation_dur'],
                        columns_trial[11] : time_data['back_mask_dur'],
                        columns_trial[12] : time_data['prime_dur'],
                        columns_trial[13] : time_data['forward_mask_dur'],
                        columns_trial[14] : time_data['target_dur'],
                        }, ignore_index=True)

                self.win.flip()

            if not full:
                if trialN >= limit - 1:
                    break 

        # ADD CORRECT COLUMN ON DATA DF
        correct_list = []
        for i in range(trials_data.shape[0]):
            if prime_target_df['correct_response'][i] == trials_data['key_name'][i]:
                correct_list.append(True)
            else:
                correct_list.append(False)
        trials_data['correct'] = correct_list

        if not full:
            self.win.close()
            return trials_data

        else:
            return trials_data

    def startExperiment(self, full=True, save=None):
        # QUESTION THE USER IF HE WANT TO SAVE THE DATA THAT WILL BE COLLECTED 
        if save is None:
            while True:
                save_first = str(input('Do you want to save the data from the experiment? (y/n)\n')).lower()

                # IF "N" THAN QUESTION AGAIN IF THE USER ARE CERTAIN ABOUT HIS DECISION
                if save_first == 'n':
                    while True:
                        save_again = str(input("""Are you certain that you DON'T WANT TO SAVE the data that will be collected?\nIf you WANT TO SAVE the data type "y".\nIf you DON'T WANT TO SAVE type "n".\n(y/n): """)).lower()
                        if save_again == 'y':
                            save = True
                            break
                        elif save_again == 'n':
                            save = False
                            break
                        else:
                            print('The command typed is not valid, please answer with "y" to save or "n" to not save.\nYour answer was: "{}"'.format(save))

                    # BREAK OF THE OUTER LOOP
                    break

                elif save_first == 'y':
                    save = True
                    break
                else:
                    print('The command typed is not valid, please answer with "y" to save or "n" to not save.\nYour answer was: "{}"'.format(save))

        # QUESTION THE USER IF HE WANT TO EXECUTE THE FULL EXPERIMENT
        if full is None:
            while True:
                full = str(input('Do you want to do the full experiment? (y/n)\n')).lower()
                if full == 'y':
                    full = True
                    break
                elif full == 'n':
                    full = False
                    break
                else:
                    print('The command typed is not valid, please answer with "y" to do the full expriment\nor "n" to do a partial experiment.\nYour answer was: "{}"'.format(save))

        # IF THE EXPERIMENT IS SET TO FULLCROSS THAN
        if self.fullcross:
            # Show instructions
            _playPractice = self.instructions(full)

            # If play practice is True than play the practice, else: skip the pratice
            if _playPractice:
                # Start practice
                _pracComplete = self.startPractice(full)

                # Show end of practice
                _endPracComplete = self.endPractice(full)

            # Start First Part of the Trial
            data_first_trial = self.startTrial('first', full)

            # inter-language break
            _interLangBreakComplete = self.interLanguageBreak(full)

            # Start Second Part of the Trial
            data_second_trial = self.startTrial('second', full)

            # Show End Text
            _endExpComplete = self.experimentEnd(full)

            # Create the dataframe use all the trial information
            data_trial_final = data_first_trial.append(data_second_trial).reset_index(drop=True)

            # IF SAVE IS TRUE THAN THE SAVE PROCEDURE WILL BEGIN
            if save:

                # VERIFY IF A FILE WITH THE SAME NAME ALREADY EXIST
                try: 
                    test = pd.read_csv(r'.\trials_data\subject-{}.csv'.format(self.subject_n))
                    print('The data for the "subject {}" already exist,\ndo you have certainty that you want to DELETE the OLD DATA and save the new data in the place?'.format(self.subject_n))

                    # QUESTION ABOUT THE DELETE OF THE OLD FILE AND SAVE THE NEW
                    while True:
                        save = str(input('(y/n): ')).lower()

                        # TYPE COMMAND INVALID
                        if save != 'y' and save != 'n':
                            print('Oops!  Your reponse "{}" was not valid. Please type "y" to DELETE the OLD DATA and save the new\nor "n" to continue without save the new data.'.format(save))
                            pass

                        # DON'T SAVE THE DATA
                        elif save == 'n':

                            # QUESTION THE USER IF HE WANT TO PRINT OUT THE DATA
                            while True:
                                printdata = str(input('Do you want to print out the data collected?\n(y/n): ')).lower()

                                # COMMAND INVALID
                                if printdata != 'y' and printdata != 'n':
                                    print('Oops!  Your reponse "{}" was not valid. Please type "y" to DELETE the OLD DATA and save the new\nor "n" to continue without save the new data.'.format(printdata))
                                    pass

                                # PRINT OUT THE DATA
                                elif printdata == 'y':
                                    with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(data_trial_final)
                                    break

                                # Don't print data
                                else:
                                    break

                            # CLOSE WINDOW AND RETURN DATA

                            self.win.close()

                            return data_trial_final
                        
                        # IF THE USER WANT TO SAVE ANYWAY THE SAVE PROCEDURE WILL CONCLUDE
                        else:
                            data.to_csv(r'.\trials_data\subject-{}-norm.csv'.format(self.subject_n))
                            print('The data was saved successfully on the file named "subject-{}-norm.csv" in the "trials_data" directory.'.format(self.subject_n))

                            # QUESTION THE USER IF HE WANT TO PRINT OUT THE DATA
                            while True:
                                printdata = str(input('Do you want to print out the data collected?\n(y/n): ')).lower()

                                # COMMAND INVALID
                                if printdata != 'y' and printdata != 'n':
                                    print('Oops!  Your reponse "{}" was not valid. Please type "y" to DELETE the OLD DATA and save the new\nor "n" to continue without save the new data.'.format(printdata))
                                    pass

                                # PRINT OUT THE DATA
                                elif printdata == 'y':
                                    with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(data_trial_final)
                                    break

                                # Don't print data
                                else:
                                    break

                            self.win.close()

                            return data_trial_final

                # THERE'S NO FILE WITH THE SAME NAME. THE SAVE PROCEDURE WILL BE EXECUTED
                except FileNotFoundError:
                    data_trial_final.to_csv(r'.\trials_data\subject-{}.csv'.format(self.subject_n))
                    print('The data was saved successfully on the file named "subject-{}.csv" in the "trials_data" directory.'.format(self.subject_n))

                    # QUESTION THE USER ABOUT PRINT OUT THE DATA
                    while True:
                        printdata = str(input('Do you want to print out the data collected?\n(y/n): ')).lower()

                        # INVALID COMMAND
                        if printdata != 'y' and printdata != 'n':
                            print('Oops!  Your reponse "{}" was not valid. Please type "y" to DELETE the OLD DATA and save the new\nor "n" to continue without save the new data.'.format(printdata))
                            pass

                        # PRINT OUT THE DATA
                        elif printdata == 'y':
                            with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(data_trial_final)
                            break

                        # Don't print data
                        else:
                            break

                    # CLOSE THE WINDOW AND RETURN THE DATA

                    self.win.close()

                    return data_trial_final

            # USER DON'T WANT TO SAVE THE DATA
            else:
                # QUESTION TO PRINT OUT THE DATA
                while True:
                    printdata = str(input('Do you want to print out the data collected?\n(y/n): ')).lower()

                    # INVALID COMMAND
                    if printdata != 'y' and printdata != 'n':
                        print('Oops!  Your reponse "{}" was not valid. Please type "y" to DELETE the OLD DATA and save the new\nor "n" to continue without save the new data.'.format(printdata))
                        pass

                    # PRINT OUT THE DATA
                    elif printdata == 'y':
                        with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(data_trial_final)
                        break

                    # Don't print data
                    else:
                        break
                # CLOSE THE WINDOW AND RETURN THE DATA

                self.win.close()

                return data_trial_final

        # NOT FULLCROSS
        else:
            # data_trial_final = self.startTrial('first', full)

            # IF SAVE IS TRUE THAN BEGIN THE SAVE PROCEDURE
            if save:

                # VERIFY IF THERE'S SOME FILE WITH THE SAME NAME
                try: 
                    test = pd.read_csv(r'.\trials_data\subject-{}.csv'.format(self.subject_n))
                    print('The data for the "subject {}" already exist,\ndo you have certainty that you want to DELETE the OLD DATA and save the new data in the place?'.format(self.subject_n))

                    # QUESTION IF THE USER WANT TO DELETE THE OLD FILE AND SAVE THE NEW
                    while True:
                        save = str(input('(y/n): ')).lower()

                        # INVALID COMMAND
                        if save != 'y' and save != 'n':
                            print('Oops!  Your reponse "{}" was not valid. Please type "y" to DELETE the OLD DATA and save the new\nor "n" to continue without save the new data.'.format(save))
                            pass

                        # NOT SAVE. RETURN THE DATA AND CLOSE WINDOW
                        elif save == 'n':

                            self.win.close()

                            return data_trial_final

                        # SAVE THE DATA
                        else:
                            data.to_csv(r'.\trials_data\subject-{}-norm.csv'.format(self.subject_n))
                            print('The data was saved successfully on the file named "subject-{}-norm.csv" in the "trials_data" directory.'.format(self.subject_n))

                # THERE'S NO FILE WITH THE SAME NAME. SAVE THE DATA, RETURN DATA AND CLOSE WINDOW.
                except FileNotFoundError:
                    data_trial_final.to_csv(r'.\trials_data\subject-{}.csv'.format(self.subject_n))
                    print('The data was saved successfully on the file named "subject-{}.csv" in the "trials_data" directory.'.format(self.subject_n))

                    self.win.close()

                    return data_trial_final

            # IF SAVE IS FALSE THAN RETURN DATA AND CLOSE WINDOW.
            else:
                self.win.close()

                return data_trial_final

Experiment(n=None, fullscreen=True)

##############################################################################################################################################################################

class StatisticalAnalysis():
    def __init__(self, n=None, columns=None, save=None, view_data=None):
        # QUESTION TO THE USER WHAT IS THE VOLUNTEER NUMBER
        if n is None:
            while True:
                try:
                    n = int(input('Please enter the number of the volunteer: '))
                    break
                except ValueError:
                    print("Oops!  That was no valid number.  Try again...")

        self.subject_n = n

        # Create the attribute subject_df that will be used in the full_preprocess function
        self.subject_raw_df = pd.read_csv(r'.\trials_data\subject-{}.csv'.format(n), index_col=0)
        if columns is None:
            columns = ['response_time', 'group', 'correct', 'pair_index', 'l1_l2'] 
        
        subject_df = self.subject_raw_df[columns]
        subject_df['response_time'] * 1000
        self.subject_df = subject_df

        # VERIFY IF A FILE WITH THE SAME NAME ALREADY EXIST. IF IT'S NIETHER PREPROCESS DATA OR SAVE WILL BE EXECUTED
        try:
            self.full_preprocess_data = pd.read_csv(r'.\trials_data\subject-{}-norm.csv'.format(n), index_col=0)
            preprocess_data = False
        except FileNotFoundError:
            preprocess_data = True

        if preprocess_data:
            self.full_preprocess_data = self.full_preprocess()

        # QUESTION TO THE USER ABOUT HIS DESIRE OF SAVE THE PREPROCESS DATA
        if preprocess_data:
            if save is None:
                while True:
                    save = str(input('Do you want to save the normalized data? (y/n)\n')).lower()
                    if save != 'y' and save != 'n':
                        print('Oops!  Your reponse "{}" was not valid. Please type "y" to save or "n" to not save.'.format(save))
                        pass
                    else:
                        break

                if save == 'y':
                    self.save()

        # QUESTION TO THE USER IF HE WANT TO VIEW THE DATA
        if view_data is None:
            while True:
                vd = str(input('Do you want to visualize the graphs that describe the normalized data? (y/n)\n')).lower()
                if vd != 'y' and vd != 'n':
                    print('Oops!  Your reponse "{}" was not valid. Please type "y" to visualize the graphs\nor "n" to continue without visualize the graphs.'.format(vd))
                    pass
                else:
                    break

            # IF THE USER DON'T WANT TO SHOW GRAPHS THAN RAISE A QUESTION ABOUT PRINT OUT THE DATA
            if vd == 'n':
                # QUESTION ABOUT PRINT DATA
                while True:
                    vdf = str(input('Do you want to print out the normalized dataframe? (y/n)\n')).lower()
                    # INVALID COMMAND
                    if vdf != 'y' and vdf != 'n':
                        print('Oops!  Your reponse "{}" was not valid. Please type "y" to print out the normalized dataframe\nor "n" to continue without print out.'.format(vdf))
                        pass
                    # WANT TO PRINT
                    elif vdf == 'y':
                        with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(self.full_preprocess_data)
                        return
                    # DON'T WANT TO PRINT
                    else:
                        return

            # IF VD IS "Y" THAN CALL VIEW_DATA()
            else:
                self.plotdata()

                # QUESTION ABOUT PRINT DATA
                while True:
                    vdf = str(input('Do you want to print out the normalized dataframe? (y/n)\n')).lower()
                    # INVALID COMMAND
                    if vdf != 'y' and vdf != 'n':
                        print('Oops!  Your reponse "{}" was not valid. Please type "y" to print out the normalized dataframe\nor "n" to continue without print out.'.format(vdf))
                        pass
                    # WANT TO PRINT
                    elif vdf == 'y':
                        with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(self.full_preprocess_data)
                        return
                    # DON'T WANT TO PRINT
                    else:
                        return

        # ELSE: VIEW_DATA IS NOT NONE
        else:
            if view_data:
                self.plotdata()

                # QUESTION ABOUT PRINT DATA
                while True:
                    vdf = str(input('Do you want to print out the normalized dataframe? (y/n)\n')).lower()
                    # INVALID COMMAND
                    if vdf != 'y' and vdf != 'n':
                        print('Oops!  Your reponse "{}" was not valid. Please type "y" to print out the normalized dataframe\nor "n" to continue without print out.'.format(vdf))
                        pass
                    # WANT TO PRINT
                    elif vdf == 'y':
                        with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(self.full_preprocess_data)
                        return
                    # DON'T WANT TO PRINT
                    else:
                        return
            else:
                # QUESTION ABOUT PRINT DATA
                while True:
                    vdf = str(input('Do you want to print out the normalized dataframe? (y/n)\n')).lower()
                    # INVALID COMMAND
                    if vdf != 'y' and vdf != 'n':
                        print('Oops!  Your reponse "{}" was not valid. Please type "y" to print out the normalized dataframe\nor "n" to continue without print out.'.format(vdf))
                        pass
                    # WANT TO PRINT
                    elif vdf == 'y':
                        with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(self.full_preprocess_data)
                        return
                    # DON'T WANT TO PRINT
                    else:
                        return

    def interquartile(self, group=None, df=None):
        # IF THERE'S NO DF THAN RETURN NONE
        if df is None:
            return None

        data = df

        # GROUP CONDITIONAL
        if group is None:
            data = data['response_time'].values
            q1, q3 = np.percentile(data, [25, 75])
            lower_bound = q1 -(1.5*(q3 - q1))
            upper_bound = q3 +(1.5*(q3 - q1))
            return lower_bound, upper_bound
        else:
            l1_l2 = ['PorEng', 'EngPor']
            lower_bound = {
                l1_l2[0] : None,
                l1_l2[1] : None
                }
            upper_bound = {
                l1_l2[0] : None,
                l1_l2[1] : None
                }

            for langPair in l1_l2:
                data_onel = data[data['l1_l2'] == langPair]
                data_onel = data_onel[data_onel['group'] == group]['response_time']
                q1, q3 = np.percentile(data_onel, [25, 75])
                lower_bound[langPair] = np.around((q1-(1.5*(q3-q1))), 2)
                upper_bound[langPair] = np.around((q3+(1.5*(q3-q1))), 2)

            return lower_bound, upper_bound

    def remove_outliers(self, group=None, df=None):
        # IF THERE'S NO DF THAN RETURN NONE
        if df is None:
            return None

        data = df

        n = 0

        if group is None:
            resp_time = data['response_time'].values
            bol = []
            lower, upper = self.interquartile(df=data)
            for i in range(len(resp_time)):
                if resp_time[i] < lower or resp_time[i] > upper:
                    n += 1
                    bol.append(False)
                else:
                    bol.append(True)
            data = data[bol]

            if n > 0:
                print('{} outliers removed'.format(n))

            return data.reset_index(drop=True)

        else:
            bol = []
            l1_l2 = ['PorEng', 'EngPor']
            lower, upper = self.interquartile(df=data, group=group)
            n_l0 = 0
            n_l1 = 0
            for i in range(data.shape[0]):
                if data['l1_l2'][i] == l1_l2[0]:
                    if data['group'][i] != group:
                        bol.append(True)
                    elif data['response_time'][i] < lower[l1_l2[0]] or data['response_time'][i] > upper[l1_l2[0]]:
                        n_l0 += 1
                        bol.append(False)
                    else:
                        bol.append(True)
                else:
                    if data['group'][i] != group:
                        bol.append(True)
                    elif data['response_time'][i] < lower[l1_l2[1]] or data['response_time'][i] > upper[l1_l2[1]]:
                        n_l1 += 1
                        bol.append(False)
                    else:
                        bol.append(True)

            data = data[bol]

            if n_l0 > 0 or n_l1 > 0:
                print('{group}: {n_l0} outliers removed from {l0} pairs and {n_l1} outliers removed from {l1} pairs.'.format(group=group,
                n_l0=n_l0, n_l1=n_l1, l0=l1_l2[0], l1=l1_l2[1]))

            return data.reset_index(drop=True)

    def remove_errors(self, df=None):
        # IF THERE'S NO DF THAN RETURN NONE
        if df is None:
            return None

        data = df

        # Define the specific language variables
        n_l0, cong_l0, incong_l0, control_l0, n_l1, cong_l1, incong_l1, control_l1 = (0, 0, 0, 0, 0, 0, 0, 0)
        l1_l2 = ['PorEng', 'EngPor']

        # The loop below will remove erros through booleans, but the conditionals inside the loop are recording subgroups to print out later
        bol = []
        for i in range(data.shape[0]):
            if data['correct'][i] == True:
                bol.append(True)

            else:
                bol.append(False)
                if data['l1_l2'][i] == l1_l2[0]:
                    n_l0 += 1
                    if data['group'][i] == 'congruent':
                        cong_l0 += 1
                    elif data['group'][i] == 'incongruent':
                        incong_l0 += 1
                    else:
                        control_l0 += 1
                else:
                    n_l1 += 1
                    if data['group'][i] == 'congruent':
                        cong_l1 += 1
                    elif data['group'][i] == 'incongruent':
                        incong_l1 += 1
                    else:
                        control_l1 += 1

        data = data[bol]

        # IF THERE'S AT LEAST 1 ERROR THAN:
        if n_l0 > 0 or n_l1 > 0: 
            # PRINT PORENG ERRORS
            print('{n_l0} errors removed from {l0} pairs.'.format(n_l0=n_l0, l0=l1_l2[0]))
            print('congruent: {cong_l0}, incongruent: {incong_l0}, control: {control_l0}.'.format(cong_l0=cong_l0, incong_l0=incong_l0, control_l0=control_l0))

            # PRINT ENGPOR ERRORS
            print('{n_l1} errors removed from {l1} pairs.'.format(n_l1=n_l1, l1=l1_l2[1]))
            print('congruent: {cong_l1}, incongruent: {incong_l1}, control: {control_l1}.'.format(cong_l1=cong_l1, incong_l1=incong_l1, control_l1=control_l1))

            # CREATE A DICTIONARY WITH IMPORTANT VALUES:
            # The first line will calculate the mean of errors with cong and incong. The second line will calculate the factor
            # error of test vs control analysis
            meanTestL0 = np.mean((cong_l0, incong_l0))
            testControlL0 = control_l0 / meanTestL0

            meanTestL1 = np.mean((cong_l1, incong_l1))
            testControlL1 = control_l1 / meanTestL1

            # Create to record the values:
            testControl = {
                l1_l2[0] : testControlL0,
                l1_l2[1] : testControlL1
            }

            return data.reset_index(drop=True), testControl

        return data.reset_index(drop=True), None

    def z_score_normalization(self, df=None):
        # IF THERE'S NO DF THAN RETURN NONE
        if df is None:
            return None

        data = df

        resp_t = data['response_time'].values
        mean = np.mean(resp_t)
        std = np.std(resp_t)
        norm_data = []
        for i in range(len(resp_t)):
            z = (resp_t[i] - mean) / std
            norm_data.append(z)
        data['z_score_norm'] = norm_data

        return data

    def exp_normalization(self, df=None):
        # IF THERE'S NO DF THAN RETURN NONE
        if df is None:
            return None

        data = df

        resp_t = np.array(data['response_time'].values) / 100
        exp_list = []
        for i in range(len(resp_t)):
            x = np.round(np.exp(resp_t[i]), 5)
            exp_list.append(x)
        exp_sum = sum(exp_list)
        norm_data = []
        for i in range(len(resp_t)):
            z = np.exp(resp_t[i]) / exp_sum
            norm_data.append(z)
        data['exp_norm'] = norm_data
        return data

    def rescaling(self, df=None):
        # IF THERE'S NO DF THAN RETURN NONE
        if df is None:
            return None

        data = df

        resp_t = np.array(data['response_time'].values)
        up = np.max(resp_t)
        down = np.min(resp_t)
        div = up - down
        norm_values = []
        for i in range(len(resp_t)):
            x = (resp_t[i] - down) / div
            norm_values.append(x)
        data['rescaling_data'] = norm_values
        return data

    def full_preprocess(self, df=None):
        # IF THERE'S NO DF THAN RETURN NONE
        if df is None:
            data = self.subject_df

        else:
            data = df

        # Remove errors
        data, testControlF = self.remove_errors(df=data)

        # Create testControlF attribute
        self.testControlF = testControlF 

        # Remove outliers
        data = self.remove_outliers(df=data, group='incongruent')
        data = self.remove_outliers(df=data, group='congruent')
        data = self.remove_outliers(df=data, group='control')

        # Normalize the data
        data = self.z_score_normalization(df=data)
        data = self.exp_normalization(df=data)
        data = self.rescaling(df=data)

        # Drop the correct column
        data = data.drop('correct', axis=1)

        return data

    def save(self, df=None):
        if df is None:
            data = self.full_preprocess_data
        else:
            data = df

        try: 
            test = pd.read_csv(r'.\trials_data\subject-{}-norm.csv'.format(self.subject_n))
            print('The data for the "subject {}" already exist,\ndo you have certainty that you want to DELETE the OLD DATA and save the new data in the place?'.format(self.subject_n))
            while True:
                save = str(input('(y/n): ')).lower()
                if save != 'y' and save != 'n':
                    print('Oops!  Your reponse "{}" was not valid. Please type "y" to DELETE the OLD DATA and save the new\nor "n" to continue without save the new data.'.format(save))
                    pass
                elif save == 'n':
                    return
                else:
                    data.to_csv(r'.\trials_data\subject-{}-norm.csv'.format(self.subject_n))
                    print('The data was saved successfully on the file named "subject-{}-norm.csv" in the "trials_data" directory.'.format(self.subject_n))
                    return

        except FileNotFoundError:
            data.to_csv(r'.\trials_data\subject-{}-norm.csv'.format(self.subject_n))
            print('The data was saved successfully on the file named "subject-{}-norm.csv" in the "trials_data" directory.'.format(self.subject_n))
            return

    def plotdata(self, first_hue='group', second_hue='l1_l2',  df=None):
        if df is None:
            data = self.full_preprocess()
        else:
            data = df

        sequence = ['incongruent', 'congruent', 'control']
        l1_l2 = ['PorEng', 'EngPor']

        # Print out the test vs control error analysis:
        print('Test VS Control Error Analysis:', self.testControlF)

        # Create fig and axes objects
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))

        dataPorEng = data[data['l1_l2'] == l1_l2[0]]
        sns.catplot(data=dataPorEng, x=first_hue, y='response_time', ax=axes[0, 0], order=sequence, kind='box')
        axes[0, 0].grid(axis='y', which='major')

        dataEngPor = data[data['l1_l2'] == l1_l2[1]]
        sns.catplot(data=dataEngPor, x=first_hue, y='response_time', ax=axes[0, 1], order=sequence, kind='box')
        axes[0, 1].grid(axis='y', which='major')


        sns.kdeplot(dataPorEng[dataPorEng[first_hue] == 'control']['response_time'], shade=True, alpha=.2, ax=axes[1, 0], color='g')
        sns.kdeplot(dataPorEng[dataPorEng[first_hue] == 'incongruent']['response_time'], shade=True, alpha=.2, ax=axes[1, 0], color='b')
        sns.kdeplot(dataPorEng[dataPorEng[first_hue] == 'congruent']['response_time'], shade=True, alpha=.2, ax=axes[1, 0], color='tab:orange')
        sns.kdeplot(dataPorEng[dataPorEng[first_hue] == 'control']['response_time'], alpha=.8, ax=axes[1, 0], color='g')
        sns.kdeplot(dataPorEng[dataPorEng[first_hue] == 'incongruent']['response_time'], alpha=.8, ax=axes[1, 0], color='b')
        sns.kdeplot(dataPorEng[dataPorEng[first_hue] == 'congruent']['response_time'], alpha=.8, ax=axes[1, 0], color='tab:orange')
        axes[1, 0].axvline(np.median(dataPorEng[dataPorEng[first_hue] == 'control']['response_time']), alpha=.8, ymax=.5, c='g')
        axes[1, 0].axvline(np.median(dataPorEng[dataPorEng[first_hue] == 'incongruent']['response_time']), alpha=.8, ymax=.5, c='b')
        axes[1, 0].axvline(np.median(dataPorEng[dataPorEng[first_hue] == 'congruent']['response_time']), alpha=.8, ymax=.5, c='tab:orange')
        axes[1, 0].legend(['control', 'incongruent', 'congruent'])

        sns.kdeplot(dataEngPor[dataEngPor[first_hue] == 'control']['response_time'], shade=True, alpha=.2, ax=axes[1, 1], color='g')
        sns.kdeplot(dataEngPor[dataEngPor[first_hue] == 'incongruent']['response_time'], shade=True, alpha=.2, ax=axes[1, 1], color='b')
        sns.kdeplot(dataEngPor[dataEngPor[first_hue] == 'congruent']['response_time'], shade=True, alpha=.2, ax=axes[1, 1], color='tab:orange')
        sns.kdeplot(dataEngPor[dataEngPor[first_hue] == 'control']['response_time'], alpha=.8, ax=axes[1, 1], color='g')
        sns.kdeplot(dataEngPor[dataEngPor[first_hue] == 'incongruent']['response_time'], alpha=.8, ax=axes[1, 1], color='b')
        sns.kdeplot(dataEngPor[dataEngPor[first_hue] == 'congruent']['response_time'], alpha=.8, ax=axes[1, 1], color='tab:orange')
        axes[1, 1].axvline(np.median(dataEngPor[dataEngPor[first_hue] == 'control']['response_time']), alpha=.8, ymax=.5, c='g')
        axes[1, 1].axvline(np.median(dataEngPor[dataEngPor[first_hue] == 'incongruent']['response_time']), alpha=.8, ymax=.5, c='b')
        axes[1, 1].axvline(np.median(dataEngPor[dataEngPor[first_hue] == 'congruent']['response_time']), alpha=.8, ymax=.5, c='tab:orange')
        axes[1, 1].legend(['control', 'incongruent', 'congruentn'])

        plt.show()
