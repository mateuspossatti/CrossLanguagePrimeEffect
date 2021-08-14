from psychopy import visual, core, monitors, event, clock
from psychopy.hardware import keyboard
from matplotlib import pyplot as plt
from playsound import playsound
from pyglet.window.key import A
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os

class Experiment:
    def __init__(self, n=0, mask_case='upper', fullcross=True, control=True, mask_size=8, onelanguageorder=None,
    fullscreen=True, timeparadigm=None, kb_keys=None, save=True, practiceLeng=50):
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

        # Define attributes based on coditional statements.
        if mask_case == 'upper':
            self.mask_char = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        elif mask_case == 'lower':
            self.mask_char = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

        if control == True:
            conditions_n = 3
            pairs_n = 50
            self.conditions = ['congruent', 'incongruent', 'control']    
        else:
            conditions_n = 2
            pairs_n = 75
            self.conditions = ['congruent', 'incongruent']

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

        # Define others attributes
        self.subject_n = n
        self.control = control
        self.pairs_n = pairs_n
        self.language_n = pairs_n * conditions_n
        self.fullcross = fullcross
        self.mask_size = mask_size
        self.fullscreen = fullscreen
        self.screen_hz = self.monDict['monitor_frequency']
        self.practiceLeng = practiceLeng

        # Print full dataframe.
        pd.set_option("display.max_rows", None, "display.max_columns", None)

        # Create a global key event to quit the program.
        key = 'q'
        modifiers = ['ctrl']
        event.globalKeys.add(key=key, modifiers=modifiers, func=core.quit)

        # Determine language order for the actual subject.
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

        # Determine frame duration time in ms.
        def frame_duration():
            ms_paradigm = self.timeparadigm
            screen_hz = self.screen_hz
            ms_frame = 1000 / screen_hz
            frame_paradigm = {'fixation' : np.round((ms_paradigm['fixation'] / ms_frame)), 'back_mask' : np.round((ms_paradigm['back_mask'] / ms_frame)), 
            'prime' : np.round((ms_paradigm['prime'] / ms_frame)), 'forward_mask' : np.round((ms_paradigm['forward_mask'] / ms_frame))}
            return frame_paradigm

        self.frame_paradigm = frame_duration()

        # CREATE CLOCK, KEYBOARD
        self.kbclock = clock.Clock()

        self.monitorclock = clock.Clock()

        # Create a monitor object.
        def set_monitor():
            # Load monitor settings.
            name = self.monDict['monitor_name']
            width = self.monDict['monitor_width']
            resol = self.monDict['monitor_resolution']

            mon = monitors.Monitor(name=name, width=width)
            mon.setSizePix(resol)

            return mon

        self.mon = set_monitor()

        # Define words sequence.
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

            # Create a list with the paris classes
            class_list = []
            for i in range(conditions_n):
                for _ in range(self.pairs_n):
                    class_list.append(self.conditions[i])

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
                if self.control == True:
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
                
                    ex_prime = list(np.random.choice(np.arange(50), replace=False, size=25))

                    df = words_df.loc[ex_prime, first[0]].values


                    print(df)
                elif self.control == False:
                    if lo == 'PorEng-EngPor':
                        first = ['Portuguese', 'English']
                        second = ['English', 'Portuguese']
                    else:
                        first = ['English', 'Portuguese']
                        second = ['Portuguese', 'English']

                    # CHOOSE 25 EXTRA PRIMES


                    

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

# QUESTION THE USER IF HIS WANT TO START THE EXPERIMENT
        # self.data_trial_final = self.startExperiment(save=save)

############# END OF __INIT__() #################

    def set_window(self):
        # If there's no monitor object, create it
        try:
            self.mon
        except AttributeError:
            self.mon = self.set_monitor()

        # Load monitor frequency
        freq = self.monDict['monitor_frequency']

        if self.fullscreen:
            win = visual.Window(monitor=self.mon, size=self.monDict['monitor_resolution'], fullscr=True, units=['cm', 'norm'], color=(1, 1, 1))
        else:
            win = visual.Window(size=[1920, 1080], monitor=self.mon, units=['cm', 'norm'], color=(1, 1, 1))
        return win

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
        back_mask = visual.TextStim(self.win, text='', units='cm', height=3, anchorHoriz='center', anchorVert='center', color=(-1, -1, -1))
        prime = visual.TextStim(self.win, text='', units='cm', height=3, anchorHoriz='center', anchorVert='center', color=(-1, -1, -1))
        forward_mask = visual.TextStim(self.win, text='', units='cm', height=3, anchorHoriz='center', anchorVert='center', color=(-1, -1, -1))
        target = visual.TextStim(self.win, text='', units='cm', height=3, anchorHoriz='center', anchorVert='center', color=(-1, -1, -1))

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
        textPor = open(r'.\support_material\text_msg\introductionTextPor.txt', 'r', encoding='utf8').read().format(concrete_key=concrete_key, abstract_key=abstract_key)

        textEng = open(r'.\support_material\text_msg\introductionTextEng.txt', 'r', encoding='utf8').read().format(concrete_key=concrete_key, abstract_key=abstract_key)

        # Verify what is the correct idiom of the instructions
        textLang = self.language_order[:3]

        # Condition statement to choose the text's verion
        if textLang == 'Por':
            _intText = textPor

        # Change elif para a textEng
        elif textLang == 'Eng':
            _intText = textEng

        title = visual.TextStim(self.win, text=studyTitle, units='norm', pos=(0, 0.8), color=(-1, -1, -1), wrapWidth=1.75)
        intText = visual.TextStim(self.win, text=_intText, units='norm', alignText='left', height=0.07, pos=(0.0, -0.12), wrapWidth=1.75, color=(-1, -1, -1))

        # Loop to re-draw the text
        while True:
            # Draw the introduction with the title
            title.draw(), intText.draw()
            self.win.flip()

            # Record key:
            _keyname = event.waitKeys(keyList=('return', 'backspace'))[0]

            if _keyname == 'return':
                if not full:
                    self.win.close()
                    return True
                else:
                    return True
            elif _keyname == 'backspace':
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

        index_list = [np.random.choice(np.arange(25), replace=False, size=25) for _ in range(2)]

        if self.subject_n % 2 == 0:
            _EngTarg = self.first_sequence
            _PorTarg = self.second_sequence
            EngTarg = _EngTarg[_EngTarg['class'] == 'control'][:int(trialsN / 2)]
            PorTarg = _PorTarg[_PorTarg['class'] == 'control'][:int(trialsN / 2)]
            EngTarg.set_index(index_list[0], drop=True, inplace=True)
            PorTarg.set_index(index_list[1], drop=True, inplace=True)
            EngTarg.sort_index(axis=0, ignore_index=True, inplace=True)
            PorTarg.sort_index(axis=0, ignore_index=True, inplace=True)
            firstLang = 'Português'
            secondLang = 'English'
        else:
            _PorTarg = self.first_sequence
            _EngTarg = self.second_sequence
            EngTarg = _EngTarg[_EngTarg['class'] == 'control'][:int(trialsN / 2)]
            PorTarg = _PorTarg[_PorTarg['class'] == 'control'][:int(trialsN / 2)]
            EngTarg.set_index(index_list[0], drop=True, inplace=True)
            PorTarg.set_index(index_list[1], drop=True, inplace=True)
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
                PorTarg = PorTarg[['target_por', 'correct_response']]
                PorTarg.rename(columns={'target_por' : 'target'}, inplace=True)
                EngTarg = EngTarg[['target_eng', 'correct_response']]
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
            elif trialN == int(trialsN / 2):
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
                    # RESET MONITORCLOCK
                    self.kbclock.reset()
                    # DRAW TARGET AND FLIP WINDOW
                    self.target.draw()

                    self.win.flip()

                    key = event.waitKeys(keyList=('z', 'm'), timeStamped=self.kbclock)
                    keyname, time = key[0]

                    # If the response was incorrect play error sound
                    if keyname != target_df['correct_response'][trialN]:
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
            _endPracText = open(r'.\support_material\text_msg\endPracticePor.txt', 'r', encoding='utf8').read()
            _titleText = 'Fim da Prática'
            _endText = """Por favor, pressione a tecla "Enter" no teclado para começar o período de avaliação."""
        elif self.language_order[:3] == 'Eng':
            _endText = """Please press "Enter" key on your keyboard to start the trial period."""
            _titleText = 'End of Practice'
            _endPracText = open(r'.\support_material\text_msg\endPracticeEng.txt', 'r', encoding='utf8').read()

        titleText = visual.TextStim(self.win, text=_titleText, color=(-1, -1, -1), units='norm', wrapWidth=1.8, alignText='center', height=0.15, pos=(0, 0.8))

        endPracText = visual.TextStim(self.win, text=_endPracText, color=(-1, -1, -1), units='norm', wrapWidth=1.8, alignText='left', height=0.1, pos=(0, 0.2))

        countdownText = visual.TextStim(self.win, text='', color=(-1, -1, -1), units='norm', wrapWidth=1.8, alignText='center', height=0.3, pos=(0, -0.5))

        countdown = core.CountdownTimer(60)

        def display_countdown():
            countdown.reset()
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

                        key = event.getKeys(keyList=('return'))

                        if key:
                            if key[0] == 'return':
                                stop = True
                                break

                else:
                    titleText.autoDraw = False
                    endPracText.autoDraw = False
                    pass

            if not stop:
                titleText.autoDraw = False
                endPracText.autoDraw = False

                endText = visual.TextStim(self.win, text=_endText, color=(-1, -1, -1), units='norm', wrapWidth=1.8, alignText='center', height=0.1)

                endText.draw()
                self.win.flip()
                event.waitKeys(keyList=('return'))

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
            _interText = open(r'.\support_material\text_msg\interLanguageTextPor.txt', 'r', encoding='utf8').read()
            _titleText = 'Fim da Avaliação do Primeiro Idioma'
            _endText = """Por favor, pressione a tecla "Enter" no teclado para começar o período de avaliação da próxima língua."""
        elif self.language_order[:3] == 'Eng':
            _interText = open(r'.\support_material\text_msg\interLanguageTextEng.txt', 'r', encoding='utf8').read()
            _titleText = 'End of the First Language Evaluation'
            _endText = """Please press "Enter" key on your keyboard to start the next language trial period."""

        titleText = visual.TextStim(self.win, text=_titleText, color=(-1, -1, -1), units='norm', wrapWidth=1.8, alignText='center', height=0.15, pos=(0, 0.8))

        endPracText = visual.TextStim(self.win, text=_interText, color=(-1, -1, -1), units='norm', wrapWidth=1.8, alignText='left', height=0.1, pos=(0, 0.2))

        countdownText = visual.TextStim(self.win, text='', color=(-1, -1, -1), units='norm', wrapWidth=1.8, alignText='center', height=0.3, pos=(0, -0.5))

        countdown = core.CountdownTimer(60)

        def display_countdown():
            countdown.reset()
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

                        key = event.getKeys(keyList=('return'))

                        if key:
                            if key[0] == 'return':
                                stop = True
                                break

                else:
                    titleText.autoDraw = False
                    endPracText.autoDraw = False
                    pass

            if not stop:
                titleText.autoDraw = False
                endPracText.autoDraw = False

                endText = visual.TextStim(self.win, text=_endText, color=(-1, -1, -1), units='norm', wrapWidth=1.8, alignText='center', height=0.1)

                endText.draw()
                self.win.flip()
                event.waitKeys(keyList=('return'))


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
            _endText = open(r'.\support_material\text_msg\endExpTextPor.txt', 'r', encoding='utf8').read() 
            _titleText = 'Fim do Experimento'
        elif self.language_order[:3] == 'Eng':
            _endText = open(r'.\support_material\text_msg\endExpTextEng.txt', 'r', encoding='utf8').read()
            _titleText = 'End of the Experiment'

        title = visual.TextStim(self.win, text=_titleText, units='norm', pos=(0, 0.8), color=(-1, -1, -1), wrapWidth=1.75, height=0.15)
        endText = visual.TextStim(self.win, text=_endText, units='norm', alignText='left', height=0.1, pos=(0.0, 0), wrapWidth=1.75, color=(-1, -1, -1))

        title.autoDraw = True
        endText.autoDraw = True

        title.draw()
        endText.draw()
        self.win.flip()

        event.waitKeys(keyList=('return'))

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

                    # RESET KB CLOCK, FLIP WINDOW AND DEFINE TARGET ONSET
                    self.win.flip()

                    self.kbclock.reset()

                    target_onset = self.monitorclock.getTime()

                    # WAIT FOR KEY
                    key = event.waitKeys(keyList=('z', 'm'), timeStamped=self.kbclock)
                    keyname, keytime = key[0]
                
                    target_time_end = self.monitorclock.getTime()

                    # Play incorrect sound
                    if keyname != prime_target_df['correct_response'][trialN]:
                        playsound(r'.\support_material\incorrect.mp3')
                        correct_key = False
                    
                    else:
                        correct_key = True

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
                        columns_trial[6] : keyname,
                        columns_trial[7] : correct_key,
                        columns_trial[8] : keytime,
                        columns_trial[9] : None, 
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
                                    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                                        print(data_trial_final)
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

if __name__ == '__main__':
    root_dir = os.path.dirname(__file__)
    os.makedirs(os.path.join(root_dir, 'trials_data'), exist_ok=True)

    Experiment()
