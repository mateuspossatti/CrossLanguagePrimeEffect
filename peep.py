from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pip

# TRY IMPORT PSYCHOPY MODULE
try:
    from psychopy import visual, core, monitors, event, clock
    from psychopy.hardware import keyboard
except ImportError:
    import pip
    pip.main(['install', 'psychopy'])
    from psychopy import visual, core, monitors, event, clock
    from psychopy.hardware import keyboard


class Experiment(object):
    def __init__(self, n=None, mask_case='upper', pairs_n=50, fullcross=True, conditions_n=3, mask_size=8, onelanguageorder=None,
    fullscreen=False, screen_hz=60, timeparadigm=None, kb_keys=None, monitor_name='SurfaceBook2-manual', save=None):
        """:Parameters:
        fullcross will ditermine if the effect will be studied in the two ways. 
        """

        # QUESTION THE USER WHAT IS THE VOLUNTEER'S NUMBER
        if n is None:
            while True:
                try:
                    n = int(input('Please enter the number of the volunteer: '))
                    break
                except ValueError:
                    print("Oops!  That was no valid number.  Try again...")

        # QUESTION THE USER ABOUT HIS FULLSCREEN PREFERENCE
        if fullscreen is None:
            while True:
                fs = str(input('Do you want that the trial be in fullscreen?\n(y/n): ')).lower()
                if fs != 'y' and fs != 'n':
                    print('The command typed ("{}") is invalid, please type "y" to make the experiment fullscreen\nor "n" to not make the experiment fullscreen'.format(fs))
                    pass
                elif fs == 'y':
                    self.fullscreen = True
                else:
                    self.fullscreen = False                    

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
            self.timeparadigm = {'fixation' : 500, 'back_mask' : 100, 'prime' : 50, 'forward_mask' : 50}
        else:
            self.timeparadigm = timeparadigm
        if kb_keys == None:
            self.kb_keys = ('z', 'm')
        else:
            self.kb_keys = kb_keys

        self.conditions = ['congruent', 'incongruent', 'control']
        self.subject_n = n
        self.pairs_n = pairs_n
        self.language_n = pairs_n * conditions_n
        self.fullcross = fullcross
        self.mask_size = mask_size
        self.fullscreen = fullscreen
        self.screen_hz = screen_hz

        # DETERMINE LANGUAGE ORDER FOR THE ACTUAL SUBJECT
        def subject_experiment_order():
            left, right = self.kb_keys
            if fullcross == True:
                if n % 2 == 0:
                    language_order = 'PorEng-EngPor'
                    kb_key_response = {'object' : right, 'not-object' : left}
                else:
                    language_order = 'EngPor-PorEng'
                    kb_key_response = {'object' : left, 'not-object' : right}

            elif fullcross == False:
                language_order = self.onelanguageorder
                if n % 2 == 0:
                    kb_key_response = {'object' : left, 'not-object' : right}
                else:
                    kb_key_response = {'object' : right, 'not-object' : left}

            return language_order, kb_key_response

        self.language_order, self.kb_key_response = subject_experiment_order() 

        # CREATE MONITOR AND WINDOW
        def set_monitor():
            mon = monitors.Monitor(monitor_name)

            return mon

        self.mon = set_monitor()

        # DETERMINE FRAME DURATION:
        def frame_duration():
            ms_paradigm = self.timeparadigm
            screen_hz = self.screen_hz
            ms_frame = 1000 / screen_hz
            frame_paradigm = {'fixation' : np.round((ms_paradigm['fixation'] / ms_frame)), 'back_mask' : np.round((ms_paradigm['back_mask'] / ms_frame)), 
            'prime' : np.round((ms_paradigm['prime'] / ms_frame)), 'forward_mask' : np.round((ms_paradigm['forward_mask'] / ms_frame))}
            return frame_paradigm

        self.frame_paradigm = frame_duration()

        # GENERATORS
        def clock_generator():
            monitorclock = clock.Clock()
            return monitorclock

        self.monitorclock = clock_generator()

        def hardware_generator():
            kb = keyboard.Keyboard(waitForStart=True)

            return kb

        ######## VERIFY THE IMPORTANCE OF THE KEYBOARD BELLOW
        # self.kb = hardware_generator()

        def mask_generator(mask_char=None, total=None, mask_size=None):
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

        self.mask_df = mask_generator()

        def words_sequence():
            # LOAD CSV
            words_df = pd.read_csv(r'words.csv').columns
            col_names = {}
            for col in words_df:
                col_names[col] = str
            words_df = pd.read_csv(r'words.csv', dtype=col_names)

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
        while True:
            startexp = str(input('Do you want to begin the expriment?\n(y/n): ')).lower()
            if startexp != 'y' and startexp != 'n':
                print('The command typed ("{}") is invalid, please type "y" to begin the experiment\nor "n" to continue without begin the experiment.'.format(startexp))
            elif startexp == 'y':
                self.data_trial_final = self.startExperiment(save=save)
                break
            else:
                try:
                    self.win.close()
                    break
                except AttributeError:
                    break


##############################################################################################################################################

    def set_window(self):
        if self.fullscreen:
            win = visual.Window(monitorFramePeriod=60, monitor=self.mon, fullscr=True, units=['cm', 'norm'], color=(1, 1, 1))
        else:
            win = visual.Window(size=[1200, 800], monitorFramePeriod=60, monitor=self.mon, units=['cm', 'norm'], color=(1, 1, 1))
        return win

    def confirmDisplaySet(self):
        try:
            self.win
        except AttributeError:
            self.win = self.set_window()

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

    def startTrial(self, order, full):
        """:Parameters:
        order must be a string with the position of the trial.
        order = ['first', 'second', 'third']
        """
        # TRY TO CALL THE SELF.WIN, EXCEPT CREATE SELF.WIN
        try:
            self.win
        except AttributeError:
            self.win = self.set_window()

        # IF NOT FULL, CHOOSE THE NUMBER OF TRIALS
        if not full:
            while True:
                try:
                    limit = int(input('Please, insert the number of trials: '))
                    break
                except ValueError:
                    print("Oops!  That was no valid number.  Try again...")

        # CREATE STIMULUS OBJECT
        def stimulus_generator():
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

        self.fixation, self.back_mask, self.prime, self.forward_mask, self.target = stimulus_generator()

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
        total_duration_f = sum(list(self.frame_paradigm.values()))
        fixation_end = self.frame_paradigm['fixation']
        back_mask_end = fixation_end + self.frame_paradigm['back_mask']
        prime_end = back_mask_end + self.frame_paradigm['prime']
        forward_mask_end = prime_end + self.frame_paradigm['forward_mask']
        end_frames = {'fixation_end' : fixation_end, 'back_mask_end' : back_mask_end, 'prime_end' : prime_end, 'forward_mask_end' : forward_mask_end}

        # LOAD MASK DATA FRAME
        mask_df = self.mask_df

        # CREATE TRIALS DATA FRAME
        trials_data = pd.DataFrame(columns=['prime', 'target', 'group', 'pair_index', 'mask', 'l1_l2',
        'key_name', 'correct', 'response_time', 'key_tDown',
        'fixation_dur', 'bm_dur', 'prime_dur', 'fm_dur', 'target_dur'])

        columns_trial = list(trials_data.columns)

        # EXPERIMENT LOOP
        for trialN in np.arange(self.language_n): 
            # STIMULUS PREPARATION
            self.back_mask.text = mask_df['mask'][trialN]
            self.prime.text = prime_target_df[columns_pt[0]][trialN]
            self.forward_mask.text = mask_df['mask'][trialN]
            self.target.text = prime_target_df[columns_pt[1]][trialN]

            # DATA VARIABLES
            tClass, pair_index = prime_target_df[columns_pt[3]][trialN], prime_target_df[columns_pt[4]][trialN]

            frame_rate = self.win.getActualFrameRate(10, 40, 0, 1)

            # RESET MONITOR CLOCK
            self.monitorclock.reset()

            for frameN in np.arange(total_duration_f + 1):

                # FIXATION DRAW
                if frameN < fixation_end:
                    self.fixation.draw()

                # BACK MASK DRAW
                elif frameN == fixation_end:
                    back_mask_onset = self.monitorclock.getTime()
                    self.back_mask.draw()
                    
                elif frameN < back_mask_end:
                    self.back_mask.draw()

                # PRIME DRAW
                elif frameN == back_mask_end:
                    prime_onset = self.monitorclock.getTime()
                    self.prime.draw()                    
                elif frameN < prime_end:
                    self.prime.draw()

                # FORWARD MASK DRAW
                elif frameN == prime_end:
                    forward_mask_onset = self.monitorclock.getTime()
                    self.forward_mask.draw()
                elif frameN < forward_mask_end:
                    self.forward_mask.draw()

                # TARGET DRAW AND RESPONSE COLLECT
                else:
                    # DRAW TARGET, GET TARGET ONSET, FPS AND START FRAME COUNT
                    target_onset = self.monitorclock.getTime()
                    self.target.draw()

                    # START KB AND RESET KB CLOCK
                    trial_kb.start(), trial_kb.clock.reset()

                    # REDRAW TARGET LOOP, WAIT FOR KEY
                    while True:
                        self.win.flip()
                        self.target.draw()
                        key = trial_kb.getKeys(keyList=('z', 'm'))
                        if key:
                            self.win.flip()
                            trial_kb.stop()
                            target_time_end = self.monitorclock.getTime()
                            break

                    # COLLECT TRIAL DATA


                    time_data = {
                        'fixation_dur' : back_mask_onset,
                        'back_mask_dur' : prime_onset - back_mask_onset,
                        'prime_dur' : forward_mask_onset - prime_onset,
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
                        columns_trial[6] : key[0].name,
                        columns_trial[7] : None,
                        columns_trial[8] : key[0].rt,
                        columns_trial[9] : key[0].tDown, 
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

        return trials_data

    def startExperiment(self, full=None, save=None):
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
            data_first_trial = self.startTrial('first', full)

            # REMEMBER OF DELETE LOOP
            while True:
                value_horz = str(input('Do you want to proceed to the next language trial? (y/n)\n')).lower()
                if value_horz == 'n':
                    self.win.close()
                    return data_first_trial
                elif value_horz == 'y':
                    break
                else:
                    print('The command typed is not valid, please answer with "y" to continue or "n" to stop.\nYour answer was: "{}"'.format(value_horz))

            data_second_trial = self.startTrial('second', full)

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

                # CLOSE THE WINDOW AND RETURN THE DATA

                self.win.close()

                return data_trial_final

        # NOT FULLCROSS
        else:
            data_trial_final = self.startTrial('first', full)

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

test = Experiment()
# print(test)
# Experiment()

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

        # VERIFY IF A FILE WITH THE SAME NAME ALREADY EXIST. IF IT'S NIETHER PREPROCESS DATA OR SAVE WILL BE EXECUTED
        try:
            self.full_preprocess_data = pd.read_csv(r'.\trials_data\subject-{}-norm.csv'.format(n), index_col=0)
            preprocess_data = False
        except FileNotFoundError:
            preprocess_data = True

        if preprocess_data:
            self.subject_raw_df = pd.read_csv(r'.\trials_data\subject-{}.csv'.format(n), index_col=0)

            if columns is None:
                columns = ['response_time', 'group', 'correct', 'pair_index', 'l1_l2'] 
            
            subject_df = self.subject_raw_df[columns]
            subject_df['response_time'] = np.around(np.array((subject_df['response_time'].values) * 1000), 2)
            self.subject_df = subject_df

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

        self.fig, self.axis = self.plotdata()

        # QUESTION TO THE USER IF HE WANT TO VIEW THE DATA
        if view_data is None:
            while True:
                vd = str(input('Do you want to visualize the graphs that describe the normalized data? (y/n)\n')).lower()
                if vd != 'y' and vd != 'n':
                    print('Oops!  Your reponse "{}" was not valid. Please type "y" to visualize the graphs\nor "n" to continue without visualize the graphs.'.format(vd))
                    pass
                else:
                    break

            if vd == 'y':
                self.view_data()

            while True:
                vdf = str(input('Do you want to print out the normalized data frame? (y/n)\n')).lower()
                if vdf != 'y' and vdf != 'n':
                    print('Oops!  Your reponse "{}" was not valid. Please type "y" to print out the normalized data frame\nor "n" to continue without print out.'.format(vdf))
                    pass
                else:
                    break

            if vdf == 'y':
                with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(self.full_preprocess_data)

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
                print(data_onel)
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

        n_l0, cong_l0, incong_l0, control_l0, n_l1, cong_l1, incong_l1, control_l1 = (0, 0, 0, 0, 0, 0, 0, 0)

        l1_l2 = ['PorEng', 'EngPor']

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

        if n_l0 > 0 or n_l1 > 0: 
            # PRINT PORENG ERRORS
            print('{n_l0} errors removed from {l0} pairs.'.format(n_l0=n_l0, l0=l1_l2[0]))
            print('congruent: {cong_l0}, incongruent: {incong_l0}, control: {control_l0}.'.format(cong_l0=cong_l0, incong_l0=incong_l0, control_l0=control_l0))

            # PRINT ENGPOR ERRORS
            print('{n_l1} errors removed from {l1} pairs.'.format(n_l1=n_l1, l1=l1_l2[1]))
            print('congruent: {cong_l1}, incongruent: {incong_l1}, control: {control_l1}.'.format(cong_l1=cong_l1, incong_l1=incong_l1, control_l1=control_l1))
    
        return data.reset_index(drop=True)

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

        data = self.remove_errors(df=data)
        data = self.remove_outliers(df=data, group='incongruent')
        data = self.remove_outliers(df=data, group='congruent')
        data = self.remove_outliers(df=data, group='control')
        data = self.z_score_normalization(df=data)
        data = self.exp_normalization(df=data)
        data = self.rescaling(df=data)
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
            data = self.full_preprocess_data
        else:
            data = df

        sequence = ['incongruent', 'congruent', 'control']

        fig, axes = plt.subplots(2, 2, figsize=(16, 16), squeeze=True)

        sns.catplot(data=data, x=first_hue, y='z_score_norm', ax=axes[0, 0], order=sequence, kind='box')
        axes[0, 0].grid(axis='y', which='major')

        sns.violinplot(data=data, x=first_hue, y='response_time', hue=second_hue, split=True, ax=axes[0, 1], order=sequence, legend=False)
        axes[0, 1].grid(axis='y', which='major')


        sns.kdeplot(data[data[first_hue] == 'control']['response_time'], shade=True, alpha=.2, ax=axes[1, 0], color='g')
        sns.kdeplot(data[data[first_hue] == 'incongruent']['response_time'], shade=True, alpha=.2, ax=axes[1, 0], color='b')
        sns.kdeplot(data[data[first_hue] == 'congruent']['response_time'], shade=True, alpha=.2, ax=axes[1, 0], color='tab:orange')
        sns.kdeplot(data[data[first_hue] == 'control']['response_time'], alpha=.8, ax=axes[1, 0], color='g')
        sns.kdeplot(data[data[first_hue] == 'incongruent']['response_time'], alpha=.8, ax=axes[1, 0], color='b')
        sns.kdeplot(data[data[first_hue] == 'congruent']['response_time'], alpha=.8, ax=axes[1, 0], color='tab:orange')
        axes[1, 0].axvline(np.median(data[data[first_hue] == 'control']['response_time']), alpha=.8, ymax=.5, c='g')
        axes[1, 0].axvline(np.median(data[data[first_hue] == 'incongruent']['response_time']), alpha=.8, ymax=.5, c='b')
        axes[1, 0].axvline(np.median(data[data[first_hue] == 'congruent']['response_time']), alpha=.8, ymax=.5, c='tab:orange')
        axes[1, 0].legend(['control', 'incongruent', 'congruent'])

        cong = data[data['group'] == 'congruent']
        sns.distplot(cong['response_time'], ax=axes[1, 1], color='tab:orange')
        axes[1, 1].grid(axis='y', which='major')

        return fig, axes

    def view_data(self):
        fig, axis = self.fig, self.axis
        plt.show()

# sa = StatisticalAnalysis(n=10, save=None, view_data=None)
# print(sa.remove_errors(df=sa.subject_df))
# with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(StatisticalAnalysis(n=6, save=False).full_preprocess_data)