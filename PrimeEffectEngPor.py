from psychopy import visual, core, monitors, event, clock
from psychopy.hardware import keyboard
import pandas as pd
import numpy as np

class Experiment(object):
    def __init__(self, n, mask_case='upper', pairs_n=50, fullcross=True, conditions_n=3, mask_size=8, onelanguageorder=None,
    fullscreen=False, screen_hz=60, timeparadigm=None, kb_keys=None, useDisplay=True, monitor_name='SurfaceBook2-manual'):
        # while True:
        #     try:
        #         n = int(input('Please enter the number of the volunteer: '))
        #         break
        #     except ValueError:
        #         print("Oops!  That was no valid number.  Try again...")

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
        self.useDisplay = useDisplay

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

        def set_monitor_window():
            mon = monitors.Monitor(monitor_name)

            if not useDisplay:
                return mon, None

            if self.fullscreen:
                win = visual.Window(monitor=mon, fullscr=True, units=['cm', 'norm'])
            else:
                win = visual.Window(size=[1200, 800], monitor=mon, units=['cm', 'norm'])

            return mon, win

        self.mon, self.win = set_monitor_window()

        def frame_duration():
            ms_paradigm = self.timeparadigm
            screen_hz = self.screen_hz
            ms_frame = 1000 / screen_hz
            frame_paradigm = {'fixation' : np.round((ms_paradigm['fixation'] / ms_frame)), 'back_mask' : np.round((ms_paradigm['back_mask'] / ms_frame)), 
            'prime' : np.round((ms_paradigm['prime'] / ms_frame)), 'forward_mask' : np.round((ms_paradigm['forward_mask'] / ms_frame))}
            return frame_paradigm

        self.frame_paradigm = frame_duration()

        # Generators

        def stimulus_generator():
            if not useDisplay:
                return None, None, None, None, None

            fixation = visual.ShapeStim(self.win, 
                vertices=((0, -0.5), (0, 0.5), (0,0), (-0.5,0), (0.5, 0)),
                lineWidth=5,
                closeShape=False,
                lineColor="white",
                units='cm'
            )
            back_mask = visual.TextStim(self.win, text='', units='cm')
            prime = visual.TextStim(self.win, text='', units='cm')
            forward_mask = visual.TextStim(self.win, text='', units='cm')
            target = visual.TextStim(self.win, text='', units='cm')

            return fixation, back_mask, prime, forward_mask, target

        self.fixation, self.back_mask, self.prime, self.forward_mask, self.target = stimulus_generator()

        def clock_generator():
            monitorclock = clock.Clock()
            return monitorclock

        self.monitorclock = clock_generator()

        def hardware_generator():
            kb = keyboard.Keyboard()

            return kb

        self.kb = hardware_generator()

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


##############################################################################################################################################

    def confirmDisplaySet(self):
        horz_leng = np.random.choice(np.arange(1, 10, 0.5)) / 2
        virt_leng = np.random.choice(np.arange(1, 10, 0.5)) / 2

        horz_line = visual.Line(self.win, start=(-horz_leng, 0), end=(horz_leng, 0), units='cm')
        virt_line = visual.Line(self.win, start=(0, -virt_leng), end=(0, virt_leng), units='cm')


        horz_line.draw()
        self.win.flip()
        while True:
            try:
                value_horz = float(input('Please type the size (in "cm") of the horizontal line displayed: '))
                break
            except ValueError():
                print('The format used to describe the size is wrong,\nplease type the correct size in "cm": ')

        virt_line.draw()
        self.win.flip()
        while True:
            try:
                value_virt = float(input('Please type the size (in "cm") of the vertical line displayed: '))
                break
            except ValueError():
                print('The format used to describe the size is wrong,\nplease type the correct size in "cm": ')

        if value_horz == horz_leng * 2 and value_virt == virt_leng * 2:
            print('Monitor configuration is CORRECT, you can proceed with the trials.')
        else:
            raise Exception('Monitor configuration is WRONG, please stop the trials until corrected.')

    def startTrial(self, order, full):
        """:Parameters:
        order must be a string with the position of the trial.
        order = ['first', 'second', 'third']
        """

        if not self.useDisplay:
            return None, None

        # IF NOT FULL, CHOOSE THE NUMBER OF TRIALS
        if not full:
            while True:
                try:
                    limit = int(input('Please, insert the number of trials: '))
                    break
                except ValueError:
                    print("Oops!  That was no valid number.  Try again...")

        # LOAD PRIME-TARGET DATAFRAME:
        if order == 'first':
            prime_target_df = self.first_sequence
        elif order == 'second':
            prime_target_df = self.second_sequence
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

        # RECORD DATA TRIALS
        trials_data = pd.DataFrame(columns=['prime', 'target', 'class', 'pair_index', 'mask', 'key_name', 'correct', 'key_rt', 'key_tDown', 'bm_dur', 'prime_dur', 'fm_dur', 'target_dur', 'frame_duration'])
        columns_trial = list(trials_data.columns)

        # EXPERIMENT LOOP
        for trialN in np.arange(self.language_n): 
            # STIMULUS PREPARATION
            # self.back_mask.text = mask_df['mask'][trialN]
            self.prime.text = prime_target_df[columns_pt[0]][trialN]
            self.forward_mask.text = mask_df['mask'][trialN]
            self.target.text = prime_target_df[columns_pt[1]][trialN]

            # DATA VARIABLES
            tClass, pair_index = prime_target_df[columns_pt[3]][trialN], prime_target_df[columns_pt[4]][trialN]

            # RESET MONITOR CLOCK
            self.monitorclock.reset()

            # TRIAL LOOP
            for frameN in np.arange(total_duration_f + 1):
                if frameN < fixation_end:
                    self.fixation.draw()
                elif frameN == fixation_end:
                    back_mask_onset = self.monitorclock.getTime()
                    self.back_mask.draw()
                elif frameN < back_mask_end:
                    self.back_mask.draw()
                elif frameN == back_mask_end:
                    prime_onset = self.monitorclock.getTime()
                    self.prime.draw()
                elif frameN < prime_end:
                    self.prime.draw()
                elif frameN == prime_end:
                    forward_mask_onset = self.monitorclock.getTime()
                    self.forward_mask.draw()
                elif frameN < forward_mask_end:
                    self.forward_mask.draw()
                else:
                    got_keypress = False
                    target_onset = self.monitorclock.getTime()
                    self.target.draw()
                    self.kb.clock.reset()
                    while not got_keypress:
                        self.target.draw()
                        key = self.kb.getKeys(keyList=('z', 'm'))
                        if key:
                            got_keypress = True
                        self.win.flip()

                    # COLLECT TRIAL DATA
                    frame_dur = self.win.getActualFrameRate(10, 40, 0, 1)

                    time_data = {
                        'back_mask_dur' : back_mask_onset,
                        'prime_dur' : prime_onset - back_mask_onset,
                        'forward_mask_dur' : forward_mask_onset - prime_onset,
                        'target_dur' : target_onset - forward_mask_onset
                    }

                    trials_data = trials_data.append({
                        columns_trial[0] : self.prime.text,
                        columns_trial[1] : self.target.text,
                        columns_trial[2] : tClass,
                        columns_trial[3] : pair_index,
                        columns_trial[4] : self.back_mask.text,
                        columns_trial[5] : key[0].name,
                        columns_trial[6] : None,
                        columns_trial[7] : key[0].rt,
                        columns_trial[8] : key[0].tDown, 
                        columns_trial[9] : time_data['back_mask_dur'],
                        columns_trial[10] : time_data['prime_dur'],
                        columns_trial[11] : time_data['forward_mask_dur'],
                        columns_trial[12] : time_data['target_dur'],
                        columns_trial[13] : frame_dur
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

        frame_dur = self.win.monitorFramePeriod

        return trials_data, frame_dur


test = Experiment(4, useDisplay=True)
# print(test.kb_key_response)
# test.startTrial()
# print(test.first_sequence, test.second_sequence)
# print(len(test.mask_list))
# test.confirmDisplaySet()
print(test.startTrial(order='first', full=False))

# with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(test.first_sequence, test.second_sequence)

