from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
class StatisticalAnalysis:
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

StatisticalAnalysis()