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



df = pd.read_csv(r'.\trials_data\subject-10.csv', index_col=0)

bol = []

for i in range(df.shape[0]):
    if df['response_time'][i] <= 0:
        bol.append(False)
    else:
        bol.append(True)

df = df[bol].reset_index(drop=True)

with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(df)

# df.to_csv(r'.\trials_data\subject-10.csv')