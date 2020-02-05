from psychopy import visual, core, monitors, event
from psychopy.hardware import keyboard
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# TEST AREA

# test = Experiment(2, useDisplay=True)
# print(test.startTrial('first', False))
# print(test.startExperiment(True, True))

# with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(test.startExperiment(False))


df = pd.read_csv(r'.\trials_data\subject-1-norm.csv', index_col=0)

eng_prime_df = df[df['l1_l2'] == 'EngPor']
por_prime_df = df[150:].reset_index(drop=True)

# print(eng_prime_df)
# for i in range(150):
#     if por_prime_df['correct'][i] == True:
#         por_cor += 1
#     if eng_prime_df['correct'][i] == True:
#         eng_cor += 1

# print(por_cor/150, eng_cor/150)
# print(df.columns)

sequence = ['congruent', 'incongruent', 'control']

sns.catplot(data=eng_prime_df, x='group', y='response_time', order=sequence, kind='box')
# ax.grid(axis='y', which='major')

plt.show()

# df = pd.read_csv(r'.\trials_data\subject-11.csv', index_col=0)

# bol = []
# n = 0

# for i in range(df.shape[0]):
#     if df['response_time'][i] <= 0:
#         bol.append(False)
#         n += 1
#     else:
#         bol.append(True)

# df = df[bol].reset_index(drop=True)

# with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(df, n)

# df.to_csv(r'.\trials_data\subject-13.csv')