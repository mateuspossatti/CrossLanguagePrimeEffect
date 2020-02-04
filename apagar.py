from psychopy import visual, core, monitors, event
from psychopy.hardware import keyboard
import pandas as pd
import numpy as np

df = pd.read_csv(r'.\trials_data\subject-2.csv', index_col=0)


df.rename(columns={'class' : 'group'}, inplace=True)



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

df.to_csv(r'.\trials_data\subject-.csv')

