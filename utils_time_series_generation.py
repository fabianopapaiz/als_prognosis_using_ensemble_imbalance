import math
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from datetime import datetime
from datetime import timedelta
from dateutil import relativedelta

import utils


def generate_time_series_fvc(df_temporal, dir_dest):
    pass


def generate_time_series_alsfrs(df_temporal, dir_dest):

    # Get values by month up to 72 months (i.e., 6 years)
    n_years = 10
    threshold = 12 * n_years # "n" years
    months = np.linspace(0, threshold, threshold+1, dtype=float) #[1.0, 2.0, 3.0,..., 72.0]


    baselines = [
        'Symptoms_Onset'
    ]

    columns_questions_ALSFRS = [
        'Q1_Speech',
        'Q2_Salivation',
        'Q3_Swallowing',
        'Q4_Handwriting',
        'Q5_Cutting',
        'Q6_Dressing_and_Hygiene',
        'Q7_Turning_in_Bed',
        'Q8_Walking',
        'Q9_Climbing_Stairs',
        'Q10_Respiratory',
    ]

    columns_not_to_interpolate = [
        'Q1_Speech',
        'Q2_Salivation',
        'Q3_Swallowing',
        'Q4_Handwriting',
        'Q5_Cutting',
        'Q6_Dressing_and_Hygiene',
        'Q7_Turning_in_Bed',
        'Q8_Walking',
        'Q9_Climbing_Stairs',
        'Q10_Respiratory',
        #
        'Region_Involved_Bulbar',
        'Region_Involved_Upper_Limb',
        'Region_Involved_Lower_Limb',
        'Region_Involved_Respiratory',
        'Qty_Regions_Involved',
        # boolean columns
        'Patient_with_Gastrostomy',

    ]
    columns_to_interpolate = [
        #
        'Slope_from_Onset_Q1_Speech',
        'Slope_from_Onset_Q2_Salivation',
        'Slope_from_Onset_Q3_Swallowing',
        'Slope_from_Onset_Q4_Handwriting',
        'Slope_from_Onset_Q5_Cutting',
        'Slope_from_Onset_Q6_Dressing_and_Hygiene',
        'Slope_from_Onset_Q7_Turning_in_Bed',
        'Slope_from_Onset_Q8_Walking',
        'Slope_from_Onset_Q9_Climbing_Stairs',
        'Slope_from_Onset_Q10_Respiratory',
    ]

    columns = columns_not_to_interpolate + columns_to_interpolate

    # dir_dest = os.path.abspath('../03_preprocessed_data/')


    for baseline in baselines:
        
        for column in columns:

            #dont process the score, only the slopes
            if column in columns_questions_ALSFRS:
                continue

            col_baseline = f'Delta_from_{baseline}'

            # copy data ordering by col_baseline
            df_copy = df_temporal.sort_values(by=['subject_id', col_baseline]).copy()


            # convert boolean values to 0/1 for Boolean cloumns
            if (column == 'Patient_with_Gastrostomy') | (column.startswith('Region_Involved_')):
                df_copy[column].replace({True: 1, False: 0}, inplace=True)
                
            # drop rows with NaN in "col_baseline" and "column"
            df_copy.dropna(
                subset=[
                    col_baseline, 
                    column,
                ], 
                inplace=True
            )

            # filter rows by threshold
            df_pivot = df_copy.copy()

            # get only the names of the Values columns 
            cols_to_pivot = df_pivot.columns[2:]

            # create pivot by column Result
            df_aux = df_pivot.pivot_table(
                index='subject_id', 
                columns=col_baseline, 
                values=column,
                aggfunc=np.max, # get max value in that month (can exist 2+ measurements for a same month)
            )

            # reset index
            df_aux.reset_index(inplace=True)

            # get the month-columns existing in the pivot-table
            cols_months = df_aux.columns[1:]

            # check if all months columns were created [1-72]
            for month in months:
                # if month not present in the columns
                if month not in cols_months:
                    # Creating column for this month and set its values to NaN
                    # PS: "int(month)" is used to keep columns ordered by month number
                    df_aux.insert(int(month), month, np.NaN)

            # code to ensure the order of the columns
            cols_months_ordered = list(sorted(months))
            cols_months_ordered.insert(0, 'subject_id')
            df_aux = df_aux[cols_months_ordered]
            
            
            round_decimal_places = 2
            
            # if column is a slope Total-Score from onset, set month-0 = 0.0 (none decline or increase)
            if (column == 'Slope_from_Onset_Total_Score') | ('Slope_from_Onset_Q' in column):
                df_aux[0.0] = 0.0
            # if column is a ALSFRS question, set month-0 = 4 (Max score for each ALSFRS question)
            elif (column in columns_questions_ALSFRS):
                df_aux[0.0] = 4.0
                round_decimal_places = 0
            # if column is to do not interpolate, set round_decimal_places = 0
            elif (column in columns_not_to_interpolate):
                round_decimal_places = 0

            
            col_name = column.replace('_from_Onset', '')

            # read file saved to fill NaN values using interpolation
            df_fill_nan_using_interpolation = df_aux

            # get columns ignoring 'subject_id'
            cols_months = df_fill_nan_using_interpolation.columns[1:]


            # perform Missing Imputation using interpolation
            df_aux = df_fill_nan_using_interpolation[cols_months].interpolate(
                method='linear', 
                limit_direction='both',
                limit=1000, 
                axis=1, 
                inplace=False,
            ).copy()
            
            # round Values using "round_decimal_places" variable
            df_aux[cols_months] = np.round(df_aux[cols_months], round_decimal_places)
            
            # get subject_id column
            df_fill_nan_using_interpolation[cols_months] = df_aux[cols_months]

            # drop rows with NaN values (where there is no Value registered)
            df_fill_nan_using_interpolation.dropna(inplace=True)

            # save data again for each Value column with interpolation
            print(f'{col_name}')
            csv_file = f'{dir_dest}/TimeSeries/ALSFRS/ALSFRS_TimeSeries_{col_name}.csv'
            utils.save_to_csv(df=df_fill_nan_using_interpolation, csv_file=csv_file)

            # just for further tests
            df_aux = df_fill_nan_using_interpolation.copy()

            print()

