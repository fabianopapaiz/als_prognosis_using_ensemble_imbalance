import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter





CLASS_NAME = 'Short_Survival'
SLOW = 'Long'
RAPID = 'Short'

CODE_VALUES_1 = {
    0.0: 0,
    0.5: 1,
    1.0: 2,
}

# read a CSV file using Pandas
def read_csv(csv_file):
    return pd.read_csv(csv_file, delimiter=',')


def get_relevant_features_names():
    return [
        'Diagnosis_Delay',
        'Age_Range_at_Onset',
        'Riluzole',
        'Sex_Male',
        'Site_Onset',
        'FVC',
        'BMI',
        'Q1_Speech_slope',
        'Q2_Salivation_slope',
        'Q3_Swallowing_slope',
        'Q4_Handwriting_slope',
        'Q5_Cutting_slope',
        'Q6_Dressing_and_Hygiene_slope',
        'Q7_Turning_in_Bed_slope',
        'Q8_Walking_slope',
        'Q9_Climbing_Stairs_slope',
        'Q10_Respiratory_slope',
        'Qty_Regions_Involved',
        'Region_Involved_Bulbar',
        'Region_Involved_Upper_Limb',
        'Region_Involved_Lower_Limb',
        'Region_Involved_Respiratory',
        'Patient_with_Gastrostomy',
    ]


def get_inputs_and_output_variables(df):
    
    features_all = get_relevant_features_names()

    X    = df[features_all].copy() 
    y    = pd.DataFrame(df[CLASS_NAME].values, columns=['Survival_Group'])

    return X, y


def get_train_and_validation_data(dir_data=None, return_training_sets=False):

    if dir_data is None:
        dir_data = os.path.abspath('../data')

    # Training set
    df_train = read_csv(f'{dir_data}/train_all_data.csv')
    X_train, y_train = get_inputs_and_output_variables(df_train)

    # Validation set
    df_validation = read_csv(f'{dir_data}/validation_all_data.csv')
    X_valid, y_valid = get_inputs_and_output_variables(df_validation)


    X_all = pd.concat([X_train, X_valid])
    y_all = pd.concat([y_train, y_valid])

    return X_train, y_train, X_valid, y_valid, X_all, y_all


def get_decoded_data_frame(df, scaled=False, undo_dummy_sex_male=False, print_code_value=False, clean_feat_values=False):
    df_decoded = df.copy()

    for col in df_decoded.columns:
        series = df_decoded[col]
        df_decoded[col] = get_coded_to_string(series=series, scaled=scaled, print_code_value=print_code_value)

        if col.endswith('_Coded'):
            df_decoded.rename(columns={col: col.replace('_Coded', '')}, inplace=True)


    if undo_dummy_sex_male:
        series_sex = undo_dummy_variable(
            series=df_decoded.Sex_Male,
            map_values={'Yes': 'Male', 'No': 'Female'},
            new_col_name='Sex'
        )
        df_decoded.Sex_Male = series_sex
        df_decoded.rename(columns={'Sex_Male': 'Sex'}, inplace=True)


    if clean_feat_values:
        for c in df_decoded.columns:     
            df_decoded[c] = df_decoded[c].apply(lambda x: clean_feature_values(x))


    return df_decoded    


def get_coded_to_string(series, scaled=False, print_code_value=False):
    col = series.name

    if col in ['Diagnosis_Delay_Coded', 'Diagnosis_Delay']:
        series = series.apply(lambda x: get_coded_diagnosis_delay_to_string(x, scaled=scaled, 
                                                                            print_code_value=print_code_value))

    elif col in ['Age_Range_at_Onset_Coded', 'Age_Range_at_Onset']:
        series = series.apply(lambda x: get_coded_age_range_to_string(x, scaled=scaled, 
                                                                      print_code_value=print_code_value))

    elif col in ['Site_Onset_Coded', 'Site_Onset']:
        series = series.apply(lambda x: get_coded_site_onset_to_string(x, scaled=scaled, 
                                                                       print_code_value=print_code_value))

    elif col.startswith('FVC'):
        series = series.apply(lambda x: get_coded_fvc_to_string(x, 
                                                                print_code_value=print_code_value))

    elif col.startswith('BMI'):
        series = series.apply(lambda x: get_coded_bmi_to_string(x, scaled=scaled, 
                                                                print_code_value=print_code_value))

    elif col.startswith('Q') and '_slope' in col:
        series = series.apply(lambda x: get_coded_alsfrs_slope_to_string(x, scaled=scaled, 
                                                                         print_code_value=print_code_value))

    elif col == 'Survival_Group':
        series = series.apply(lambda x: get_coded_group_survival_to_string(x, 
                                                                           print_code_value=print_code_value))

    elif col in ['Qty_Regions_Involved_at_Diagnosis', 'Qty_Regions_Involved']:
        series = series.apply(lambda x: get_coded_qty_regions_involved_to_string(x, scaled=scaled)).astype(str)
    
    # # if was a integer column
    # if ('Qty_Regions_Involved' in col):
    #     series = series.astype(int)
        
    # if was a boolean column
    if (col in ['Riluzole', 'Sex_Male', 'Group_Short_Survival', 'Short_Survival']) \
    or ('Patient_with_Gastrostomy' in col) \
    or ('Region_Involved' in col)    :
        series = series.map({1.0: 'Yes', 0.0: 'No'})
        # series = series.astype(bool)

    return series    



def get_coded_qty_regions_involved_to_string(code, scaled=False):
    code = float(code)
    text = None    

    if scaled: #[0-1]
        if code == 0.0:
            text = 0
        elif code == 0.25:
            text = 1
        elif code == 0.5:
            text = 2
        elif code == 0.75:
            text = 3
        elif code == 1.0:
            text = 4
    else:
        text = int(code)

    return text


def get_coded_group_survival_to_string(code, print_code_value=False):
    code = float(code)

    if code == 0.0:
        text = 'Non-Short'
    elif code == 1.0:
        text = 'Short'
    else:
        text = None    

    return text


def get_coded_fvc_to_string(code, print_code_value=False):
    code = float(code)
    
    if code == 0.0:
        text = 'Normal'
    elif code == 1.0:
        text = 'Abnormal'
    else:
        text = None    

    return text

def get_coded_diagnosis_delay_to_string(code, scaled=False, print_code_value=False):
    code = float(code)
    text = None    

    if scaled: #[0-1]
        if code == 0.0:
            text = RAPID
        elif code == 0.5:
            text = 'Average'
        elif code == 1.0:
            text = SLOW

        #code
        # code = CODE_VALUES_1[code]    
    else:
        if code == 0.0:
            text = RAPID
        elif code == 1.0:
            text = 'Average'
        elif code == 2.0:
            text = SLOW

    return text if not print_code_value else f'({code}) {text}'


def get_coded_alsfrs_slope_to_string(code, scaled=False, print_code_value=False):
    code = float(code)
    text = None    

    if scaled: #[0-1]
        if code == 0.0:
            text = 'Slow'
        elif code == 0.5:
            text = 'Average'
        elif code == 1.0:
            text = 'Rapid'
        #code
        code = CODE_VALUES_1[code]    
    else:
        if code == 0.0:
            text = 'Slow'
        elif code == 1.0:
            text = 'Average'
        elif code == 2.0:
            text = 'Rapid'

    return text if not print_code_value else f'({int(code)}) {text}'


def get_coded_age_range_to_string(code, scaled=False, print_code_value=False):
    code = float(code)

    text = None
    # [0.0, 0.25, 0.5, 0.75, 1.0]
    if scaled:
        if code == 0.0:
            text = '0-39'  if not print_code_value else '(0) 0-39'
        elif code == 0.25:
            text = '40-49' if not print_code_value else '(1) 40-49'
        elif code == 0.5:
            text = '50-59' if not print_code_value else '(2) 50-59'
        elif code == 0.75:
            text = '60-69' if not print_code_value else '(3) 60-69'
        elif code == 1.0:
            text = '70+'   if not print_code_value else '(4) 70+'
    else:
        if code == 0.0:
            text = '0-39'  if not print_code_value else '(0) 0-39'
        elif code == 1.0:
            text = '40-49' if not print_code_value else '(1) 40-49'
        elif code == 2.0:
            text = '50-59' if not print_code_value else '(2) 50-59'
        elif code == 3.0:
            text = '60-69' if not print_code_value else '(3) 60-69'
        elif code == 4.0:
            text = '70+'   if not print_code_value else '(4) 70+'

    return text


def get_coded_site_onset_to_string(code, scaled=False, print_code_value=False):
    code = float(code)

    text = None    

    ## NOTE ONLY HAVE 'BULBAR' AND 'LIMB/SPINAL' IN DATASET (n=1967)

    if code == 0.0:
        text = 'Bulbar'
    elif code == 1.0:
        text = 'Limb/Spinal'
    elif code == 2.0:
        text = 'Other'

    return text


def get_coded_bmi_to_string(code, scaled=False, print_code_value=False):
    code = float(code)

    text = None    

    if scaled: #[0-1]
        if code == 0.0:
            text = '(0) Severely underweight'
        elif code == 0.25:
            text = '(1) Underweight'
        elif code == 0.5:
            text = '(2) Normal weight'
        elif code == 0.75:
            text = '(3) Overweight'
        elif code == 1.0:
            text = '(4) Obesity'
    else:
        if code == 0.0:
            text = '(0) Severely underweight'
        elif code == 1.0:
            text = '(1) Underweight'
        elif code == 2.0:
            text = '(2) Normal weight'
        elif code == 3.0:
            text = '(3) Overweight'
        elif code == 4.0:
            text = '(4) Obesity'
  
    return text



def undo_dummy_variable(series, map_values, new_col_name):
    
    series_aux = series.copy()
    
    values = series_aux.unique()

    series_return = pd.Series(data=series_aux.values, name=new_col_name)

    series_return = series_return.map(map_values)

    return series_return


def clean_feature_values(value):
    try:
        new_value = value.split(')')[1].strip()

    except Exception as ex: 
        new_value = value
    
    return new_value


# Plot the distributions of informed variables (columns)
def plot_variables_distributions(df, columns=None, zero_and_one_as_categorical=True, bins=30, 
    figsize=[15,5], print_more_info=True, plot_one_graph_per_row=False, 
    plot_pareto_graph=True, fill_nan=False, plot_percentage_lines=True):
    
    #if dont set columns, get all coumns from dataset
    if columns is None:
        columns = list(df.columns.values)
        if 'subject_id' in columns:
            columns.remove('subject_id')

    for col in columns:
        plot_variable_distribution(df=df, column=col, zero_and_one_as_categorical=zero_and_one_as_categorical, bins=bins, figsize=figsize,
                                   print_more_info=print_more_info, plot_one_graph_per_row=plot_one_graph_per_row,
                                   fill_nan=fill_nan, plot_pareto_graph=plot_pareto_graph,
                                    plot_percentage_lines=plot_percentage_lines,
                                   )


# Plot the distribution of one given column
def plot_variable_distribution(series=None, df=None, column=None, zero_and_one_as_categorical=True, 
        bins=30, figsize=[15,5], print_more_info=True, 
        plot_one_graph_per_row=False, plot_pareto_graph=True, fill_nan=False,
        plot_percentage_lines=True):
    
    if column != 'subject_id':
        plot_histogram_and_boxplot(df=df, column=column, series=series, bins=bins,
                                figsize=figsize, 
                                print_more_info=print_more_info, plot_one_graph_per_row=plot_one_graph_per_row,
                                zero_and_one_as_categorical=zero_and_one_as_categorical,
                                fill_nan=fill_nan,
                                plot_pareto_graph=plot_pareto_graph,
                                plot_percentage_lines=plot_percentage_lines,
                                )


# Plot the distributions of informed [dataFrame + column] or [Series]
def plot_histogram_and_boxplot(df=None, column=None, series=None, zero_and_one_as_categorical=True, 
        bins=30, figsize=[15,5], orientation='horizontal',  
        print_more_info=True, plot_one_graph_per_row=False, plot_pareto_graph=True, fill_nan=False,
        plot_percentage_lines=True):

    series = get_series_from_parameters(df=df, column=column, series=series)

    total_of_rows = series.count()

    series_dtype = series.dtype

    if (plot_one_graph_per_row) and (figsize==[15,5]):
        figsize=[15, 13]


    # lead with boolean values on data_column
    if series.dtype == bool:
        series = series.map({True: 'YES', False: 'NO'})

    # check if zero-and-one columns must be treated as categorical (Yes/No)
    if zero_and_one_as_categorical:
        #get unique values
        unique_values = series.unique()
        #check if have only 2 elements
        if get_quantity_of_rows(unique_values) == 2:
            #compares with zer0-one lists with different orders
            comparison_1 = unique_values == np.array([1, 0])
            comparison_2 = unique_values == np.array([0, 1])
            #if some comparison was True, then set YES/NO values
            if comparison_1.all() or comparison_2.all():
                series = series.map({1: 'YES', 0: 'NO'})

    # print more info about distribution
    if print_more_info:
        print_variable_distribution(df=df, column=column, series=series, fill_nan=fill_nan)
    else:
        s_name = series.name  # .upper()

        print(f'==============================================================================')
        print(f'Column {s_name}  ({total_of_rows} rows)   (DataType: {series_dtype})')
        print(f'==============================================================================')


    #if variable is CONTINUOUS or INTEGER
    if series.dtype in [float, int]:
        # --------------------------------------------
        # HISTOGRAM ----------------------------------
        # --------------------------------------------
        plt.figure(figsize=figsize)
        # chech if is to plot the graphs in [1 row X 2 cols] or in [1 col X 2 rows]
        plt.subplot(2, 1, 1) if plot_one_graph_per_row else plt.subplot(1, 2, 1)
        #
        plt.title(f'{series.name.upper()}')
        ax = series.hist(bins=bins)

        # --------------------------------------------
        # BOXPLOT GRAPH
        # --------------------------------------------
        # chech if is to plot the graphs in [1 row X 2 cols] or in [1 col X 2 rows]
        plt.subplot(2, 1, 2) if plot_one_graph_per_row else plt.subplot(1, 2, 2)
        #
        ax = sns.boxplot(data=series.values, orient=orientation)

    #if variable is CATEGORICAL
    else:
        if fill_nan:
            series.fillna('**Not Informed', inplace=True)    

        #create a dataframe to plot the data
        df_distrib = pd.DataFrame(series.value_counts())
        df_distrib = df_distrib.reset_index()
        sn = series.name
        df_distrib[sn] = df_distrib[sn].astype(float)
        df_distrib.rename(columns={series.name: 'count', 'index': sn}, inplace=True)
        sum = df_distrib['count'].sum()

        df_distrib = df_distrib.sort_values(by=sn)

        df_distrib['x'] = '' + df_distrib[sn].astype(str)
        df_distrib['percentage'] = np.round(df_distrib['count'] / sum * 100, 2)
        df_distrib['percentage'] = df_distrib['percentage'].map('{:,.2f}%'.format)
        df_distrib['cum_sum'] = df_distrib['count'].cumsum()
        df_distrib['cum_percentage'] = df_distrib['count'].cumsum() / df_distrib['count'].sum() * 100

        # --------------------------------------------
        # BAR GRAPH (WITH QUANTITY AND PERCENTAGE)
        # --------------------------------------------
        plt.figure(figsize=figsize)
        # chech if is to plot the graphs in [1 row X 2 cols] or in [1 col X 2 rows]
        ax = plt.subplot(2, 1, 1) if plot_one_graph_per_row else plt.subplot(1, 2, 1)
        #
        x = df_distrib['x']
        y = df_distrib['count']
        y_cum = df_distrib['cum_percentage']

        # plot bar graph
        plt.title(f"{series.name.upper()}\n")
        sns.barplot(x=x, y=y, ax=ax)
        ax.set_xlabel('')
        plt.xticks(rotation=15)

        show_quantity_and_percentage_on_bars(ax=ax, total_of_rows=sum)

        #plot cumulative line
        if plot_percentage_lines:
            ax2 = ax.twinx()
            ax2.plot(x, y_cum, color='blue', marker='D', ms=4, alpha=0.5)
            ax2.yaxis.set_major_formatter(PercentFormatter())
            ax2.spines['right'].set_color('#a70000')
            ax2.tick_params(axis='y', colors='blue', grid_color='blue', grid_alpha=0.5, grid_linestyle='--')


        if plot_pareto_graph:
            # --------------------------------------------
            # PARETO GRAPH
            # --------------------------------------------
            # calculate the cumulative percentage
            df_distrib = df_distrib.sort_values(by='count', ascending=False)
            df_distrib['cum_percentage'] = df_distrib['count'].cumsum() / df_distrib['count'].sum() * 100

            # chech if is to plot the graphs in [1 row X 2 cols] or in [1 col X 2 rows]
            plt.subplot(2, 1, 2) if plot_one_graph_per_row else plt.subplot(1, 2, 2)
            #
            plt.title(f'Cumulative Percentage (Pareto Graph)\n')
            plt.xticks(rotation=15)


            # Plot Pareto graph
            x = df_distrib['x']
            y = df_distrib['count']
            ax = sns.barplot(x=x, y=y)
            # ax.set_xlabel(sn)
            ax.set_xlabel('')

            #
            show_quantity_and_percentage_on_bars(ax=ax, total_of_rows=sum)

            # Plot lines with cumulative percentage
            if plot_percentage_lines:
                ax2 = ax.twinx()
                y_cum_perc = df_distrib['cum_percentage']
                ax2.plot(x, y_cum_perc, color='#a70000', marker='D', ms=5, alpha=0.5)
                ax2.yaxis.set_major_formatter(PercentFormatter())
                ax2.spines['right'].set_color('#a70000')
                ax2.tick_params(axis='y', colors='red', grid_color='red', grid_alpha=0.5, grid_linestyle='--')



    plt.show()


# show quantity and percentage at the top of each bar on graph
def show_quantity_and_percentage_on_bars(ax, total_of_rows, fontsize=11, format='.2f',
                                         show_only_percentage=False, show_only_quantity=False):
    for p in ax.patches:
        value = p.get_height()
        perc = np.round((value / total_of_rows * 100), 2)

        if show_only_percentage:
            text = f'{perc:.0f}%'
        elif show_only_quantity:
            text = f'{value:.0f}%'
        else:
            text = f'{value:.0f}\n{perc:.0f}%'

        ax.annotate(text, (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=fontsize, color='black', xytext=(0, 13),
                    textcoords='offset points')



# check errors and return series from parameters
# ** AUXILIARY METHOD USED ONLY INTERNALLY **
def get_series_from_parameters(df=None, column=None, series=None):
    if df is not None:
        if column is None:
            raise NameError('Parameter COLUMN must be informed')

        df = df.copy()
        series = df[column]
    #
    elif series is not None:
        series = series.copy()
    else:
        raise NameError('Must inform [DF + COLUMN] parameters or [SERIES] parameter')

    return series


# Print the distribution of one given column
def print_variable_distribution(df=None, column=None, series=None, fill_nan=False):

    series = get_series_from_parameters(df=df, column=column, series=series)

    s_name = series.name #.upper()

    # get total of samples
    qty_samples = series.shape[0]
    # get total of samples without NaN values
    qty_samples_non_nan = series.count()
    # get total of samples without NaN values
    qty_samples_nan = qty_samples - qty_samples_non_nan
    perc_samples_nan = np.round( (qty_samples_nan / qty_samples * 100), 2)



    print(f'==============================================================================')
    print(f'Column: {s_name}  (DataType: {series.dtype})')
    print(f'==============================================================================')

    N = series.count()
    print(f'N         = {N} [Non-NaN: {qty_samples_non_nan}]')
    print(f'Missingness: {qty_samples_nan} ({perc_samples_nan}%)')

    #if variable is CONTINUOUS or INTEGER
    if series.dtype in [float, int]:
        if series.mode().count() > 10:
            mode = ' *** TOO MANY VALUES FOR Mode (length = %s)' % series.mode().count()
        else:
            mode = str(series.mode().to_list()).replace('[', '').replace(']', '').replace(',', ';')

        #
        min = series.min()
        max = series.max()
        mean = series.mean()
        std_dev = series.std()
        variance = series.var()
        coef_variation = std_dev / mean
        skewness = get_skewness(series)
        kurtosis = get_kurtosis(series)
        sem = std_dev / np.sqrt(N)
        # print more information about distribution
        more_info = 'Min       = %.2f  ' \
                    '\nMax       = %.2f  ' \
                    '\nSE / Std  = %.2f  (Standard Error or Std.Dev.)' \
                    '\nSEM       = %.2f  (Standard Error of the Mean) [Formula: Std.Dev/sqrt(N) => %.2f/sqrt(%s)] ' \
                    '\nMean      = %.2f +/- %.2f (Std.Dev.) [Precision of the Mean = %.2f +/- %.2f (SEM)] ' \
                    '\nMedian    = %.2f  ' \
                    '\nMode      = %s'  \
                    '\nVariance  = %.2f (Coefficient of Variation = %.2f)' \
                    '\nSkewness  = %s '  \
                    '\nKurtosis  = %s ' % \
                    (min,
                     max,
                     std_dev,
                     sem,
                     std_dev,
                     N,
                     mean,
                     std_dev,
                     mean,
                     sem,
                     series.median(),
                     mode,
                     variance,
                     coef_variation,
                     skewness,
                     kurtosis
                     )
        q25 = series.quantile(q=0.25)
        q50 = series.quantile(q=0.5)
        q75 = series.quantile(q=0.75)
        more_info += '\nQuartiles = [Q1: 25%% < %.2f]   [Q2: 50%% < %.2f]   [Q3: 75%% < %.2f]' % \
                    (q25, q50, q75)

        Q1 = q25
        Q3 = q75
        IQR = Q3 - Q1

        outliers_threshold_inf = (Q1 - 1.5 * IQR)
        if outliers_threshold_inf < min:
            outliers_threshold_inf = min

        outliers_threshold_sup = (Q3 + 1.5 * IQR)
        if outliers_threshold_sup > max:
            outliers_threshold_sup = max

        more_info += f'\n -IQR     = {IQR:.3f} (Interquartile Range: IQR = Q3-Q1)'
        more_info += f'\n -Outliers Threshold (IQR * +/-1.5):  [Lower = {outliers_threshold_inf:.3f}]   [Upper = {outliers_threshold_sup:.3f}]'

        print(more_info)

    else:
        if fill_nan:
            series.fillna('**Not Informed', inplace=True)    

        # print(f'N         = {N} {column} (rows with value, excluding NaN´s)')
        print('\nSummary Table:')
        # df_distrib = pd.DataFrame(df[column].value_counts())
        df_distrib = pd.DataFrame(series.value_counts())
        sum = df_distrib[series.name].sum()
        df_distrib['percentage'] = np.round(df_distrib[series.name] / sum * 100, 2)
        df_distrib['percentage'] = df_distrib['percentage'].map('{:,.2f}%'.format)
        df_distrib.rename(columns={series.name: 'count'}, inplace=True)
        # df_distrib['count'] = df_distrib['count'].map('{:,.0f}'.format)

        print(df_distrib)


 # return a string with Skewness info of given series
def get_kurtosis(series, short_description=False):
    kurtosis = series.kurtosis()
    ret = '%.2f' % kurtosis
    #
    if short_description:
        if kurtosis > 0:
            ret += ' (point head)'
        elif kurtosis < 0:
            ret += ' (rounded)'
        else:  # kurtosis = 0:
            ret += ' (normal)'
    #
    else:
        if kurtosis > 0:
            ret += ' (leptokurtic, point head appearance)'
        elif kurtosis < 0:
            ret += ' (platykurtic, rounded appearance)'
        else: # kurtosis = 0:
            ret += ' (mesokurtic, normal appearance)'
        #
        ret += '(PS: a high kurtosis indicates too many outliers)'
    #
    return ret



# return a string with Skewness info of given series
def get_skewness(series, short_description=False):
    skewness = series.skew()
    ret = '%.2f' % skewness
    #
    if short_description:
        if skewness < 0:
            ret += ' (Left)'
        elif skewness > 0:
            ret += ' (Right)'
        else: # skewness = 0:
            ret += ' (Symmetric)'
    #
    else:
        if skewness < 0:
            ret += ' (Left-Skewed)'
        elif skewness > 0:
            ret += ' (Right-Skewed)'
        else: # skewness = 0:
            ret += ' (Symmetric, Unskewed)'
        #
        if (skewness < -1.0) or (skewness > 1.0):
            ret += ' (Highly Skewed)'
        elif (skewness < -0.5) or (skewness > 0.5):
            ret += ' (Moderately Skewed)'
        else:
            ret += ' (Approximately Symmetric)'
    #
    return ret


# get total of samples in dataset
def get_quantity_of_rows(df):
    return df.shape[0]