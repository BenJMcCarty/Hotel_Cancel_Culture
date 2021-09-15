'''Name: EDA

Description: Functions created to assist with data cleaning and exploratory data analysis (EDA).

By Ben McCarty (bmccarty505@gmail.com)'''

import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

## 
def sort_report(Source, Sort_by, Show_Only_Missing = False, Drop_Cols = False, Cols = ['N/A - Specify Columns'], Highlight_All=False, Ascending_Values = False, Color='#d65f5f'):
    """Function for sorting results of "report_df" function - requires "report_df" to be imported.

    Args:
        Source (string): Link to dataset - either local or remote
        Sort_by (list): List of values to use for sorting the resulting dataframe values.
        Show_Only_Missing (bool, optional): Show only features with missing values. Defaults to False.
        Drop_Cols (bool, optional): Drop specific columns. Defaults to False.
        Cols (list, optional): List of columns to drop; only works when Drop_Cols is True. Defaults to ['Keep all columns'].
        Highlight_All (bool, optional): Apply cell highlighting based on values. Defaults to False.
        Ascending_Values (bool, optional): Sorting results displayed in ascending order. Defaults to False.
        Color (str, optional): Color used for bars in cells. Defaults to '#d65f5f'.
    """

    df = pd.read_csv(Source)


    try:
        if Drop_Cols is True:
            df = df.drop(columns = Cols).copy()
        else:
            df = df
    except Exception:
        e = Exception
        print('**No columns were dropped.** Please make sure you specify the correct column names in the "Cols" keyword argument.')

    if Highlight_All == False:
        columns = ['null_sum', 'null_pct']
    else:
        columns = list(report_df(df).select_dtypes(exclude=object).columns)


    if Show_Only_Missing == True:
        results = report_df(df)
        display(results[results['null_sum'] > 0].sort_values(by=Sort_by, ascending=Ascending_Values).style.bar(subset=columns, color=Color))
    else:
        display(report_df(df).sort_values(by=Sort_by, ascending=Ascending_Values).style.bar(subset=columns, color=Color))

def find_outliers_z(data):
    """Detects outliers using the Z-score>3 cutoff.
    Returns a boolean Series where True=outlier
    
    Source: https://github.com/jirvingphd/dsc-phase-2-project/blob/main/functions_SG.py

    Args:
        data (pd.Series): Series for which to determine the outliers via the Z-score

    Returns:
        pd.Series: Boolean index indicating "True" if a value is an outlier
    """

    zFP = np.abs(stats.zscore(data))
    zFP = pd.Series(zFP, index=data.index)
    idx_outliers = zFP > 3

    return idx_outliers

def id_num_cols(data, drop_cols_list = None):
    """Filters a given Series index/DataFrame columns for numeric datatypes.

    Args:
        data (Series or DataFrame): dataset to filter
        drop_cols_list (list of strings, optional): Optional list of strings to drop from results. Defaults to None.

    Returns:
        temp_indx [list of strings]: index/column labels for numeric columns
    """    

    temp_idx = data.columns[(data.dtypes == 'int64') | (data.dtypes == 'float64')]
    
    if drop_cols_list != None:
        temp_idx = temp_idx.drop(drop_cols_list)

    temp_idx = list(temp_idx)

    return temp_idx

def outlier_percentage(data):
    """Identifies numerical columns in a given Series/DataFrame, then calculates the percentage of values
     exceeding an absolute-valued z-score of 3.

    Displays the resulting dataframe using Pandas styling with a darkening gradient to highlight the highest percentages

    Args:
        data (Series or DataFrame): Source data.

    Returns:
        DataFrame: non-styled dataframe containing column names (as index) and outlier percentages.
    """    

    list_col_names = id_num_cols(data)
    outlier_results = {}
    for col in list_col_names:
        outlier_results[col] = round(round(find_outliers_z(data[col]).sum()/len(data[col]), 4)*100,2)

    pct_df = pd.DataFrame(pd.Series(outlier_results), columns= ['% Outliers'])
    
    display(pct_df.style.background_gradient().format("{:,.2f}"))

    return outlier_results

def filter_outliers_zscore(data):

    drop_idx = set()

    num_idx = data.columns[(data.dtypes == 'int64') | (data.dtypes == 'float64')]

    for col in num_idx:
        temp_idx = find_outliers_z(data[col])
        drop_idx = data[col][temp_idx].index

    return drop_idx

def find_outliers_IQR(data):
    """Determines outliers using the 1.5*IQR thresholds.

    Source: https://github.com/jirvingphd/dsc-phase-2-project/blob/main/functions_SG.py

    Args:
        data (pd.Series): [description]

    Returns:
        pd.Series: Boolean Series where True=outlier
    """    
        
    res = data.describe()
    q1 = res['25%']
    q3 = res['75%']
    thresh = 1.5*(q3-q1)
    idx_outliers =(data < (q1-thresh)) | (data > (q3+thresh))
    return idx_outliers

def feature_vis(data, x, y, categorical = False, kde = True):
    """
    Prints the selected Series for reference.
    
    Creates two plots via Seaborn:

        * Scatter plot with regression line
        * Histogram of the data
            * Optional KDE ifkde = True

    Args:
        data (pd.DataFrame): Source dataframe
        x (pd.Series): Independent variable for visualizations
        y (str, optional): Dependent variable. Defaults to 'price'.
        categorical (bool, optional): Indicate if the values are categorical. Defaults to False.
        kde (bool, optional): Add KDE plot to histogram. Defaults to True.
    
    Returns:
        N/a
    """

    print(data[x].value_counts().sort_index())
      
    fig, axs = plt.subplots(ncols=2, figsize= (12,6))
    
    sns.regplot(data=data, x=x, y=y, ax=axs[0])
    sns.histplot(data=data, x=x, discrete=categorical, kde=kde, ax=axs[1])
    
    fig.suptitle(f'{x.title()} vs. {y.title()}', fontsize=16)
    plt.tight_layout()
    
    return

def filter_outliers(data):
    """Filters outliers from given data via the "find_outliers_IQR" function and saves filtered
    values to a new DataFrame

    Args:
        data (pd.Series): Selected Series

    Returns:
        pd.DataFrame: DataFrame of non-outlier data filtered from original DataFrame
    """    
       
    idx_out = find_outliers_IQR(data)
 
    cleaned = data[~idx_out].copy()

    # print(f'There were {idx_out.sum()} outliers.')
    
    return cleaned

def show_cleaned_vis(data, x, y = 'price', categorical = False, kde = True):
    """Combines helper functions to filter outliers and to create the feature 
        visualizations.
    
    * Requres 'find_outliers_IQR' and 'feature_vis' to be pre-defined

    Args:
        data (pd.DataFrame): Source data
        x (str): Independent variable to visualize
        y (str, optional): Dependent variable against which to plot the independent variable. Defaults to 'price'.
        categorical (bool, optional): Indicates whether or not 'x' is categorical. Defaults to False.
        kde (bool, optional): Overlay a KDE plot. Defaults to True.

    Returns:
        None
    """

    ### Filter outliers first
    
    idx_out = find_outliers_IQR(data[x])
 
    df_cleaned = data[~idx_out].copy()

    ### Plot Data
        
    df_cleaned.value_counts().sort_index()
        
    fig, axs = plt.subplots(ncols=2, figsize= (12,6))
    
    sns.regplot(data=df_cleaned, x=x, y=y, ax=axs[0],line_kws={"color": "red"})
    sns.histplot(data=df_cleaned, x=x, discrete=categorical, kde=kde, ax=axs[1])
    
    fig.suptitle(f'{x.title()} vs. {y.title()}', fontsize=16)
    plt.tight_layout();
    
    return #df_cleaned


def corr_val(df,figsize=(15,15),cmap="OrRd",):
    """Generates a Seaborn heatmap of correlations between each independent variable.

    Args:
        df (pd.Dataframe): Source DataFrame
        figsize (tuple, optional): Size of resulting figure. Defaults to (15,15).
        cmap (str, optional): Color scheme. Defaults to "OrRd".

    Returns:
        fig, ax: resulting visualization
    """

    # Calculate correlations
    corr = df.corr()
       
    # Create a mask of the same size as our correlation data
    mask = np.zeros_like(corr)
    
    # Set the upper values of the numpy array to "True" to ignore them
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots(figsize=figsize)
    
    # Mask=mask to hide the upper-right half of values (otherwise mirrored)
    sns.heatmap(corr, annot=True,cmap="Reds",mask=mask)
    return fig, ax


def report_df(dataframe):
    """Creates a summary of a given dataframe per column, including:
        * Datatypes
        * Number of unique values
        * Number of NaN values
        * Percent of NaN values
        * statistics via .describe()

    Args:
        dataframe ([pd.DataFrame): Source DataFrame for summary

    Returns:
        pd.DataFrame: DataFrame containing results of summary
    """

    report_df = pd.DataFrame({'null_sum':dataframe.isna().sum(),
                            'null_pct':dataframe.isna().sum()/len(dataframe), 
                            'datatypes':dataframe.dtypes,
                            'num_unique':dataframe.nunique()})

    report_df = pd.concat([report_df, dataframe.describe().T], axis=1, sort=True)

    # print(dataframe.shape)

    return report_df

def plot_boxes(data, x_label, y_label, suptitle, figsize=(13,8)):
    _, ax = plt.subplots(figsize=figsize)
    ax = sns.boxplot(data=data)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.suptitle(suptitle, size = 18)
    plt.tight_layout()

    return

def explore_feature(dataframe, column_name, show_visualization = False, target_feature= 'is_canceled',
                    plot_type = 'histogram', normalize = True,sort_val = True, 
                    width = 800, height=600, bins=None, marginal=None, plot_label = None, plot_title=None):
    """Generates a dataframe summarizing .describe(), .value_counts(), and .dtypes for a given feature from a dataframe.

    Also creates a Plotly Express histogram plot for data representation w/ option for marginal box plot on x-axis.

    ---

    DataFrame styling code adapted from:
    
    https://stackoverflow.com/questions/59769161/python-color-pandas-dataframe-based-on-multiindex#:~:text=2-,You,-can%20use%20Styler.

    Args:
        dataframe ([type]): [description]
        column_name ([type]): [description]
        normalize (bool, optional): [description]. Defaults to True.
        width (int, optional): [description]. Defaults to 800.
        height (int, optional): [description]. Defaults to 600.
        target_feature ([type], optional): [description]. Defaults to None.
        bins ([type], optional): [description]. Defaults to None.
        plot_type ([type], optional): [description]. Defaults to None.
        marginal ([type], optional): [description]. Defaults to None.
        plot_label ([type], optional): [description]. Defaults to None.
        plot_title ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """    

    print('\n|','---'*9,'Feature Details','---'*10+'-','|')

    negative = dataframe[dataframe[target_feature] == 0][column_name]
    positive = dataframe[dataframe[target_feature] == 1][column_name]

    ## Creating dataframe for .describe() results
    temp_neg = pd.DataFrame(negative.describe(datetime_is_numeric=True))
    temp_pos = pd.DataFrame(positive.describe(datetime_is_numeric=True))
    
    ## Format floats to 2 decimal points
    if (dataframe[column_name].dtype == 'int64') or (dataframe[column_name].dtype == 'float64'):
        temp_neg = round(temp_neg, 2)
        temp_pos = round(temp_pos, 2)
    else:
        pass

    ## Creating placeholder row for dataframe legibility
    temp_neg = temp_neg.T
    temp_pos = temp_pos.T
    
    temp_neg.insert(len(temp_neg.columns),' ',' ')
    temp_pos.insert(len(temp_pos.columns),' ',' ')

    temp_neg = temp_neg.T
    temp_pos = temp_pos.T
    
    ## Concat temp dataframes into one main w/ multi-index
    comb_desc = pd.DataFrame(pd.concat([temp_pos, temp_neg], keys = ['Check-Out', 'Canceled']))

    ## Creating dataframe for .value_counts() results
    if bins == None:
        vc_cxl = pd.DataFrame(positive.value_counts(dropna=False, normalize=normalize, sort=sort_val).sort_index())
        vc_co = pd.DataFrame(negative.value_counts(dropna=False, normalize=normalize, sort=sort_val).sort_index())

        if normalize == True:
            vc_cxl = vc_cxl.applymap("{0:.2%}".format)
            vc_co = vc_co.applymap("{0:.2%}".format)
        else:
            pass

        # Creating placeholder row for dataframe legibility
        vc_cxl = vc_cxl.T
        vc_co = vc_co.T

        vc_cxl.insert(len(vc_cxl.columns),' ',' ')
        vc_co.insert(len(vc_co.columns),' ',' ')

        vc_cxl = vc_cxl.T
        vc_co = vc_co.T

    else:
        vc_cxl = pd.DataFrame(positive.value_counts(dropna=False, normalize=normalize, bins=bins, sort=sort_val).sort_index()).applymap("{0:.2%}".format)   
        vc_co = pd.DataFrame(negative.value_counts(dropna=False, normalize=normalize, bins=bins, sort=sort_val).sort_index()).applymap("{0:.2%}".format)
        
    ## Concat temp dataframes into one main w/ multi-index
    value_counts = pd.DataFrame(pd.concat([vc_co,vc_cxl], keys = ['Check-Out', 'Canceled']))

    ## Creating dataframe for .dtypes results
    dt_cxl = pd.DataFrame(positive.dtypes, index=[' '], columns=[column_name])
    dt_co = pd.DataFrame(negative.dtypes, index=[' '], columns=[column_name])
    dtypes = pd.concat([dt_co,dt_cxl], keys = [' '])

    df = pd.concat([comb_desc, value_counts, dtypes], axis=0, keys=['Statistics', 'Value Counts', 'Datatype'])

    ## Setting RGBA values for blue, orange
    colors = {'Statistics': (76, 120, 168, 1), 'Value Counts': (245, 133, 24, 1), 'Datatype': (66, 153, 80, 1)}

    ## Setting full/quarter alpha levels for colors 
    c1 = {k: (r,g,b, .40) for k, (r,g,b,a) in colors.items()}
    c2 = {k: (r,g,b, .30) for k, (r,g,b,a) in colors.items()}
    c3 = {k: (r,g,b, .25) for k, (r,g,b,a) in colors.items()}

    ## Get values of first level of multi-index
    idx = df.index.get_level_values(0)

    ## Set CSS for first level
    css = [{'selector': f'.row{i}.level0', 
            'props': [('background-color', f'rgba{c1[j]}')]} for i,j in enumerate(idx)]

    ## Counter per first level for pair and unpair coloring
    zipped = zip(df.groupby(idx).cumcount()+1, enumerate(idx))

    # Setting alternating color scheme for rows
    css1 = [{'selector': f'.row{i}', 'props': [('background-color', f'rgba{c3[j]}')]} 
        if v % 2 == 0 
        else {'selector': f'.row{i}', 'props': [('background-color', f'rgba{c2[j]}')]} 
        for v,(i, j) in zipped]

    css2 = [{'selector': 'th', 'props': [('font-size', '115%'), ('text-align', 'left')],
            'selector': 'tb', 'props': [('font-size', '110%'), ('text-align', 'right')]}]

    display(df.style.set_table_styles(css + css1 + css2))
    
    if show_visualization == True:
    
        print('\n\n|','---'*9,'Visualizing Results','---'*9,'|')

        if plot_type == 'histogram':
            fig = px.histogram(dataframe,column_name,
                    color=target_feature,
                    marginal = marginal,
                    labels={column_name: plot_label}, 
                    title=plot_title,
                    width = width, height=height)
            fig.update_layout(bargap=0.2)

        if plot_type == 'box':
            fig = px.box(dataframe,column_name,
                        color=target_feature,
                        labels={column_name: plot_label},
                        title=plot_title,
                        width = width, height=height)
            fig.update_layout(bargap=0.2)
        
        fig.show()
    else:
        pass

    return None


## Function used for testing purposes
# def test_explore_feature(dataframe, column_name, normalize = True, width = 800, height=600, target_feature= None, bins=None, plot_type = None,
#                     marginal=None, plot_label = None, plot_title=None):
#     """Used for testing and development.
    
#     ---

#     DataFrame styling code adapted from:
    
#     https://stackoverflow.com/questions/59769161/python-color-pandas-dataframe-based-on-multiindex#:~:text=2-,You,-can%20use%20Styler.
#     """


#     print('\n|','---'*9,'Feature Details','---'*10+'-','|\n')    

#     negative = dataframe[dataframe[target_feature] == 0][column_name]
#     positive = dataframe[dataframe[target_feature] == 1][column_name]

#     ## Creating dataframe for .describe() results
#     comb_desc = pd.DataFrame(pd.concat([negative.describe(), positive.describe()], keys = ['Check-Out', 'Canceled']))
    
#     if dataframe[column_name].dtype != 'O':
#         comb_desc = comb_desc.applymap("{0:.2f}".format)

#     ## Creating dataframe for .value_counts() results
#     vc_cxl = positive.value_counts(dropna=False, normalize=normalize, bins=bins, sort=False).sort_index()
#     vc_co = negative.value_counts(dropna=False, normalize=normalize, bins=bins, sort=False).sort_index()
#     value_counts = pd.DataFrame(pd.concat([vc_co,vc_cxl], keys = ['Check-Out', 'Canceled'])).applymap("{0:.2f}".format)
#     # value_counts = value_counts.applymap("{0:.2f}".format)
    
#     ## Creating dataframe for .dtypes results
#     dt_cxl = pd.DataFrame(positive.dtypes, index=[' '], columns=[column_name])
#     dt_co = pd.DataFrame(negative.dtypes, index=[' '], columns=[column_name])
#     dtypes = pd.concat([dt_co,dt_cxl], keys = [' '])

#     df = pd.concat([comb_desc, value_counts, dtypes], axis=0, keys=['Statistics', 'Value Counts', 'Datatype'])

#     ## Setting RGBA values for blue, orange
#     colors = {'Statistics': (76, 120, 168, 1), 'Value Counts': (245, 133, 24, 1), 'Datatype': (66, 153, 80, 1)}

#     ## Setting full/quarter alpha levels for colors 
#     c1 = {k: (r,g,b, .35) for k, (r,g,b,a) in colors.items()}
#     c2 = {k: (r,g,b, .2) for k, (r,g,b,a) in colors.items()}
#     c3 = {k: (r,g,b, .05) for k, (r,g,b,a) in colors.items()}


#     ## Get values of first level of multi-index
#     idx = df.index.get_level_values(0)
#     # idx2 = df.index.get_level_values(1)

#     ## Set CSS for first level
#     css = [{'selector': f'.row{i}.level0', 
#             'props': [('background-color', f'rgba{c1[j]}')]} for i,j in enumerate(idx)]

#     ## Counter per first level for pair and unpair coloring
#     zipped = zip(df.groupby(idx).cumcount()+2, enumerate(idx))
#     # zipped2 = zip(df.groupby(idx2).cumcount(), enumerate(idx2))

#     # Setting alternating color scheme for rows
#     css1 = [{'selector': f'.row{i}', 'props': [('background-color', f'rgba{c1[j]}')]} 
#         if v % 2 == 0 
#         else {'selector': f'.row{i}', 'props': [('background-color', f'rgba{c2[j]}')]} 
#         for v,(i, j) in zipped]

#     css3 = [{'selector': f'.row{i}.level2', 'props': [('background-color', f'rgba{c2[j]}')]} 
#         if v % 2 == 0 
#         else {'selector': f'.row{i}.level2', 'props': [('background-color', f'rgba{c3[j]}')]} 
#         for v,(i, j) in zipped]

#     css2 = [{'selector': 'th', 'props': [('font-size', '115%'), ('text-align', 'left')],
#             'selector': 'tb', 'props': [('font-size', '110%'), ('text-align', 'right')]}]

#     display(df.style.set_table_styles(css + css1 + css2))# + css3))

#     print('\n\n|','---'*9,'Visualizing Results','---'*9,'|')

#     if plot_type == 'histogram':
#         fig = px.histogram(dataframe,column_name,
#                    color=target_feature,
#                    marginal = marginal,
#                    labels={column_name: plot_label}, 
#                    title=plot_title,
#                    width = width, height=height)
#         fig.update_layout(bargap=0.2)

#     if plot_type == 'box':
#         fig = px.box(dataframe,column_name,
#                     color=target_feature,
#                     labels={column_name: plot_label},
#                     title=plot_title,
#                     width = width, height=height)
#         fig.update_layout(bargap=0.2)
    
#     fig.show()

#     return df