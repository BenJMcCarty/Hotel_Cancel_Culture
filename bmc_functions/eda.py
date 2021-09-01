'''Name: EDA

Description: Functions created to assist with data cleaning and exploratory data analysis (EDA).

By Ben McCarty (bmccarty505@gmail.com)'''

import numpy as np
import pandas as pd
import statsmodels as stats

import matplotlib.pyplot as plt
import matplotlib.colors as col
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
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.boxplot(data=data)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.suptitle(suptitle, size = 18)
    plt.tight_layout()

    return

def explore_feature(dataframe, column_name, normalize = True, width = 800, height=600, bins=None, plot_type = None,
                    marginal_x=None, marginal_y = None,target_feature = None,
                    plot_label = None, plot_title=None):

    """Generates a dataframe containing details about the selected feature. 
    Offers options to produce a Plotly Express plot: histogram, box, scatter (w/ regression line),
    and options to include marginal plots

    Args:
        dataframe (DataFrame): Data source

        column_name (str): Column/feature name

        width (int, optional): Plot width. Defaults to 600.

        height (int, optional): Plot height. Defaults to 400.
        
        plot_type (str, optional): One of "histogram," "box," or "scatter." Defaults to None.

        marginal_x (string, one of 'rug', 'box', 'violin', 'histogram'): Generate a Plotly Express plot on the marginal x axis. Defaults to None.
        
        marginal_y (string, one of 'rug', 'box', 'violin', 'histogram'): Generate a Plotly Express plot on the marginal y axis. Defaults to None.
        
        target_feature (str, optional): Target feature to use as "color" parameter. Defaults to None.
        
        plot_label (str, optional): Plot x-label; dictionary of column name and desired label. Defaults to None.
        
        plot_title (str, optional): Plot title. Defaults to None.

    Source: https://stackoverflow.com/questions/59769161/python-color-pandas-dataframe-based-on-multiindex
      
    """

    print('\n|','---'*9,'Feature Details','---'*10+'-','|\n')    

    stats = dataframe[column_name].describe()
    stats = pd.DataFrame(stats)
    stats['report'] = 'Stats'

    value_counts = dataframe[column_name].value_counts(dropna=False, normalize=normalize, bins=bins)
    value_counts = pd.DataFrame(value_counts)
    value_counts['report'] = 'Counts'

    df = pd.concat([stats, value_counts], axis=0, keys=['Statistics', 'Value_Counts']).drop(columns='report')

    ## Setting RGBA values for blue, orange
    colors = {'Statistics': (76, 120, 168, 1), 'Value_Counts': (245, 133, 24, 1)}

    ## Setting full/quarter alpha levels for colors 
    c1 = {k: (r,g,b, .35) for k, (r,g,b,a) in colors.items()}
    c2 = {k: (r,g,b, .2) for k, (r,g,b,a) in colors.items()}

    ## Get values of first level of multi-index
    idx = df.index.get_level_values(0)

    ## Set CSS for first level
    css = [{'selector': f'.row{i}.level0', 
            'props': [('background-color', f'rgba{c1[j]}')]} for i,j in enumerate(idx)]

    ## Counter per first level for pair and unpair coloring
    zipped = zip(df.groupby(idx).cumcount(), enumerate(idx))

    # 
    css1 = [{'selector': f'.row{i}', 'props': [('background-color', f'rgba{c1[j]}')]} 
        if v % 2 == 0 
        else {'selector': f'.row{i}', 'props': [('background-color', f'rgba{c2[j]}')]} 
        for v,(i, j) in zipped]

    css2 = [{'selector': 'th', 'props': [('font-size', '115%')]}]

    display(df.style.set_properties(**{'font-size': "115%"}).set_table_styles(css1 + css + css2))

    print('\n\n|','---'*9,'Visualizing Results','---'*9,'|')

    if plot_type is 'histogram':
        fig = px.histogram(dataframe,column_name,
                   color=target_feature,
                   marginal = marginal_x,
                   labels={column_name: plot_label}, 
                   title=plot_title,
                   width = width, height=height)
        fig.update_layout(bargap=0.2)
        fig.show()

    if plot_type is 'box':
        fig = px.box(dataframe,column_name,
                    color=target_feature,
                    labels={column_name: plot_label},
                    title=plot_title,
                    width = width, height=height)
        fig.update_layout(bargap=0.2)
        fig.show()

    return df


def explore_feature_test(dataframe, column_name, normalize = True, width = 800, height=600, bins=None, plot_type = None,
                    marginal_x=None, marginal_y = None,target_feature = None,
                    plot_label = None, plot_title=None):

    """Generates a dataframe containing details about the selected feature. 
    Offers options to produce a Plotly Express plot: histogram, box, scatter (w/ regression line),
    and options to include marginal plots

    Args:
        dataframe (DataFrame): Data source

        column_name (str): Column/feature name

        width (int, optional): Plot width. Defaults to 600.

        height (int, optional): Plot height. Defaults to 400.
        
        plot_type (str, optional): One of "histogram," "box," or "scatter." Defaults to None.

        marginal_x (string, one of 'rug', 'box', 'violin', 'histogram'): Generate a Plotly Express plot on the marginal x axis. Defaults to None.
        
        marginal_y (string, one of 'rug', 'box', 'violin', 'histogram'): Generate a Plotly Express plot on the marginal y axis. Defaults to None.
        
        target_feature (str, optional): Target feature to use as "color" parameter. Defaults to None.
        
        plot_label (str, optional): Plot x-label; dictionary of column name and desired label. Defaults to None.
        
        plot_title (str, optional): Plot title. Defaults to None.

    Source: https://stackoverflow.com/questions/59769161/python-color-pandas-dataframe-based-on-multiindex
      
    """

    print('\n|','---'*9,'Feature Details','---'*10+'-','|\n')    

    stats = dataframe[column_name].describe()
    stats = pd.DataFrame(stats)
    stats['report'] = 'Stats'

    value_counts = dataframe[column_name].value_counts(dropna=False, normalize=normalize, bins=bins)
    value_counts = pd.DataFrame(value_counts)
    value_counts['report'] = 'Counts'

    df = pd.concat([stats, value_counts], axis=0, keys=['Statistics', 'Value_Counts']).drop(columns='report')

    ## Setting RGBA values for blue, orange
    colors = {'Statistics': (76, 120, 168, 1), 'Value_Counts': (245, 133, 24, 1)}

    ## Setting full/quarter alpha levels for colors 
    c1 = {k: (r,g,b, .275) for k, (r,g,b,a) in colors.items()}
    c2 = {k: (r,g,b, .2) for k, (r,g,b,a) in colors.items()}

    ## Get values of first level of multi-index
    idx = df.index.get_level_values(0)

    ## Set CSS for first level
    css = [{'selector': f'.row{i}.level0', 
            'props': [('background-color', f'rgba{c1[j]}')]} for i,j in enumerate(idx)]

    ## Counter per first level for pair and unpair coloring
    zipped = zip(df.groupby(idx).cumcount(), enumerate(idx))

    # 
    css1 = [{'selector': f'.row{i}', 'props': [('background-color', f'rgba{c1[j]}')]} 
        if v % 2 == 0 
        else {'selector': f'.row{i}', 'props': [('background-color', f'rgba{c2[j]}')]} 
        for v,(i, j) in zipped]

    css2 = [{'selector': 'th', 'props': [('font-size', '115%')]}]

    display(df.style.set_properties(**{'font-size': "110%"}).set_table_styles(css1 + css + css2))

    print('\n\n|','---'*9,'Visualizing Results','---'*9,'|')

    if plot_type is 'histogram':
        fig = px.histogram(dataframe,column_name,
                   color=target_feature,
                   marginal = marginal_x,
                   labels={column_name: plot_label}, 
                   title=plot_title,
                   width = width, height=height)
        fig.update_layout(bargap=0.2)
        fig.show()

    if plot_type is 'box':
        fig = px.box(dataframe,column_name,
                    color=target_feature,
                    labels={column_name: plot_label},
                    title=plot_title,
                    width = width, height=height)
        fig.update_layout(bargap=0.2)
        fig.show()

    return df