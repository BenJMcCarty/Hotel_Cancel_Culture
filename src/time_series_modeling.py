'''Name: Time Series Modeling

Description: Functions created to assist with the creation and evaluation of time series models.

By Ben McCarty (bmccarty505@gmail.com)'''

### ----- Importing Dependencies ----- ###

from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import statsmodels.tsa.api as tsa

import pmdarima as pmd
from pmdarima.arima import ndiffs
from pmdarima.arima import nsdiffs

### ----------------------------------- Functions ----------------------------------- ###

## --------------- Stationary Methods --------------- ##

def adf_test(ts, p = .05):
    zipdf_results = tsa.stattools.adfuller(ts)
    
    index_label = [f'Results: {ts.name}']
    labels = ['Test Stat','P-Value','Number of Lags Used','Number of Obs. Used',
            'Critical Thresholds', 'AIC Value']
    results_dict  = dict(zip(labels,zipdf_results))

    ## Saving results to a dictionary and adding T/F indicating stationarity
    results_dict[f'p < {p}'] = results_dict['P-Value'] < p
    results_dict['Stationary'] = results_dict[f'p < {p}']

    ## Creating DataFrame from dictionary
    if isinstance(index_label,str):
        index_label = [index_label]
    results_dict = pd.DataFrame(results_dict,index=index_label)
    results_dict = results_dict[['Test Stat','P-Value','Number of Lags Used',
                                 'Number of Obs. Used','P-Value',f'p < {p}',
                                 'Stationary']]
    
    return results_dict

def remove_trends(timeseries, method, window = 4, figsize=(10,5)):
    if method == 'diff':
        results = timeseries.diff().dropna()
    elif method == 'log':
        results = np.log(timeseries)
    elif method == 'rolling' or method == 'rolling mean':
        results = timeseries - timeseries.rolling(window = window).mean()
        results.dropna(inplace=True)
    elif method == 'ewm' or method == 'EWM':
        results = timeseries-timeseries.ewm(4).mean()
        results.dropna(inplace=True)
    
    print("|","---"*7,f"{method.title()} Effect on Zipcode {timeseries.name}",
      "-----"*6,"|",'\n\n')
    print("|","---",f"Zipcode {timeseries.name}","---","|","\n")
    print(results)
    print('\n\n',"|","----"*5,f"ADF Results for Zipcode {timeseries.name}",
          "-----"*6,"|")
    display(adf_test(results))

    print('\n\n','|',"---"*8,f"Visualizing {method.title()} Effect","----"*8,
          "|")
    fig, ax = plt.subplots(figsize=figsize)
    ax = results.plot(label=f'{timeseries.name}')
    ax.legend()
    ax.set_xlabel('Years')
    ax.set_ylabel('Price ($)')
    
    if method != 'ewm' and method != 'EWM':
        ax.set_title(f'{method.title()} Effect on Zipcode {timeseries.name}')
    else:
        ax.set_title(f'{method.capitalize()} Effect on Zipcode \
                                                        {timeseries.name}')
    plt.show()
    
    return results

def plot_acf_pacf(data, lags=52, suptitle=None, figsize = (10,5)):
    """Plot pacf and acf using statsmodels
    
    Adapted from: https://github.com/flatiron-school/Online-DS-FT-022221-\
    Cohort-Notes/blob/master/Phase_4/topic_38_time_series_models/topic_38-\
    time_series_models_v3_SG.ipynb"""
    
    fig,axes=plt.subplots(nrows=2, figsize = figsize)
    
    tsa.graphics.plot_acf(data,ax=axes[0],lags=lags)
    tsa.graphics.plot_pacf(data,ax=axes[1],lags=lags)
    
    ## Add gridlines and y-labels
    [ax.grid(axis='both',which='both') for ax in axes]
    [ax.set_ylabel('Corr. Strength') for ax in axes]
    
    if suptitle is not None:
        fig.suptitle(suptitle,fontweight='bold',fontsize=15)
        
    fig.tight_layout()

    return fig,axes

### --------------- Modeling --------------- ###

# Creating train/test split for time series modeling
def train_test_split(data, split_point, xlabel, ylabel, title,
                     figsize = (10,5), show_vis = True):
    """Creates train/test split for time series modeling.

    Args:
        data (Series): Pandas Series with datetime index.
        split_point (str or float): either specific date (as string) or percentage (as float) for dividing the data
        x_label (str): Label for x-axis
        y_label (str): Label for y-axis
        title (str): Label for visualization title.
        figsize (tuple, optional): Matplotlib figure size. Defaults to (10,5).
        show_vis (bool, optional): Boolearn controlling whether to show figure. Defaults to True.

    Returns:
        dictionary: Dictionary storing train/test series and visualization of results.
    """    

    ## Generate dictionary to store results
    split_dict = {}

    ## Split the data based on datatype (numeric vs. string)
    if type(split_point) == float:
        tts_cutoff = round(data.shape[0]*split_point)
        train = data.iloc[:tts_cutoff]
        test = data.iloc[tts_cutoff:]

    elif split_point == int:
        train = data.iloc[:split_point]
        test = data.iloc[split_point:]

    else:
        train = data.iloc[:split_point]
        test = data.iloc[split_point:]

    ## Generate visualization of train/test split
    fig,ax=plt.subplots(figsize = figsize)

    ax = train.plot(label='Training Data')
    test.plot(ax=ax, label='Testing Data')
    ax.set(xlabel = xlabel, ylabel = ylabel, title = f'{title} {data.name}: Train/Test Split')
    ax.axvline(train.index[-1], linestyle=':', c='k',
               label=f'Split Date: {train.index[-1].year}/{train.index[-1].month}/{train.index[-1].day}')
    ax.legend()

    if show_vis is True:
        plt.show(fig)

    plt.close(fig)

    split_dict['train'] = train
    split_dict['test'] = test
    split_dict['split_fig'] = fig
    
    return split_dict


## Generate best model parameters via auto_arima
def auto_arima_model(timeseries_dataset, m = 12, start_p=0,max_p=5,
                        start_q=0,max_q=5,start_P=0,
                        start_Q=0, max_P=5, max_Q = 5):
    
    """Fits an auto_arima model to a given timeseries dataset.

    Args:
        timeseries_dataset (Series/DataFrame): dataset for modeling
        m (int): The number of periods in each season (for seasonal differencing).
        start_p (int, optional): Starting value for "p". Defaults to 0.
        max_p (int, optional): Max value for "p". Defaults to 3.
        start_q (int, optional): Starting value for "pq. Defaults to 0.
        max_q (int, optional): Max value for "q". Defaults to 3.
        start_P (int, optional): Starting value for "P". Defaults to 0.
        start_Q (int, optional): Starting value for "Q". Defaults to 0.
        max_P (int, optional): Max value for "P". Defaults to 3.
        max_Q (int, optional): Max value for "Q". Defaults to 3.

    Returns:
        auto_arima_model: Fitted auto_arima model for use in SARIMAX modeling.
    """    

    ## Determine d, D values for SARIMA model
    n_d = ndiffs(timeseries_dataset)
    n_D = nsdiffs(timeseries_dataset, m=m)

    auto_arima_model = pmd.auto_arima(timeseries_dataset,m = m,
                                start_p = start_p,max_p = max_p,
                                start_q = start_q, max_q = max_q,
                                start_P = start_P, max_P = max_P,
                                start_Q = start_Q, max_Q = max_Q,
                                d = n_d, D = n_D, error_action="ignore")

    return auto_arima_model

## Create new SARIMA model via Statsmodels with pre-created auto_ARIMA model
def create_best_model(timeseries_dataset, m, 
                        start_p=0, max_p=5, start_q=0, max_q=5,
                        start_P=0, start_Q=0, max_P=5, max_Q=5,
                        show_vis=True, figsize=(10,5)):

    """Calculates best model parameters via auto-arima,
     then fits a new SARIMAX model for results.

    Args:
        timeseries_dataset (Series/DataFrame): dataset for modeling
        m (int): The number of periods in each season (for seasonal differencing)
        start_p (int, optional): Starting value for "p". Defaults to 0.
        max_p (int, optional): Max value for "p". Defaults to 3.
        start_q (int, optional): Starting value for "pq. Defaults to 0.
        max_q (int, optional): Max value for "q". Defaults to 3.
        start_P (int, optional): Starting value for "P". Defaults to 0.
        start_Q (int, optional): Starting value for "Q". Defaults to 0.
        max_P (int, optional): Max value for "P". Defaults to 3.
        max_Q (int, optional): Max value for "Q". Defaults to 3.
        show_vis (boolean, optional): Whether to show the model summary and plot diagnostics. Defaults to False.

    Returns:
        auto_model, best_model: auto_arima-generated model with best parameters,
                                SARIMAX model using best parameters.
    """

    ## Determine d, D values for SARIMA model
    n_d = ndiffs(timeseries_dataset)
    n_D = nsdiffs(timeseries_dataset, m=m)

    auto_model_best = pmd.auto_arima(timeseries_dataset,m = m,
                                start_p = start_p,max_p = max_p,
                                start_q = start_q, max_q = max_q,
                                start_P = start_P, max_P = max_P,
                                start_Q = start_Q, max_Q = max_Q,
                                d = n_d, D = n_D, error_action="ignore")
      
    best_model = tsa.SARIMAX(timeseries_dataset,order=auto_model_best.order,
                             seasonal_order = auto_model_best.seasonal_order,
                             enforce_invertibility=False).fit()
        
    if show_vis is True:
        display(auto_model_best.summary())
        display(model_performance(best_model, show_vis=True, figsize=figsize))
        plt.show()
    
    plt.close()

    return auto_model_best, best_model


## Display model results
def model_performance(ts_model, show_vis = False, figsize = (12, 6)):
    """Displays a fitted model's summary and plot diagnostics.

    Args:
        ts_model (model): fitted model for evaluation
    """    
    perf = {}

    perf['summary'] = ts_model.summary()

    perf['vis'] = ts_model.plot_diagnostics(figsize = figsize)

    if show_vis == True:
        plt.show(perf['vis'])

    plt.close(perf['vis'])

    return perf


## Using get_forecast to generate forecasted data
def forecast_and_ci(model, test_data, alpha=0.05):
    """Generates forecasted data and the upper/lower confidence intervals for forecast.

    Args:
        model (Statsmodels SARIMA model): fitted SARIMA model
        test_data (Series): Test data used for evaluation

    Returns:
        DataFrame: Forecast results and upper/lower confidence intervals.
    """

    forecast = model.get_forecast(steps=len(test_data))
    forecast_df = forecast.conf_int(alpha=alpha)
    forecast_df.columns = ['Lower CI','Upper CI']
    forecast_df['Forecast'] = forecast.predicted_mean

    return forecast_df

## Plotting training, testing datasets
def plot_forecast_ttf(split_dict, forecast_df, xlabel, ylabel, title, figsize = (10,5), show_vis = True):
    """Visualizes the results of a train/test split with the training forecast overlaid
     to compare against the test set.

    Args:
        split_dict (dictionary): Dictionary generated by train/test split function
        forecast_df (DataFrame): DataFrame of forecasted results.
        xlabel (str): Label for x-axis
        ylabel (str): Label for y-axis
        title (str): Label for title.
        figsize (tuple, optional): Matplotlib figure size. Defaults to (10,5).
        show_vis (bool, optional): Boolean control to show visualizations. Defaults to True.

    Returns:
        [type]: [description]
    """    

    train = split_dict.get('train')
    test = split_dict.get('test')

    last_n_lags=len(train)
    
    fig,ax=plt.subplots(figsize = figsize)

    train.iloc[-last_n_lags:].plot(label='Training Data', ax=ax)
    test.plot(label='Test Data', ax=ax)

    ## Plotting forecasted data and confidence intervals
    forecast_df['Forecast'].plot(label='Forecast', color='g', ax=ax)
    ax.fill_between(forecast_df.index,forecast_df['Lower CI'],
                    forecast_df['Upper CI'],color='y',alpha=0.275)
    ax.set(xlabel = xlabel, ylabel = ylabel, title = f'{title} {train.name}: Validating Forecast')
    ax.axvline(test.index[0], linestyle=":",
     label=f'Beginning of Forecast: {test.index[0].year} - {test.index[0].month}',
      color='k')
    ax.legend(loc='upper left')
    
    ttf_dict = {}
    ttf_dict['vis'] = fig

    if show_vis == True:
        plt.show()
    
    plt.close()

    return ttf_dict


## Plotting training, testing datasets
def plot_forecast_final(data, forecast_df, xlabel, ylabel, title, figsize = (10,5), show_vis = True):
    """Visualizes the forecasted values from a time series model alongside the original data

    Args:
        data (Series): Pandas Series with a datetime index for forecasting
        forecast_df (DataFrame): Forecast results.
        figsize (tuple, optional): Matplotlib figure size. Defaults to (10,5).
        show_vis (bool, optional): Boolean control to show visualization. Defaults to True.

    Returns:
        figure : Matplotlib figure
    """    
    
    ## Plotting original data and forecasted results
    fig,ax=plt.subplots(figsize = figsize)

    ## Plotting original data
    data.plot(ax=ax, label='Original Data')

    ## Plotting forecasted data and confidence intervals
    forecast_df['Forecast'].plot(ax=ax,label='Forecast', color='g')
    ax.fill_between(forecast_df.index,forecast_df['Lower CI'],
                    forecast_df['Upper CI'],color='y',alpha=0.275)
    ax.set(xlabel={xlabel}, ylabel={ylabel}, title = f'{title} {data.name}: Forecast Results')
    ax.axvline(data.index[-1], linestyle=":",
     label=f'Beginning of Forecast: {data.index[-1].year}'+'-'+f'{data.index[-1].month}',
      color='k')
    ax.legend(loc='upper left')

    if show_vis == True:
        plt.show()

    plt.close()

    return fig


### --------------- Workflow --------------- ###

def ts_modeling_workflow(data, threshold, m, xlabel, ylabel, title, figsize = (10,5), alpha=0.05, show_vis = True):

    tsa_results = {}
    metrics = {}
    forecast_vis = {}

    ## Split dataset
    split_dict = train_test_split(data, threshold, xlabel = xlabel, ylabel = ylabel, title = title,
                                    show_vis = show_vis, figsize=figsize)

    train = split_dict.get('train')
    test = split_dict.get('test')
    split_vis = split_dict.get('split_vis')
    
    if show_vis == True:
        plt.show()
    plt.close()

    ## Generating auto_arima model and SARIMAX model
    auto_model_train, best_model_train = create_best_model(timeseries_dataset = train, m=m, show_vis=show_vis)
    
    ## Saving training model results
    metrics['train'] = model_performance(best_model_train)

    vis = metrics.get('train').get('vis')

    if show_vis == True:
        plt.show()
    plt.close()

    ## Generating dataframe to store forecast results
    forecast_train = forecast_and_ci(best_model_train, test, alpha=alpha)

    ## Plotting forecast results against train/test split
    forecast_vis['train'] = plot_forecast_ttf(split_dict, forecast_df = forecast_train,
                                                xlabel = xlabel, ylabel = ylabel, title = title,
                                                figsize=figsize, show_vis = show_vis)

    vis = forecast_vis.get('train').get('vis')

    if show_vis == True:
        plt.show()
    plt.close()

    ## Fitting best model using whole dataset
    best_model_full = tsa.SARIMAX(data,order=auto_model_train.order,
                            seasonal_order = auto_model_train.seasonal_order,
                            enforce_invertibility=False).fit()

    metrics['full'] = model_performance(best_model_full, show_vis = show_vis)
    
    vis = metrics.get('vis')

    if show_vis == True:
        plt.show(vis)
    plt.close(vis)

    ## Generating forecast
    best_forecast = forecast_and_ci(best_model_full, test, alpha=alpha)

    tsa_results['forecasted_data'] = best_forecast

    ## Plotting original data and forecast results
    forecast_vis['full'] = plot_forecast_final(data, tsa_results['forecasted_data'],
                                                xlabel = xlabel, ylabel = ylabel, title = title,
                                                figsize=figsize, show_vis= show_vis)
    
    plt.show(forecast_vis['full'])

    # ## Calculating investment cost and ROI across dataframe
    # investment_cost = tsa_results['forecasted_prices'].iloc[0,2]
    # tsa_results['roi'] = (tsa_results['forecasted_prices'] - investment_cost)/investment_cost*100
    
    tsa_results['model'] = best_model_full
    tsa_results['forecast_length'] = len(split_dict['test'])
    tsa_results['model_metrics'] = metrics
    tsa_results['model_visuals'] = forecast_vis
    
    plt.close()

    return tsa_results