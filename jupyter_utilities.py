# ## Useful utilities for merging data and graphing
import pandas as pd
import numpy as np
import os, sys
import datetime
# import pandas_datareader.data as pdr
import yfinance as yf

import matplotlib.pyplot as plt
# import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot
import plotly.tools as tls

import zipfile
import urllib.request
from PIL import Image
import pandasql as psql
import importlib
import time
import re

def str_to_yyyymmdd(d,sep='-'):
    try:
        dt = datetime.datetime.strptime(str(d)[:10],f'%Y{sep}%m{sep}%d')
    except:
        return None
    s = '%04d%02d%02d' %(dt.year,dt.month,dt.day)
    return int(s)

def str_to_date(d,sep='-'):
    try:
        dt = datetime.datetime.strptime(str(d)[:10],f'%Y{sep}%m{sep}%d')
    except:
        return None
    return dt


def fetch_history(symbol,dt_beg,dt_end):
#     df = pdr.DataReader(symbol, 'yahoo', dt_beg, dt_end)
    df = yf.download(symbol,dt_beg,dt_end)
    # move index to date column, sort and recreate index
    df['date'] = df.index
    df = df.sort_values('date')
    df.index = list(range(len(df)))
    # make adj close the close
    df['Close'] = df['Adj Close']
    df = df.drop(['Adj Close'],axis=1)
    cols = df.columns.values 
    cols_dict = {c:c[0].lower() + c[1:] for c in cols}
    df = df.rename(columns = cols_dict)
    df['trade_date'] = df.date.apply(str_to_yyyymmdd)
    return df
    

def psql_merge(df1_name,col1_low,col1_high,df2_name,col2_low,col2_high):
    q = f"""select * from {df1_name}
    join {df2_name} on {df1_name}.{col1_low} >= {df2_name}.{col2_low} 
         and {df1_name}.{col1_high} < {df2_name}.{col2_high} 
    """
    pysqldf = lambda q: psql.sqldf(q, globals())
    df3 = pysqldf(q)
    return df3



def plot_pandas(df_in,x_column,num_of_x_ticks=20,bar_plot=False,figsize=(16,10),alt_yaxis=True):
    '''
    '''
    df_cl = df_in.copy()
    df_cl.index = list(range(len(df_cl)))
    df_cl = df_cl.drop_duplicates()
    xs = list(df_cl[x_column])
    df_cl[x_column] = df_cl[x_column].apply(lambda i:str(i))

    x = list(range(len(df_cl)))
    n = len(x)
    s = num_of_x_ticks
    x_indices = x[::n//s][::-1]
    x_labels = [str(t) for t in list(df_cl.iloc[x_indices][x_column])]
    ycols = list(filter(lambda c: c!=x_column,df_cl.columns.values))
    all_cols = [x_column] + ycols
    if bar_plot:
        if len(ycols)>1 and alt_yaxis:
            ax = df_cl[ycols].plot.bar(secondary_y=ycols[1:],figsize=figsize)
        else:
            ax = df_cl[ycols].plot.bar(figsize=figsize)
    else:
        if len(ycols)>1 and alt_yaxis:
            ax = df_cl[ycols].plot(secondary_y=ycols[1:],figsize=figsize)
        else:
            ax = df_cl[ycols].plot(figsize=figsize)

    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels, rotation='vertical')
    ax.grid()
    ax.figure.set_size_inches(figsize)
    return ax



def multi_plot(df,x_column,save_file_prefix=None,save_image_folder=None,dates_per_plot=100,num_of_x_ticks=20,figsize=(16,10),bar_plot=False):
    plots = int(len(df)/dates_per_plot) + 1 if len(df) % dates_per_plot > 0 else 0
    f = plt.figure()
    image_names = []
    all_axes = []
    for p in range(plots):
        low_row = p * dates_per_plot
        high_row = low_row + dates_per_plot
        df_sub = df.iloc[low_row:high_row]
        ax = plot_pandas(df_sub,x_column,num_of_x_ticks=num_of_x_ticks,figsize=figsize,bar_plot= bar_plot)
        all_axes.append(ax)
        if save_file_prefix is None or save_image_folder is None:
            continue
        image_name = f'{save_image_folder}/{save_file_prefix}_{p+1}.png'
        ax.get_figure().savefig(image_name)
        image_names.append(image_name)
    return (all_axes,image_names)

def multi_df_plot(dict_df,x_column,save_file_prefix=None,save_image_folder=None,num_of_x_ticks=20,figsize=(16,10),bar_plot=False):
    f = plt.figure()
    image_names = []
    all_axes = []
    p = 0
    for k in dict_df.keys():
        df = dict_df[k]
        ax = plot_pandas(df,x_column,num_of_x_ticks=num_of_x_ticks,figsize=figsize,bar_plot= bar_plot)
        ax.set_title(k)
        all_axes.append(ax)
        if save_file_prefix is None or save_image_folder is None:
            continue
        image_name = f'{save_image_folder}/{save_file_prefix}_{k}{p+1}.png'
        ax.get_figure().savefig(image_name)
        image_names.append(image_name)
        p += 1
    return (all_axes,image_names)



# create random open interest and price data
def create_df_random():
    random_open_interest = 100000 * (np.random.randn(1000)+1)
    random_prices = 20 * (np.random.randn(1000)+1)

    # create a bunch of weekday dates
    dt_beg = datetime.datetime(2014,1,1)
    dt_end = datetime.datetime(2019,1,1)
    dates_no_weekends = pd.bdate_range(dt_beg, dt_end)[-1000:]

    # get all of the fridays
    all_fridays = np.array(list(filter(lambda d: d.weekday()==4,dates_no_weekends)))
    current_friday_index = 0
    fridays = []

    # for each date in dates_no_weekends, get the friday date of that week
    for d in dates_no_weekends:
        if current_friday_index < len(all_fridays) and (all_fridays[current_friday_index] - d).days <0:
            current_friday_index += 1
        cf = all_fridays[current_friday_index] if current_friday_index < len(all_fridays) else             all_fridays[-1] + datetime.timedelta(7)
        fridays.append(cf)
    yyyymmdd_dates = [int(str(d)[0:4]+str(d)[5:7]+str(d)[8:10]) for d in dates_no_weekends]
    yyyymmdd_fridays = [int(str(d)[0:4]+str(d)[5:7]+str(d)[8:10]) for d in fridays]

    # create a DataFrame with the random data, the dates, and the fridays
    df_random = pd.DataFrame({'nav':random_prices,'open_interest':random_open_interest,'trade_date':yyyymmdd_dates,'friday_date':yyyymmdd_fridays,'friday':fridays})

    # create another DataFrame, that has weekly average prices for the data in df_random
    df_friday_avg = df_random[['friday','nav']].groupby('friday',as_index=False).mean()
    df_friday_avg = df_friday_avg.rename(columns = {'nav':'friday_nav'})
    this_fridays = [int(str(d)[0:4]+str(d)[5:7]+str(d)[8:10]) for d in list(df_friday_avg.friday)]
    next_fridays = list(df_friday_avg[1:].friday) + [df_friday_avg.iloc[0-1].friday + datetime.timedelta(7)]
    next_yyyymmdd_fridays = [int(str(d)[0:4]+str(d)[5:7]+str(d)[8:10]) for d in next_fridays]
    df_friday_avg['this_friday_date'] = this_fridays
    df_friday_avg['next_friday_date'] = next_yyyymmdd_fridays
    df_friday_avg = df_friday_avg[['this_friday_date','next_friday_date','friday_nav']]
    

    # ### Merge the 2 random DataFrames using ```psql_merge```
    df_merged = psql_merge('df_random','trade_date','trade_date',
                          'df_friday_avg','this_friday_date','next_friday_date')
    return df_merged


def plot_pandas_example():
    df_merged = create_df_random()
    plot_pandas(df_in=df_merged[['trade_date','nav','open_interest']][-200:],x_column='trade_date');


def multi_plot_example(saved_image_folder):
    image_names = multi_plot(df_merged[['trade_date','nav','open_interest']],'trade_date','random_nav')


    imgs    = [ Image.open(i) for i in image_names ]
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )

    # for a vertical stacking it is simple: use vstack
    # imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
    imgs_comb = np.vstack( (np.asarray(i) for i in imgs ) )
    imgs_comb = Image.fromarray( imgs_comb)
    save_file = f"{saved_image_folder}/random_nav_all.png"
    imgs_comb.save( save_file)

    
# Now, generate to small plots inside a larger figure, using plt.figure
def multi_subplots(y):
    '''
        y is a 2 dimensional a "matrix-like" np.array, where the dimension 0 is the number of rows,
          and dimension 1 is the number of columns
        For each column, create a subplot.
    '''
    plt.figure(figsize=(8,3*y.shape[1]))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    # on each iteration of this loop, we create another subplot, BUT NOT ON THE SAME PLO.
    # Previously, we created 2 plots, with different y axis's, on the same plot.
    # Now, we generate 2 plots (subplots).
    for i in range(y.shape[1]):
        plt.subplot(y.shape[1],1,1+i)
        if i<=0:
            plt.title('A Simple Plot')
        plt.plot(y[:, i],colors[i], lw=1.5, label=f'{i+1}st')
        plt.plot(y[:, i],'ro')
        plt.legend(loc=0)
        plt.grid(True)
        plt.axis('tight')
        plt.xlabel('index')
        plt.ylabel(f'value {i+1}')
    
    
def reload_module(module_name):
    importlib.reload(module_name)


nymex_sd_dict = {s:'NYMEX' for s in ['CL','NG','HO','RB','PL','PA']}
mcs_lists = {c:'FGHJKMNQUVXZ' for c in nymex_sd_dict.keys()}
secdef_dict = nymex_sd_dict.copy()

ecbot_hknuz_sd_dict = {s:'ECBOT' for s in ['ZW','ZC','ZS']}
mcs_lists.update({c:'HKNUZ' for c in ecbot_hknuz_sd_dict.keys()})
secdef_dict.update(ecbot_hknuz_sd_dict)

ecbot_hmuz_sd_dict = {s:'ECBOT' for s in ['ZN','ZB']}
mcs_lists.update({c:'HMUZ' for c in ecbot_hmuz_sd_dict.keys()})
secdef_dict.update(ecbot_hmuz_sd_dict)

ecbot_fhknqux_sd_dict = {s:'ECBOT' for s in ['ZS']}
mcs_lists.update({c:'FHKNQUX' for c in ecbot_fhknqux_sd_dict.keys()})
secdef_dict.update(ecbot_fhknqux_sd_dict)

cme_hmuz_sd_dict = {s:'GLOBEX' for s in ['ES','GE','6E','6B','EUR','GBP','SF']}
mcs_lists.update({c:'HMUZ' for c in cme_hmuz_sd_dict.keys()})
secdef_dict.update(cme_hmuz_sd_dict)

comex_gjmqvz_sd_dict = {s:'NYMEX' for s in ['GC']}
mcs_lists.update({c:'GJMQVZ' for c in comex_gjmqvz_sd_dict.keys()})
secdef_dict.update(comex_gjmqvz_sd_dict)


month_codes = ','.join('FGHJKMNQUVXZ').split(',')
monthcode_to_monthnum = {month_codes[i]:i for i in range(len(month_codes))}

symbol_exceptions = {'GLD':'GLD.STK.ARCA'}
    
def get_spot_code(commod):
    global mcs_lists,secdef_dict
    mcs = mcs_lists[commod]
    n = datetime.datetime.now()
    endy = n.year - 2000
    begy = endy - 2
    m = n.month 
    code_this_month = month_codes[m-1]
    spot_code = None
    for c in mcs:
        if c>=code_this_month:
            spot_code=c
            break
    if spot_code is None:
        spot_code = mcs[0]
        endy +=1
    return spot_code

def get_series(commod,SAVE_CSV_FOLDER):
    global mcs_lists,secdef_dict
    mcs = mcs_lists[commod]
    n = datetime.datetime.now()
    endy = n.year - 2000
    begy = endy - 2
    spot_code = get_spot_code(commod)
    last_contract = spot_code + '%02d' %(endy)
    print(last_contract)
    years = np.linspace(begy,endy,endy-begy+1,dtype=int)
    print('years',years)
    for y in years:
        for mc in mcs:
            symbol = '%s%s%02d' %(commod,mc,y)
            save_file = f'{SAVE_CSV_FOLDER}/{symbol}.csv'
            if os.path.isfile(save_file):
                continue
            print(f'fetch symbol {symbol}')
            try:
                df = fetch_ib_history(symbol)
                df.to_csv(save_file)
                yym = symbol[-2:] + symbol[-3:-2]
                lyym = last_contract[-2:] + last_contract[-3:-2]
                print('yym',yym,'lyym',lyym)
                if yym >= lyym:
                    break
            except Exception as e:
                print('get_series EXCEPTION: ',str(e))
                continue
                      


def make_nav_csv(df,symbol):
    df['trade_date'] = df.date.apply(lambda d: int(str(d)[0:4] + str(d)[5:7] + str(d)[8:10]))
    df.date = df.date.apply(lambda d: str(d)[0:4] + "-" + str(d)[4:6] + "-" + str(d)[6:8])
    df = df.sort_values('date')
    df.index = list(range(len(df)))
    # make adj close the close
    df['nav'] = df['adjusted']
    df['symbol'] = symbol
    df['shares'] = 0
    df = df[['symbol','date','nav','shares','trade_date']]
    return df

def make_ib_full_symbol(symbol_or_contract,currency='USD'):
    if symbol_or_contract in symbol_exceptions.keys():
        return symbol_exceptions[symbol_or_contract]
    isfuture = len(re.findall(string=symbol_or_contract.strip(),pattern='^.{1,4}[FGHJKMNQUVXZ][0-4][0-9]$'))>0
    if isfuture:
        if len(symbol_or_contract)>3:
            commod = symbol_or_contract[:-3]
        else:
            commod = symbol_or_contract
        exch = secdef_dict[commod]
        y = str(2000 + int(symbol_or_contract[-2:]))
        m = '%02d' %(monthcode_to_monthnum[symbol_or_contract[-3:-2]] + 1)
        contract = ".".join([commod,'FUT',exch,currency,y])+m
    else:
        iscontract = len(symbol_or_contract.split('.'))>=3
        if iscontract:
            contract = symbol_or_contract
        else:
            contract = symbol_or_contract + '.STK.SMART'
    return contract

def fetch_ib_history(symbol_or_contract,return_raw=False,days_back=60,time_period=86400000,currency='USD'):
    contract = make_ib_full_symbol(symbol_or_contract,currency=currency)
    url = f"http://127.0.0.1:8899/ibhistory?{time_period}%20{days_back}%200%205%20{contract}"
    df = pd.read_csv(url)
    if not return_raw:
        df = make_nav_csv(df,symbol_or_contract)
    return df
    
    
    
def symbol_merge(symbol_list,value_column='close',date_column='date',fetcher=None,days=120):
    def _default_fetcher(days_to_fetch=None):
        dtf = days if days_to_fetch is None else days_to_fetch
        def _fetch_inner(sym):
            return fetch_ib_history(symbol_or_contract=sym,return_raw=True,days_back=dtf)
        return _fetch_inner
    ff = fetcher
    if ff is None:
        ff = _default_fetcher(days)
    df_final = None
    df_merged = None
    for sym in symbol_list:
        try:
            dft = ff(sym)
            dft['symbol'] = sym
            if df_final is None:
                df_final = dft.copy()
            else:
                df_final = df_final.append(dft,ignore_index=True)
            df_final.index = list(range(len(df_final)))
            dft2 = dft[[date_column,value_column]]
            dft2 = dft2.rename(columns={value_column:sym})
            if df_merged is None:
                df_merged = dft2.copy()
            else:
                df_merged = df_merged.merge(dft2,how='inner',on=date_column)
        except Exception as e:
            print(str(e))
            continue
    dfm2 = df_merged.copy()
    dfm2 = dfm2.drop(columns=[date_column])
    df_corr = dfm2.corr()
    return {'raw':df_final,'merged':df_merged,'corr':df_corr}

def plotly_pandas(df_in,x_column,num_of_x_ticks=20,plot_title=None,
                  y_left_label=None,y_right_label=None,bar_plot=False,figsize=(16,10),alt_yaxis=True):    
#     f = plot_pandas(df_in,x_column=x_column,bar_plot=False)#.get_figure()
    f = plot_pandas(df_in,x_column=x_column,bar_plot=bar_plot,
                    num_of_x_ticks=num_of_x_ticks,alt_yaxis=alt_yaxis)#.get_figure()
    # list(filter(lambda s: 'get_y' in s,dir(f)))
    plotly_fig = tls.mpl_to_plotly(f.get_figure())
    d1 = plotly_fig['data'][0]
    number_of_ticks_display=20
    td = list(df_in[x_column]) 
    spacing = len(td)//number_of_ticks_display
    tdvals = td[::spacing]
    d1.x = td
    d_array = [d1]
    if len(plotly_fig['data'])>1:
        d2 = plotly_fig['data'][1]
        d2.x = td
        d2.xaxis = 'x'
        d_array.append(d2)

    layout = go.Layout(
        title='plotly' if plot_title is None else plot_title,
        xaxis=dict(
            ticktext=tdvals,
            tickvals=tdvals,
            tickangle=90,
            type='category'),
        yaxis=dict(
            title='y main' if y_left_label is None else y_left_label
        ),
    )
    if len(d_array)>1:
        layout = go.Layout(
            title='plotly' if plot_title is None else plot_title,
            xaxis=dict(
                ticktext=tdvals,
                tickvals=tdvals,
                tickangle=90,
                type='category'),
            xaxis2=dict(
                ticktext=tdvals,
                tickvals=tdvals,
                tickangle=90,
                type='category'),
            yaxis=dict(
                title='y main' if y_left_label is None else y_left_label
            ),
            yaxis2=dict(
                title='y alt' if y_right_label is None else y_right_label,
                overlaying='y',
                side='right')
        )

    fig = go.Figure(data=d_array,layout=layout)
    if bar_plot:  # fix y values, which have all become positive
        df_yvals = df_in[[c for c in df_in.columns.values if c != x_column]]
        for i in range(len(df_yvals.columns.values)):
            fig.data[i].y = df_yvals[df_yvals.columns.values[i]]
    return fig

def plotly_plot(df_in,x_column,plot_title=None,
                y_left_label=None,y_right_label=None,
                bar_plot=False,figsize=None,#figsize=(16,10),
                number_of_ticks_display=20,
                yaxis2_cols=None):
    ya2c = [] if yaxis2_cols is None else yaxis2_cols
    ycols = [c for c in df_in.columns.values if c != x_column]
    # create tdvals, which will have x axis labels
    td = list(df_in[x_column]) 
    spacing = len(td)//number_of_ticks_display
    tdvals = td[::spacing]
    
    # create data for graph
    data = []
    # iterate through all ycols to append to data that gets passed to go.Figure
    for ycol in ycols:
        if bar_plot:
            b = go.Bar(x=td,y=df_in[ycol],name=ycol,yaxis='y' if ycol not in ya2c else 'y2')
        else:
            b = go.Scatter(x=td,y=df_in[ycol],name=ycol,yaxis='y' if ycol not in ya2c else 'y2')
        data.append(b)

    # create a layout
    layout = go.Layout(
        title=plot_title,
        xaxis=dict(
            ticktext=tdvals,
            tickvals=tdvals,
            tickangle=45,
            type='category'),
        yaxis=dict(
            title='y main' if y_left_label is None else y_left_label
        ),
        yaxis2=dict(
            title='y alt' if y_right_label is None else y_right_label,
            overlaying='y',
            side='right'),
        margin=go.Margin(
            b=100
        )        
    )

    fig = go.Figure(data=data,layout=layout)
    
    fig.update_layout(
        title={
            'text': plot_title,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    if figsize is not None:
        fig.update_layout(
            autosize=False,width=figsize[0],height=figsize[1])
    return fig

def plotly_shaded_rectangles(beg_end_date_tuple_list,fig):
    ld_shapes = []
    for beg_end_date_tuple in beg_end_date_tuple_list:
        ld_beg = beg_end_date_tuple[0]
        ld_end = beg_end_date_tuple[1]
        ld_shape = dict(
            type="rect",
            # x-reference is assigned to the x-values
            xref="x",
            # y-reference is assigned to the plot paper [0,1]
            yref="paper",
            x0=ld_beg,
            y0=0,
            x1=ld_end,
            y1=1,
            fillcolor="LightSalmon",
            opacity=0.5,
            layer="below",
            line_width=0,
        )
        ld_shapes.append(ld_shape)

    fig.update_layout(shapes=ld_shapes)
    return fig

 
