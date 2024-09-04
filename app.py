#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:28:19 2024

@author: varun
"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import time
import pytz


days = pd.read_csv('trading_days.csv', index_col = 0)
days.index = pd.to_datetime(days.index)
current_date = days.index[-1]

st.header(f'FNO Dashboard for {current_date.day_name()}, {str(current_date.day)} {current_date.month_name()} {str(current_date.year)}')

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Expiry Comparison", "Backwardation", "Industry", "Stock Details", "Momentum Screens", "Relative Rotation Graph"])

with tab1:
    expiry_df = pd.read_csv('expiry_table.csv', index_col = 0)
    st.dataframe(expiry_df)
    
    oi_up_backwardation = expiry_df[(expiry_df['oi_pct_change']>0)&(expiry_df['current_basis']<0)]
    oi_down_backwardation = expiry_df[(expiry_df['oi_pct_change']<0)&(expiry_df['current_basis']<0)]
    oi_up_contango = expiry_df[(expiry_df['oi_pct_change']>0)&(expiry_df['current_basis']>0)]
    oi_down_contango = expiry_df[(expiry_df['oi_pct_change']<0)&(expiry_df['current_basis']>0)]
    
    st.subheader('OI ⬆️ and Basis is 🔻')
    st.dataframe(oi_up_backwardation)
  
    st.subheader('OI ⬇️ and Basis is 🔻')
    st.dataframe(oi_down_backwardation)
    
    st.subheader('OI ⬆️ and Basis is ⬆️')
    st.dataframe(oi_up_contango)

    st.subheader('OI ⬇️ and Basis is ⬆️')
    st.dataframe(oi_down_contango)
    

with tab2:
    backwardation_df = pd.read_csv('backwardation_table.csv', index_col = 0)
    st.dataframe(backwardation_df)
    

    
with tab3:
    industry_df = pd.read_csv('industry_table.csv', index_col = 0)
    st.dataframe(industry_df)
    ind_button = st.selectbox('Select Industry', industry_df.index)
    if ind_button:
        sub_ind_df = pd.read_csv(f'industry_sub_{ind_button}_table.csv', index_col = 0)
        st.dataframe(sub_ind_df)

with tab4:
    stock_df = pd.read_csv('stock_table.csv', index_col = 0)
    stock_df.index = pd.to_datetime(stock_df.index)
    
    selected_stock = st.selectbox('Select a stock:', backwardation_df.index)
    cols = ['open', 'high', 'low', 'close', 'delivery_pct', 'f30d_basis_pct', 'open_interest']
    cols = [selected_stock+'_'+col for col in cols]
    stock_df_slice = stock_df[cols]
    stock_df_slice.columns = stock_df_slice.columns.str.lstrip(selected_stock+'_')

    df_last_year = stock_df_slice.loc['Mar-2024':]

    st.dataframe(df_last_year)
    
    
    
    # Create a figure with 3 rows and shared x-axis
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        row_heights=[0.25, 0.2, 0.2, 0.2], vertical_spacing=0.15)
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df_last_year.index,
        open=df_last_year['open'],
        high=df_last_year['high'],
        low=df_last_year['low'],
        close=df_last_year['close'],
        name='Candlestick'
    ), row=1, col=1)
    
    # Add futures_basis plot
    fig.add_trace(go.Scatter(
        x=df_last_year.index,
        y=df_last_year['f30d_basis_pct'],
        mode='lines',
        name='Futures Basis'
    ), row=2, col=1)
    
    # Add open_interest plot
    fig.add_trace(go.Scatter(
        x=df_last_year.index,
        y=df_last_year['open_interest'],
        mode='lines',
        name='Open Interest'
    ), row=3, col=1)

    # Add delivery percent plot

    fig.add_trace(go.Scatter(
        x=df_last_year.index,
        y=df_last_year['delivery_pct'],
        mode='lines',
        name='Delivery Percentage'
    ), row=4, col=1)


    
    # Update layout
    fig.update_layout(
        title='Stock Data Visualization',
        xaxis=dict( rangeslider=dict(
             visible=True,  # Show the range slider
             thickness=0.1  # Adjust the thickness so it doesn't obscure the subplots
         ), title='Date'),
        yaxis=dict(title='Price'),
        xaxis2=dict(rangeslider=dict(visible=False), title='Date'),
        yaxis2=dict(title='Futures Basis'),
        xaxis3=dict(rangeslider=dict(visible=False), title='Date'),
        yaxis3=dict(title='Open Interest'),
          xaxis4=dict(rangeslider=dict(visible=False), title='Date'),
        yaxis4=dict(title='Delivery Percent')
       #t , xaxis_rangeslider_visible=True 
    )
    
    # Adjust x-axis labels visibility
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    
    # Adjust margins
    fig.update_layout(margin=dict(l=0, r=0, t=4, b=0))
    
    # Display the combined chart
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    symb_list = pd.read_csv('nifty500list.csv')['Symbol'].to_list()
    def yf_downloader(symbol_list):
        symbol_list_yf = [symbol+'.NS' for symbol in symbol_list]
        df = yf.download(symbol_list_yf, start = '2023-1-1')[['Adj Close', 'Volume']]
        close_df, volume_df = df['Adj Close'], df['Volume']
        volume_df = volume_df.apply(lambda x: round(x/x.rolling(20).mean(),1))
        close_df.columns = close_df.columns.str.rstrip('.NS')
        volume_df.columns = volume_df.columns.str.rstrip('.NS')
        return [close_df, volume_df]
   
    def output_momentum_screen(symb_list):
        result = yf_downloader(symb_list)
        close_df = result[0]
        volume_df = result[1]
        volume_series = volume_df.iloc[-1]
        today_datetime = pd.Timestamp(dt.datetime.today(),  tz='Asia/Kolkata')
        st.header(f'Live Momentum Screen')
        st.write(st.write(dt.datetime.today(), tzinfo=pytz.timezone("Asia/Kolkata")))
        st.subheader('Nifty 500 List')
        final = pd.DataFrame(index = close_df.columns, columns = ['high_low_signal'])
        final['high_low_signal'] = np.where(
            close_df.iloc[-1]>=close_df.rolling(252).max().iloc[-1], '252 day high', 
            np.where(close_df.iloc[-1]>=close_df.rolling(100).max().iloc[-1], '100 day high',
                     np.where(close_df.iloc[-1]>=close_df.rolling(50).max().iloc[-1], '50 day high',
                              np.where(close_df.iloc[-1]>=close_df.rolling(20).max().iloc[-1], '20 day high',
                                       np.where(close_df.iloc[-1]>=close_df.rolling(5).max().iloc[-1], '5 day high',
                              np.where(close_df.iloc[-1]<=close_df.rolling(252).min().iloc[-1], '252 day low',
                              np.where(close_df.iloc[-1]<=close_df.rolling(100).min().iloc[-1], '100 day low',
                                       np.where(close_df.iloc[-1]<=close_df.rolling(50).min().iloc[-1], '50 day low',
                              np.where(close_df.iloc[-1]<=close_df.rolling(20).min().iloc[-1], '20 day low',
                                       np.where(close_df.iloc[-1]<=close_df.rolling(5).min().iloc[-1], '5 day low', '-')
                                       )))))))))
        
        for window in [1,3,5,10,20,60]:
            final[f'{str(window)}d_return'] = round((close_df.iloc[-1] - close_df.iloc[-1-window])*100/close_df.iloc[-1-window],2)
        for symbol in final.index:
            final.loc[symbol, 'volume_signal'] = volume_series.loc[symbol]
        fno_stocks = expiry_df.index
        fno_stocks = fno_stocks.intersection(final.index)
        final['is_fno'] = False
        for symbol in fno_stocks:
            final.loc[symbol, 'is_fno'] = True
        st.dataframe(final)
    
        for criterion in ['252 day high', '100 day high', '50 day high', '20 day high', '5 day high', '252 day low', '100 day low', '50 day low', '20 day low', '5 day low']:
            temp = final[final['high_low_signal'] == criterion]
            if len(temp) > 0:
                st.subheader(f'Stocks making a new {criterion}')  
                st.dataframe(temp)
        
    mom_button = st.button('Run Momentum Screen')
    if mom_button:
        output_momentum_screen(symb_list)

with tab6:
    
# Function to calculate MACD and normalize it
    def calc_macd(price, slow=26, fast=12, signal=9):
        exp1 = price.ewm(span=fast, adjust=False).mean()
        exp2 = price.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        normalized_macd = (macd - signal_line) / price
        return normalized_macd


    st.subheader('Relative Rotation Graph (RRG)')
    sector_dict = {'Nifty Auto': '^CNXAUTO',
                   'Nifty Commodities': '^CNXCMDT',
                   'Bank Nifty': '^NSEBANK',
                   'Nifty IT': '^CNXIT',
                   'Nifty Infra': '^CNXINFRA',
                   'Nifty Energy': '^CNXENERGY',
                   'Nifty FMCG': '^CNXFMCG',
                   'Nifty Pharma': 'PHARMABEES.NS',
                   'Nifty Media': '^CNXMEDIA',
                   'Nifty Metal': '^CNXMETAL',
                   'Nifty PSU Bank': '^CNXPSUBANK',
                   'Nifty PSE': '^CNXPSE',
                   'Nifty Consumption': '^CNXCONSUM',
                   'Nifty Realty': '^CNXREALTY'}
    benchmark_dict = {'Nifty': '^NSEI'}
                   
    
    sectors = st.multiselect('Select Sectoral Indices', sector_dict.keys(), default=['Bank Nifty', 'Nifty IT'])
    benchmark = st.selectbox('Select Benchmark',benchmark_dict.keys(), index=0)
    tail_length = st.slider('Tail Length (weeks)', 1, 15, 5)
    freq = st.radio("Frequency", ('Weekly', 'Daily'))
    st.write(sectors, benchmark)
    # Download data
    end_date = dt.datetime.now().date()
    start_date = end_date - dt.timedelta(weeks=52)

    st.write(sector_dict['Nifty IT'])
    yf_sector_list = [sector_dict[sector] for sector in sectors]
    #yf_sector_list = yf_sector_list.append(benchmark_dict[benchmark])
    st.write(yf_sector_list)
    prices = yf.download(yf_sector_list, start=start_date, end=end_date)['Adj Close']
  
    renamed_columns = sectors.append(benchmark)
    prices.columns = renamed_columns
    prices[sectors] = prices[sectors].div(prices[benchmark], axis=0)
    st.write(prices)
    # Resample for weekly data if needed
    if freq == 'Weekly':
        prices = prices.resample('W-FRI').last()

    st.write(prices)
    # Calculate returns and relative strength
    returns = prices.pct_change().dropna()
    relative_strength = returns
    lambda_func = lambda x: (x + 1).prod() - 1
    relative_strength = relative_strength.rolling(window=4).apply(lambda_func, raw=True)




    
    #window = 1
    #relative_strength = (returns - returns.rolling(window).mean())/returns.rolling(window).std()
    #relative_strength = relative_strength.ewm(span=window).mean()
    
    # Calculate momentum for each sector
    momentum = prices.apply(calc_macd)

    
    # Plotly figure setup for RRG
    fig_rrg = go.Figure()
    
    fig_rrg.add_shape(type="rect", x0=0, y0=0, x1=relative_strength.max().max(), y1=momentum.max().max(),
                      xref="x", yref="y",
                      fillcolor="lightgreen", opacity=0.3, layer="below", line_width=0)
    
    fig_rrg.add_shape(type="rect", x0=0, y0=0, x1=relative_strength.max().max(), y1=momentum.min().min(),
                      xref="x", yref="y",
                      fillcolor="yellow", opacity=0.3, layer="below", line_width=0)
    
    fig_rrg.add_shape(type="rect", x0=0, y0=0, x1=relative_strength.min().min(), y1=momentum.min().min(),
                      xref="x", yref="y",
                      fillcolor="lightcoral", opacity=0.3, layer="below", line_width=0)
    
    fig_rrg.add_shape(type="rect", x0=0, y0=0, x1=relative_strength.min().min(), y1=momentum.max().max(),
                      xref="x", yref="y",
                      fillcolor="lightblue", opacity=0.3, layer="below", line_width=0)
    
    
    
    
    for sector in sectors:
        fig_rrg.add_trace(go.Scatter(
            x=relative_strength[sector].tail(tail_length),
            y=momentum[sector].tail(tail_length),
            mode='lines+markers', line=dict(shape="spline"),
            name=sector,
            text=momentum.index[-tail_length:].strftime('%Y-%m-%d'),
            hovertemplate='<b>Date:</b> %{text}<br><b>RS:</b> %{x}<br><b>Momentum:</b> %{y}<extra></extra>',
            marker=dict(size=6),
        ))
    
        # Highlight the last point
        fig_rrg.add_trace(go.Scatter(
            x=[relative_strength[sector].iloc[-1]],
            y=[momentum[sector].iloc[-1]],
            mode='markers+text',
            text=[sector],
            textposition='top right',
            hovertext=[momentum.index[-1].strftime('%Y-%m-%d')],
            hovertemplate='<b>Date:</b> %{hovertext}<br><b>RS:</b> %{x}<br><b>Momentum:</b> %{y}<extra></extra>',
            marker=dict(size=12, symbol='diamond', color='red', line=dict(width=2, color='black')),
            showlegend=False
        ))
    
    fig_rrg.update_layout(
        title='Relative Rotation Graph (RRG) with Normalized MACD Momentum',
        xaxis_title='Relative Strength',
        yaxis_title='Normalized Momentum (MACD)',
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'),
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'),
        legend=dict(x=0.9, y=1),
        width=800,
        height=800,
        template="plotly_white"
    )
    
    # Add text annotations for each quadrant
    fig_rrg.add_annotation(
        text="Improving", xref="x domain", yref="y domain",
        x=0.05, y=0.95, showarrow=False,
        font=dict(size=16, color="black", family="Arial, sans-serif"))
    
    
    fig_rrg.add_annotation(
        text="Leading", xref="x domain", yref="y domain",
        x=0.95, y=0.95, showarrow=False,
        font=dict(size=16, color="black", family="Arial, sans-serif"),
        xanchor="right")
    
    fig_rrg.add_annotation(
        text="Weakening", xref="x domain", yref="y domain",
        x=0.95, y=0.05, showarrow=False,
        font=dict(size=16, color="black", family="Arial, sans-serif"),
        xanchor="right", yanchor="bottom")
    
    fig_rrg.add_annotation(
        text="Lagging", xref="x domain", yref="y domain",
        x=0.05, y=0.05, showarrow=False,
        font=dict(size=16, color="black", family="Arial, sans-serif"),
        yanchor="bottom")
    
    
    
    # Display the initial plot
    plot_rrg = st.plotly_chart(fig_rrg)
    
    # Animation feature using Streamlit buttons
    if st.button('Animate'):
        for i in range(1, tail_length + 1):
            for j, sector in enumerate(sectors):
                fig_rrg.data[2*j].update(
                    x=relative_strength[sector].tail(i),
                    y=momentum[sector].tail(i),
                    text=momentum.index[-i:].strftime('%Y-%m-%d')
                )
                fig_rrg.data[2*j+1].update(
                    x=[relative_strength[sector].iloc[-i]],
                    y=[momentum[sector].iloc[-i]],
                    hovertext=[momentum.index[-i].strftime('%Y-%m-%d')]
                )
            plot_rrg.plotly_chart(fig_rrg)
            time.sleep(0.5)  # Add a delay for animation effect
    



