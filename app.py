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


days = pd.read_csv('trading_days.csv', index_col = 0)
days.index = pd.to_datetime(days.index)
current_date = days.index[-1]

st.header(f'FNO Dashboard for {current_date.day_name()}, {str(current_date.day)} {current_date.month_name()} {str(current_date.year)}')

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Expiry Comparison", "Backwardation", "Industry", "Stock Details", "Momentum Screens"])

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
    @st.cache
    def yf_downloader(symbol_list, date):
        symbol_list_yf = [symbol+'.NS' for symbol in symbol_list]
        df = yf.download(symbol_list_yf, start = '2023-1-1')['Adj Close']
        df.columns = symbol_list
        return df
symbol_list = pd.read_csv('nifty500list.csv')['Symbol'].to_list()

df = yf_downloader(symbol_list, current_date)

today_datetime = dt.datetime.today()
st.header(f'Live Momentum Screen for {dt.datetime.today().strftime('%H:%M')}, {str(today_datetime.day)}, {today_datetime.month_name()}, {str(today_datetime.year)}')
st.subheader('Nifty 500 List')
final = pd.DataFrame(index = df.columns, columns = ['high_low_signal'])
final['high_low_signal'] = np.where(
    df.iloc[-1]>=df.rolling(252).max().iloc[-1], '252 day high', 
    np.where(df.iloc[-1]>=df.rolling(100).max().iloc[-1], '100 day high',
             np.where(df.iloc[-1]>=df.rolling(50).max().iloc[-1], '50 day high',
                      np.where(df.iloc[-1]>=df.rolling(20).max().iloc[-1], '20 day high',
                               np.where(df.iloc[-1]>=df.rolling(5).max().iloc[-1], '5 day high',
                      np.where(df.iloc[-1]<=df.rolling(252).min().iloc[-1], '252 day low',
                      np.where(df.iloc[-1]<=df.rolling(100).min().iloc[-1], '100 day low',
                               np.where(df.iloc[-1]<=df.rolling(50).min().iloc[-1], '50 day low',
                      np.where(df.iloc[-1]<=df.rolling(20).min().iloc[-1], '20 day low',
                               np.where(df.iloc[-1]<=df.rolling(5).min().iloc[-1], '5 day low', '-')
                               )))))))))
st.subheader('NIFTY FNO Stocks')
fno_stocks = expiry_df.index
fno_final = final.loc[fno_stocks]
st.dataframe(fno_final)
