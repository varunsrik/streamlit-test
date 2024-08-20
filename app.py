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

days = pd.read_csv('trading_days.csv', index_col = 0)
days.index = pd.to_datetime(days.index)
current_date = days.index[-1]

st.header(f'FNO Dashboard for {current_date}')

tab1, tab2, tab3, tab4 = st.tabs(["Expiry Comparison", "Backwardation", "Industry", "Stock Details"])

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
    

        
