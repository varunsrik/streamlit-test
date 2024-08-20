#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:28:19 2024

@author: varun
"""
import streamlit as st
import pandas as pd


st.header('Test App')
# Load the CSV file
df = pd.read_csv('trading_days.csv')

# Display the dataframe
st.write("## Trading Days List")
st.dataframe(df)
