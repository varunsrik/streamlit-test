#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:28:19 2024

@author: varun
"""

# Load the CSV file
df = pd.read_csv('trading_days_list.csv')

# Display the dataframe
st.write("## Trading Days List")
st.dataframe(df)