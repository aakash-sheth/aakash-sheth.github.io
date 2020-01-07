---
title: "Data Wrangling Project"
date: 2020-01-07
tags: [data wrangling, data science, data science for finance, data visualization ]
header:
  image: "/images/perceptron/sharpe_ratio_header.jpg"
excerpt: "Investment analysis using data science"
mathjax: "true"
---

# How to use data science to compare two different investments apple to apple?

An investment may make sense if we expect it to return more money than it costs. But returns are only part of the story because they are risky - there may be a range of possible outcomes. How does one compare different investments that may deliver similar results on average, but exhibit different levels of risks? \n

The best measure to comapre two investments apple to apple with different return and risk is **Sharpe Ratio**.It compares the expected returns for two investment opportunities and calculates the additional return per unit of risk an investor could obtain by choosing one over the other. In particular, it looks at the difference in returns for two investments and compares the average difference to the standard deviation (as a measure of risk) of this difference.\n

$$ Sharpe Ratio= \frac{(R_x-R_t)}{Std R_x}$$

**Where**
**$ R_x= $ Expected investment return**
**$ R_t= $ Risk free rate of return**
**$ Std R_x $= Standard daviation of portfolio return**

 A higher Sharpe ratio means that the reward will be higher for a given amount of risk. It is common to compare a specific opportunity against a benchmark that represents an entire category of investments. \n

The Sharpe ratio is usually calculated for a portfolio and uses the risk-free interest rate as benchmark. The standard risk free interest rate is 3-month Treasury Bill Rate. Here, to make things easier I will consider S&P 500 as my risk free rate (Benchmark) and instead of entier portfolio I will just consider Amazon and Microsoft Share prices. you can easily find share prices online on websites like yahoo finance, google finance etc. If you want to follow along with me, you can find data [here](https://github.com/aakash-sheth/sharpe-ratio).\n

Lets get started now.
```python
#Lets import all required Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

When analyzing financial data, often its a good idea to use datatime index. You can declare that while reading your files as I am doing it over here.

```python
#Lets read and laod required data into pandas dataframe
benchmark_data=pd.read_csv('SNP500.csv', parse_dates=['Date'], index_col=['Date']).dropna()
MSFT=pd.read_csv('MSFT.csv',parse_dates=['Date'], index_col=['Date']).dropna()
AMZN=pd.read_csv('AMZN.csv',parse_dates=['Date'], index_col=['Date']).dropna()
```

Lets now combine **MSFT** and **AMZN** dataframe into a single dataframe.
```python
# Combining MSFT and AMZN into stock data
combined_data=pd.merge(MSFT,AMZN,suffixes=['_MSFT','_AMZN'],on=['Date'])
combined_data.head()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/share_ratio/head1.png" alt="Combined Data Head">

```python
# For our analysiswe just need daily closing prices for Amazon and Microsoft, also for the benchmark.
# Stock data contains Amazon and Microsoft daily closing share prices.
stock_data=combined_data[['Close_AMZN','Close_MSFT']]
stock_data.columns=['AMZN','MSFT']
print(stock_data.head())
```
<img src="{{ site.url }}{{ site.baseurl }}/images/share_ratio/head2.png" alt="Combined Data Head">














