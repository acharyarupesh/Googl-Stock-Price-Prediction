#!/usr/bin/env python
# coding: utf-8

# ### Moving average: This type of average is designed to respond quickly to price changes.

# In[1]:


# Importing libraries and loading data into the dataframe.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style='whitegrid', context='talk', palette='Dark2') # Setting the style of the plots.
my_year_month_fmt = mdates.DateFormatter('%m/%y') # Creating a custom data formatter.
import warnings

warnings.filterwarnings('ignore')


# In[2]:


data=pd.read_csv("merged_data.csv")


# In[3]:


data.head(5)


# In[4]:


df=data[['Date', 'SP500', 'Average_OC']]


# In[5]:


df.head(5)


# In[6]:


# Indexing the date to plot a good time-series on the graph
import datetime
df['Date'] = pd.to_datetime(df['Date'])
df.index = df['Date']
del df['Date']


# ### Exponential moving average for 5, 10 and 20 days for GOOGLE STOCK 

# In[7]:


# Calculating a 20-days span EMA. adjust=False specifies that we are interested in the recursive calculation mode.
start_date = '2016-01-06'
end_date = '2018-08-30'


ema_five = df.ewm(span=5, adjust=False).mean()
ema_ten = df.ewm(span=10, adjust=False).mean()
ema_twenty = df.ewm(span=20, adjust=False).mean()

fig, ax = plt.subplots(figsize=(17,10))

ax.plot(df.loc[start_date:end_date, :].index, df.loc[start_date:end_date, 'Average_OC'], label='Actual Price', color = 'red')
ax.plot(ema_five.loc[start_date:end_date, :].index, ema_five.loc[start_date:end_date, 'Average_OC'], label = '5-days EMA', color = 'green')
ax.plot(ema_ten.loc[start_date:end_date, :].index, ema_ten.loc[start_date:end_date, 'Average_OC'], label = '10-days EMA', color = 'orange')
ax.plot(ema_twenty.loc[start_date:end_date, :].index, ema_twenty.loc[start_date:end_date, 'Average_OC'], label = '20-days EMA', color = 'blue')

ax.legend(loc='best')
ax.set_title('EWMA - 5, 10, 20 Days Span')
ax.set_xlabel('Obsevation Date')
ax.set_ylabel('Price in $')
ax.xaxis.set_major_formatter(my_year_month_fmt)


# ### MOVING AVERAGE TRADING STRATEGY:
# BUY when 5 day EMA & 10 day EMA crosses from below to above the 20 days EMA (that is the GREEN & yellow line crosses BLUE line from down to top) and HOLD or create a alert strategy to prepare for BUY when 5 days EMAs crosses 20 days EMA. Also, the rule can be applies vice-versa for SELL when the 5 day & 10 day EMA crosses from above to BELOW the 20 days EMA

# In[8]:


#implementation
trad_strat = ema_five.apply(np.sign)

for index, row in trad_strat.iterrows():
    if ema_five.loc[index,'Average_OC'] > ema_twenty.loc[index,'Average_OC'] and ema_ten.loc[index,'Average_OC'] > ema_twenty.loc[index,'Average_OC']:
        trad_strat.loc[index,'Average_OC'] = 1   
    elif ema_five.loc[index,'Average_OC'] < ema_twenty.loc[index,'Average_OC'] and ema_ten.loc[index,'Average_OC'] < ema_twenty.loc[index,'Average_OC']:
        trad_strat.loc[index,'Average_OC'] = -1   
    else:
        trad_strat.loc[index,'Average_OC'] = 0


# In[9]:


# initial price 756.90
# last price 1256.80
initial_amt = 100000
total_shares = 30
print('Account balance ',initial_amt)
print('Initial number of shares owned ',total_shares)
avg_price = 756.90
initial_val = 122707  #initial amount+total_shares+avg_price

for index, row in trad_strat.iterrows():
    if trad_strat.loc[index,'Average_OC'] == 1:
        if (initial_amt - df.loc[index,'Average_OC'])> 0 :
            initial_amt -= df.loc[index,'Average_OC']
            avg_price = ((avg_price*total_shares)+df.loc[index,'Average_OC'])/(total_shares+1)
            total_shares = total_shares + 1;
            df.loc[index,'Signal'] = "Buy"
        else:
            df.loc[index,'Signal'] = "Buy Alert"         
    elif trad_strat.loc[index,'Average_OC'] == -1:
        if total_shares - 1 > 0 :
            df.loc[index,'Signal'] = "Sell"
            avg_price = ((avg_price*total_shares)-df.loc[index,'Average_OC'])/(total_shares-1)
            initial_amt += df.loc[index,'Average_OC'];
            total_shares = total_shares - 1;
        else:
            df.loc[index,'Signal'] = "Sell Alert"
    else:
        df.loc[index,'Signal'] = "Hold"

print('----------------Signals Given During Day To Day Trade------------------------')

total_val = initial_amt + total_shares * 1256.80  
print(df.head(20))

print('----------------Results-------------------------')
print('Balance left in Account',initial_amt)
print('Total Number of shares ',total_shares)
print('Total value of the shares ',total_val)
print('Percetage Profit ',((total_val-initial_val)/initial_val)*100)


# ### Portfolio Return Trading Strategy

# In[10]:


# Taking the difference between the prices and the EMA timeseries
df1 = df[['Average_OC','SP500']].copy()
trading_positions_raw = df1 - ema_five
trading_positions_raw.tail(10)


# In[11]:


# Taking the sign of the difference to determine whether the price or the EMA is greater
trading_positions = trading_positions_raw.apply(np.sign) *(1/2)
trading_positions.tail(10)


# In[12]:


# Lagging our trading signals by one day.
trading_positions_final = trading_positions.shift(1)
trading_positions_final.tail()


# In[13]:


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(17,10))

ax1.plot(df.loc[start_date:end_date, :].index, df.loc[start_date:end_date, 'Average_OC'], label='Actual Price', color = 'blue')
ax1.plot(ema_five.loc[start_date:end_date, :].index, ema_five.loc[start_date:end_date, 'Average_OC'], label = 'Span 20-days EMA', color = 'red')

ax1.set_title('20-days Span EWMA with Trading Position')
ax1.set_ylabel('Price in $')
ax1.legend(loc='best')
ax1.xaxis.set_major_formatter(my_year_month_fmt)

ax2.plot(trading_positions_final.loc[start_date:end_date, :].index, trading_positions_final.loc[start_date:end_date, 'Average_OC'], 
        label='Trading position')
ax2.set_title('Trading Signal')
ax2.set_xlabel('Obsevation Date')
ax2.set_ylabel('Trading position')
ax2.xaxis.set_major_formatter(my_year_month_fmt)


#                                  Positive value means Buy and Negative value means sell.

# In[14]:


# Log returns - First the logarithm of the prices is taken and the the difference of consecutive (log) observations
asset_log_returns = np.log(df1).diff()
asset_log_returns.head()


# To get all the strategy log-returns for all days, multiplying the strategy positions with the asset log-returns.

# In[15]:


strategy_asset_log_returns = trading_positions_final * asset_log_returns
strategy_asset_log_returns.tail()


# ### plot the cumulative log-returns and the cumulative total relative returns of our strategy for each of the assets

# In[16]:


# Get the cumulative log-returns per asset
cum_strategy_asset_log_returns = strategy_asset_log_returns.cumsum()

# Transform the cumulative log returns to relative returns
cum_strategy_asset_relative_returns = np.exp(cum_strategy_asset_log_returns) - 1

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,9))

for c in asset_log_returns:
    ax1.plot(cum_strategy_asset_log_returns.index, cum_strategy_asset_log_returns[c], label=str(c))
ax1.set_ylabel('Cumulative log-returns')
ax1.legend(loc='best')
ax1.xaxis.set_major_formatter(my_year_month_fmt)

for c in asset_log_returns:
    ax2.plot(cum_strategy_asset_relative_returns.index, 100*cum_strategy_asset_relative_returns[c], label=str(c))

ax2.set_ylabel('Total relative returns (%)')
ax2.legend(loc='best')
ax2.xaxis.set_major_formatter(my_year_month_fmt)


# In[17]:


# Total strategy relative returns. This is the exact calculation.
cum_relative_return_exact = cum_strategy_asset_relative_returns.sum(axis=1)

# Get the cumulative log-returns per asset
cum_strategy_log_return = cum_strategy_asset_log_returns.sum(axis=1)

# Transform the cumulative log returns to relative returns. This is the approximation
cum_relative_return_approx = np.exp(cum_strategy_log_return) - 1

fig, ax = plt.subplots(figsize=(16,9))

ax.plot(cum_relative_return_exact.index, 100*cum_relative_return_exact, label='Exact')
ax.plot(cum_relative_return_approx.index, 100*cum_relative_return_approx, label='Approximation')

ax.set_title('Total cumulative relative returns (%)')
ax.set_ylabel('Total cumulative relative returns (%)')
ax.legend(loc='best')
ax.xaxis.set_major_formatter(my_year_month_fmt)


# In[18]:


def print_portfolio_yearly_statistics(portfolio_cumulative_relative_returns, days_per_year = 52 * 5):

    total_days_in_simulation = portfolio_cumulative_relative_returns.shape[0]
    number_of_years = total_days_in_simulation / days_per_year

    # The last data point will give us the total portfolio return
    total_portfolio_return = portfolio_cumulative_relative_returns[-1]
    # Average portfolio return assuming compunding of returns
    average_yearly_return = (1 + total_portfolio_return)**(1/number_of_years) - 1

    print('Total portfolio return is: ' + '{:5.2f}'.format(100*total_portfolio_return) + '%')
    print('Average yearly return is: ' + '{:5.2f}'.format(100*average_yearly_return) + '%')

print_portfolio_yearly_statistics(cum_relative_return_exact)

