import pandas as pd
import numpy as np 
import datetime
import statsmodels.api
import matplotlib.pyplot as plt
import scipy.stats as stats

# visit https://www.lendingclub.com/info/download-data.action
# select years 2013-2014 or loanStats3c.csv

# retrieve data
loanStats = pd.read_csv('\Users\zecharmo\Thinkful\loanStats3c.csv', low_memory=False)

# convert string to datetime object in pandas:
loanStats['issue_d_format'] = pd.to_datetime(loanStats['issue_d']) 
loanStats_ts = loanStats.set_index(loanStats['issue_d_format']) 
year_month_summary = loanStats_ts.groupby(lambda x : x.year * 100 + x.month).count()
loan_count_summary = year_month_summary['issue_d']

# initial plot of the time series data
loan_count_summary.plot()
plt.show()
# plot shows no clear trend or seasonality
# second half of the data has high peaks and troughs, not consistent throughout

# plot the autocorrelation function of the time series data
statsmodels.api.graphics.tsa.plot_acf(loan_count_summary)
plt.show()
# none of the vertical lines cross the blue lines
# indicating no correlation between that period and period 0

# plot the partial autocorrelation function of the time series data
statsmodels.api.graphics.tsa.plot_pacf(loan_count_summary)
plt.show()
# again, none of the vertical lines cross the upper or lower limit
# indicating no correlation between that period and period 0

# QQ-plot to test for normality
statsmodels.api.graphics.tsa.plot_pacf(loan_count_summary)
plt.show()
# plot reveals data follows a fairly normal distribution
# R squared value is .8649 which is high

# this seems to be a fairly stationary series
# with more data, perhaps viewing multiple years together, may reveal a consistent trend or seasonality