import matplotlib.pyplot as plt

from statsmodels.tsa import stattools
import stattools
import statsmodels.formula.api as smf            # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs


from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

df_historicalindexdata = df_data
df_historical_Marketcap = df_giw_eligibility_fields_data[['tradedate','issueid','euroMarketCap']]
df_historical_turnonver = df_eligibilityFieldsDatapull[['effectivedate','issueid','turnoverlocal']]


marketcap_dict = {}
for each_issueid in df_historical_Marketcap['issueid']:
  marketcap_dict[each_issueid] = df_historical_Marketcap[df_historical_Marketcap['issueid'] == each_issueid]

turnover_dict = {}
for each_issueid in df_historical_turnonver['issueid']:
  turnover_dict[each_issueid] = df_historical_turnonver[df_historical_turnonver['issueid']==each_issueid]

for each_dataframe, df in marketcap_dict.items():
  df['tradedate'] = pd.to_datetime(df['tradedate'])
  df.sort_values(by = 'tradedate', inplace = True)
  df = df.drop(['issueid'], axis=1)
  
  marketcap_dict[each_dataframe] = df

for each_dataframe, df in turnover_dict.items():
  df['effectivedate'] = pd.to_datetime(df["effectivedate"])
  df.sort_values(by = 'effectivedate', inplace = True)
  df = df.drop(['issueid'], axis=1)

  turnover_dict[each_dataframe] = df

df_marketcap = marketcap_dict["21000018"]
df_marketcap

df_marketcap['year_month'] = df_marketcap['tradedate'].dt.to_period('M')
df_marketcap.sort_values(by = 'year_month', inplace = True)

df_marketcap = df_marketcap.groupby(['year_month']).agg({'euroMarketCap':"mean"})

df_turnover = turnover_dict["21000018"]
df_turnover

df_turnover['year_month'] = df_turnover['effectivedate'].dt.to_period('M')
df_turnover.drop(["effectivedate"], axis=1)
df_turnover.sort_values(by = 'year_month', inplace = True)


df_turnover = df_turnover.groupby(['year_month']).agg({'turnoverlocal':"mean"})


decomposition_check = seasonal_decompose(df_marketcap['euroMarketCap'], model='additive', period = 10)

marketcap_adfuller = adfuller(df_marketcap['euroMarketCap'])


if marketcap_adfuller[1] <= 0.05:
  print("Data is stationary")
  df_stationary_marketcap_data = df_marketcap
else:
  print("Data is non stationary")
  df_marketcap


df_marketcap["euroMakretCap_seasonal_difference"] =  df_marketcap["euroMarketCap"] - df_marketcap["euroMarketCap"].shift(9)
df_marketcap.head()

marketcap_adfuller = adfuller(df_marketcap['euroMakretCap_seasonal_difference'].dropna())
print(marketcap_adfuller[1])
if marketcap_adfuller[1] <= 0.05:
  print("Data is stationary")
  df_stationary_marketcap_data = df_marketcap
else:
  print("Data is non stationary")
  df_marketcap

def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test
        
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()


# Dividing the data into train and test
# split the dataset
df_marketcap_train = df_marketcap[0: int(len(df_marketcap)*0.8)]
df_marketcap_test  = df_marketcap[int(len(df_marketcap)*0.8):]
print(len(df_marketcap_train))
print(len(df_marketcap_test))


model = sm.tsa.statespace.SARIMAX(df_marketcap_train["euroMarketCap"], order=(1,2,1), seasonal_order = (1,2,1,12))