# Time-series-analysis
Student project of Algorithm trading coursework
## Data
Bitcoin Market Price in USD, with length $T=300$, begin from 2022-04-12, end on 2023-02-05.
```
code = 'BCHAIN/MKPRU'
data = nasdaqdatalink.get(code, start_date='2022-04-12', end_date='2023-02-05')
data.to_csv(path + code.split('/')[0]+'-'+code.split('/')[1]+'.csv')
```
## Pipeline
- Price and Return calculation
- Moving average
```
def MovAve(data, tau):
  return data.rolling(tau).mean()
```
- PACF and ACF
```
def stat_test(data, lag):
  if lag == 0:
    D_data = data
  else:
    D_data = data.diff(lag).dropna()
  tt = True
  st_test = True
  lb_test = True

  p_t = ADF(D_data)
  if p_t[1] > 0.05:
    st_test = False
    tt = False

  lb_t = acorr_ljungbox(D_data, lags=1)
  if lb_t.loc[1][1] > 0.05:
    lb_test = False
    tt = False

  ACF_data = acf(D_data)
  if tt == True:
    plot_acf(D_data)
    plt.ylabel("Correlations")
    plt.xlabel("Lags")
    plt.tight_layout()
    plt.title('The price ACF when lag = '+ str(lag))
    plt.grid()
    plt.show()
    
  return ACF_data, p_t, lb_t, st_test, lb_test, tt
  
  # Calculate partial autocorrelation
def sta_PACF(data, lag):
  if lag == 0:
    df = data
  else: 
    df = data.diff(lag)
    df = df.dropna()
  return pacf(df)
```
- ARMA models
```
def ARMA_model(d, train, test):
  train.replace([np.inf, -np.inf], np.nan, inplace=True)
  ## search for right parameters automatically
  start_time = time.time() #start time

  model = pm.auto_arima(train,
                start_p=1, start_q=1, # p,q
                max_p=19, max_q=4, # Maximum p, q
                d = d,            # difference
                m = 1, 
               # start_P=1, start_Q=1, # p,q  
               # max_P=3, max_Q=3,  
               # max_D = 7,       # season
                seasonal= False,   # seasonal
                trace=True,error_action='ignore',  
                suppress_warnings=True, stepwise=True)
  print(model.summary())
  y_pred, confint = model.predict(60, return_conf_int=True) #predictions of future 60 periods
  cf= pd.DataFrame(confint)
  t= model.predict_in_sample().to_frame()
  train_pred = t[d:]

  MSE_cal(train['Value'].iloc[d:], train_pred, test['Value'], y_pred)

  Vis_tr_te(train, train_pred, test, y_pred, d)


  end_time = time.time() #end time
  print("cost time： ", end_time - start_time) #cost
  return y_pred, model, cf
```
  - Train and test set split
  - Loss: MSE
  ```
  def MSE_cal(tr, tr_pred, te, te_pred):
  # MSE of the model
  mse_tr = mean_squared_error(tr, tr_pred)
  mse_t = mean_squared_error(te, te_pred)
  print("MSE of train： ", mse_tr)
  print("MSE of test： ", mse_t)
  ```
  - Comparison: Baseline models
    - Simple moving average
    - Simple exponential smoothing
    - SVD
- Gussianity and Stationary test
```
# Shapiro-Wilk Test
def SW_t(data, seed = 42):

  # seed the random number generator
  # generate univariate observations
  # normality test
  stat, p = shapiro(data)

  print('Statistics=%.3f, p=%.3f' % (stat, p))

  # interpret
  alpha = 0.05

  if p > alpha:
      print('Sample looks Gaussian (fail to reject H0)')
  else:
      print('Sample does not look Gaussian (reject H0)')
  return stat, p
```
