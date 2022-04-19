import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import pickle as pkl
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import statsmodels.api as sm
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')


###########
# Load Data
###########

load_t = '20220318'

df_5y = pkl.load(open('./data/df_5y_{}.pkl'.format(load_t), 'rb'))
print(f'Loaded df_5y_{load_t}.pkl')

t = datetime.datetime.strptime(str(load_t), '%Y%m%d')
t_1 = datetime.datetime.strptime(str(load_t), '%Y%m%d') - datetime.timedelta(days=7)

try:
#     df_info_raw = pkl.load(open('df_info_raw.pkl', 'rb'))
    df_info_raw = pkl.load(open('./data/df_info_raw_{}.pkl'.format(datetime.datetime.strftime(t_1, '%Y%m%d')), 'rb'))
    print(f'Loaded df_info_raw_{load_t}.pkl')
    
except:
    print('Uncomment and run below, takes about 1 hour!!!')
    
#     df_info_raw = pd.DataFrame({ticker: tick_obj.info for ticker, tick_obj in yf_tickers.tickers.items()})
#     pkl.dump(df_info_raw, open('df_info_raw.pkl', 'wb'))
#     pkl.dump(df_info_raw, open('df_info_raw_{}.pkl'.format(datetime.datetime.strftime(datetime.datetime.today(), '%Y%m%d')), 'wb'))

fields = pkl.load(open('fields.pkl', 'rb'))
print(f'Loaded fields.pkl')
list_fields = [f for k,v in fields.items() for f in v]

df_info = df_info_raw.loc[list_fields, :].T
stocks_to_drop = df_info[df_info.isnull().sum(axis=1) > 15].index.tolist()

###########
# Functions 
###########
# remove dropped attributes from fields
def remove_from_fields_dict(fields, cat, list_remove):
    fields[cat] = [f for f in fields[cat] if f not in list_remove]
    return fields

# add attributes to fields
def add_to_fields_dict(fields, cat, list_add):
    fields[cat] = fields[cat] + list_add
    return fields

## Remove an attribute from a category
def remove_attribute_from_fields(fields, category, field):
    temp_list = fields[category]
    temp_list.remove(field)
    fields[category] = temp_list
    return fields

def log_columns_missing_y(main_df, X_col, y_col, list_idx_missing, col_prefix='log-', log_X=True, log_y=True):
    df = main_df[[X_col, y_col]].copy()
    
    # Log the X on all indices
    if log_X:
        df.loc[:, '{}{}'.format(col_prefix, X_col)] = df.loc[:, '{}'.format(X_col)].map(np.log)
        
    # log the y on non-missing indices
    if log_y:
        df.loc[~df.index.isin(list_idx_missing), '{}{}'.format(col_prefix, y_col)] = \
            df.loc[~df.index.isin(list_idx_missing), '{}'.format(y_col)].map(np.log)
    
    return df


def lr_impute(df_main, X_col, y_col, list_idx_missing, drop_from_training=[]):
    df = df_main.copy()
    X_train = df.loc[~df.index.isin(list_idx_missing + drop_from_training), X_col].to_numpy().reshape(-1, 1)
    y_train = df.loc[~df.index.isin(list_idx_missing + drop_from_training), y_col].to_numpy().reshape(-1, 1)
    
    lr = LinearRegression().fit(X_train, y_train)
        
    # Impute missing by lr predict
    X_impute = df.loc[df.index.isin(list_idx_missing), X_col].to_numpy().reshape(-1, 1)
    df.loc[df.index.isin(list_idx_missing), y_col] = lr.predict(X_impute).reshape(-1)
    
    return df

# Log return
def log_return(df_close_px, fri_weekly=False, t=t, t_1=t_1):
    if fri_weekly:
        px_t = df_close_px.resample('W-FRI').last()
        px_t_1 = df_close_px.resample('W-FRI').last().shift(1)
    else:
        px_t = df_close_px.loc[t, :]
        px_t_1 = df_close_px.loc[t_1, :]
        
    # For the future: Holidays are not handled here by filling n/a with 0 returns.  
    return np.log(px_t/px_t_1).fillna(0)

# OLS features

def print_coef_t_results(model, critical_val = 1.645, added_constant=True, skip_i=[], suppress=False):
    count=0
    for i in np.argwhere(abs(model.tvalues)>critical_val).reshape(-1):
        if i not in skip_i:
            count+=1
            if added_constant and not suppress:
                print((['constant']+features)[i], 'coef=', round(model.params[i],4), 't-value=', round(model.tvalues[i],4))
            elif not added_constant and not suppress:
                print(features[i], 'coef=', round(model.params[i],4), 't-value=', round(model.tvalues[i],4))
    if not suppress:
        print('Number of features with > {} abs t-value:'.format(critical_val), count)
    res = np.argwhere(abs(model.tvalues)>critical_val).reshape(-1)
    if len(skip_i)>0:
        for i in skip_i:
            if i in res:
                res=np.delete(res, np.where(res==i))
    return res

###########
# Data Prep
###########

# Convert all numbers to float except nominal variables in 'Location' and 'Sector industry' categories
# name df_quant
df_quant = df_info.loc[:, ~(df_info.columns.isin(fields['Location']) | df_info.columns.isin(fields['Sector industry']))].apply(lambda x: x.astype(float))

# Nominal variables in Location or Sector industry will need to one-hot encode: df_nom
df_nom = df_info.loc[:, (df_info.columns.isin(fields['Location']) | df_info.columns.isin(fields['Sector industry']))]

# Execute Data Cleaning
df_quant.drop(stocks_to_drop, axis=0, inplace=True)
df_nom.drop(stocks_to_drop, axis=0, inplace=True)

# Market Expectation fields
no_analyst_stocks = df_quant[df_quant['numberOfAnalystOpinions'].isnull()].index.tolist()
df_quant.loc[no_analyst_stocks, ['targetMeanPrice', 'targetHighPrice', 'targetLowPrice']] = \
    df_quant.loc[no_analyst_stocks, 'currentPrice']
df_quant.loc[no_analyst_stocks, ['numberOfAnalystOpinions']] = 0

# Normalize targetPrice by currentPrice
df_quant[['targetMeanPrice', 'targetHighPrice', 'targetLowPrice']] \
    = np.log(df_quant[['targetMeanPrice', 'targetHighPrice', 'targetLowPrice']].divide(df_quant['currentPrice'], axis=0))

# Size
df_quant_size = df_quant.loc[:, fields['Size']]
stocks_no_employees = df_quant_size[df_quant_size['fullTimeEmployees'].isnull()].index.tolist()



# Employee number impute
df_employ_impute = lr_impute(log_columns_missing_y(df_quant_size, 'marketCap', 'fullTimeEmployees', stocks_no_employees), 
                             'log-marketCap', 'log-fullTimeEmployees', stocks_no_employees)


# Normalize and drop
df_quant.drop('fullTimeEmployees', axis=1, inplace=True)
df_quant.loc[:, 'log-fullTimeEmployees'] = df_employ_impute.loc[:, 'log-fullTimeEmployees']
fields = remove_from_fields_dict(fields, 'Size', ['fullTimeEmployees'])
fields = add_to_fields_dict(fields, 'Size', ['log-fullTimeEmployees'])

df_quant.loc[:, 'enterprise-per-mktcap'] = df_quant.loc[:, 'enterpriseValue'] / df_info.loc[:, 'marketCap'] 
df_quant.drop('enterpriseValue', axis=1, inplace=True)
fields = remove_from_fields_dict(fields, 'Size', ['enterpriseValue'])
fields = add_to_fields_dict(fields, 'Size', ['enterprise-per-mktcap'])

df_quant.loc[:, 'BV-per-P'] = df_quant.loc[:, 'bookValue'] / df_quant.loc[:, 'currentPrice']
df_quant.drop('bookValue', axis=1, inplace=True)
fields = remove_from_fields_dict(fields, 'Size', ['bookValue'])
fields = add_to_fields_dict(fields, 'Size', ['BV-per-P'])

df_quant.drop(['currentRatio', 'quickRatio'], axis=1, inplace=True)
fields = remove_from_fields_dict(fields, 'Financial leverage', ['currentRatio', 'quickRatio'])

df_quant.loc[:, 'cash-per-price'] = df_quant.loc[:, 'totalCashPerShare'] / df_quant.loc[:, 'currentPrice']
df_quant.drop(['totalCashPerShare'], axis=1, inplace=True)
fields = add_to_fields_dict(fields, 'Financial leverage', ['cash-per-price'])
fields = remove_from_fields_dict(fields, 'Financial leverage', ['totalCashPerShare'])

stock_cash_null = df_quant[df_quant['cash-per-price'].isnull()].index.to_list()
df_quant.loc[stock_cash_null, ['cash-per-price']] = 0

# D to E Impute
no_DE_stocks = df_quant[df_quant['debtToEquity'].isnull()].index.tolist()
df_cash_debt_with_log = log_columns_missing_y(df_quant, 'cash-per-price', 'debtToEquity', no_DE_stocks)
drop_cash_debt_plot = df_cash_debt_with_log.loc[(df_cash_debt_with_log.isnull().any(axis=1) | df_cash_debt_with_log.isin([np.inf, -np.inf]).any(axis=1)), :].index.tolist()

df_de_impute = lr_impute(log_columns_missing_y(df_quant, 'cash-per-price', 'debtToEquity', no_DE_stocks), 
                         'log-cash-per-price', 'log-debtToEquity', no_DE_stocks, drop_cash_debt_plot)

# exp the log imputed values to normal scale for debtToEquity
df_quant.loc[no_DE_stocks, 'debtToEquity'] = df_de_impute.loc[no_DE_stocks, 'log-debtToEquity'].map(np.exp)

# Zero Impute fields
zero_imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
mean_imp = SimpleImputer(missing_values=np.nan, strategy='mean')
df_quant[['returnOnAssets' , 'returnOnEquity']] = zero_imp.fit_transform(df_quant[['returnOnAssets' , 'returnOnEquity']])
df_quant[['forwardPE', 'trailingPE']] = mean_imp.fit_transform(df_quant[['forwardPE', 'trailingPE']])
df_quant.drop('dividendYield', axis=1, inplace=True)
df_quant[['trailingAnnualDividendYield', 'payoutRatio']] = zero_imp.fit_transform(df_quant[['trailingAnnualDividendYield', 'payoutRatio']])
fields = remove_from_fields_dict(fields, 'Value', ['dividendYield'])

# Growth
df_quant.drop(['pegRatio', 'trailingPegRatio'], axis=1, inplace=True)
df_quant[['revenueGrowth','earningsGrowth', 'earningsQuarterlyGrowth']] \
    = zero_imp.fit_transform(df_quant[['revenueGrowth', 'earningsGrowth', 'earningsQuarterlyGrowth']])
fields = remove_from_fields_dict(fields, 'Growth', ['pegRatio', 'trailingPegRatio'])


# To norm
df_quant.drop(['52WeekChange', 'forwardEps', 'trailingAnnualDividendRate', 'dividendRate'], axis=1, inplace=True)
fields = remove_from_fields_dict(fields, 'Momentum_to_norm', ['52WeekChange', 'dayLow', 'dayHigh']);
fields = remove_from_fields_dict(fields, 'Value_to_norm', ['forwardEps', 'trailingAnnualDividendRate', 'dividendRate']);

# dayLow and dayHigh are too short term
df_quant.drop(['dayLow', 'dayHigh'], axis=1, inplace=True)

# Momentum
df_mom_normed = np.log(df_quant[fields['Momentum_to_norm']].divide(df_quant['currentPrice'], axis=0))
df_mom_normed.columns = ['{}{}'.format(c, '_logprice') for c in df_mom_normed.columns]
df_quant = df_quant.drop(fields['Momentum_to_norm'],axis=1).merge(df_mom_normed, left_index=True, right_index=True)
fields = add_to_fields_dict(fields, 'Momentum', df_mom_normed.columns.tolist())

# Volume as market value: volume times currentPrice
df_liq_price_normed = df_quant[['averageDailyVolume10Day', 'averageVolume10days', 'averageVolume']].multiply(df_quant['currentPrice'], axis=0)
df_liq_price_normed.columns = ['{}{}'.format(c, '_mv') for c in df_liq_price_normed.columns]

# Volume as % of float shares
df_liq_float_normed = df_quant[['averageDailyVolume10Day', 'averageVolume10days', 'averageVolume']].divide(df_quant['floatShares'], axis=0)
df_liq_float_normed.columns = ['{}{}'.format(c, '-perFloat') for c in df_liq_float_normed.columns]

# Float %
df_quant['float_percent'] = df_quant['floatShares'] * df_quant['currentPrice'] / df_quant['marketCap']
fields = add_to_fields_dict(fields, 'Liquidity', ['float_percent'])

df_quant = (df_quant.drop(fields['Liquidity_to_norm'],axis=1)
                     .merge(df_liq_price_normed, left_index=True, right_index=True)
                     .merge(df_liq_float_normed, left_index=True, right_index=True)
)
fields = add_to_fields_dict(fields, 'Liquidity', (df_liq_price_normed.columns.tolist() 
                                                 + df_liq_float_normed.columns.tolist() 
                                                 )
                           )
                           
# Value
df_quant['e/p'] = df_quant['trailingEps'] / df_quant['currentPrice']
df_quant['rev/mktcap'] = df_quant['totalRevenue'] / df_quant['marketCap']
df_quant.drop(fields['Value_to_norm'],axis=1, inplace=True)
fields = add_to_fields_dict(fields, 'Value', ['e/p', 'rev/mktcap'])

df_quant.drop(fields['Financial leverage_to_norm'],axis=1, inplace=True)

# Drop currentPrice
df_quant.drop('currentPrice', axis=1, inplace=True)
remove_attribute_from_fields(fields,'Momentum', 'currentPrice')

# Drop revenuePerShare
df_quant.drop('revenuePerShare', axis=1, inplace=True)
remove_attribute_from_fields(fields,'Value', 'revenuePerShare');

# Standard normalize df_quant to z-scores
df_quant_z = (df_quant - df_quant.mean(axis=0))  / df_quant.std(axis=0)
df_factors = df_quant_z.merge(pd.get_dummies(df_nom['industry']), left_index=True, right_index=True, how='left')

###########
# Returns
###########

df_5y_close = df_5y['Close'].ffill(axis=0)
df_2y = df_5y_close.loc[df_5y_close.index > (df_5y_close.index[-1] - datetime.timedelta(days=2*365)), :]
last_week_return = log_return(df_2y).rename('return')

df_factor_train = df_factors.merge(last_week_return, left_index=True, right_index=True, how='left')
df_factors_X = df_factor_train.loc[:, df_factor_train.columns!='return']
df_factors_y = df_factor_train['return']

###########
# OLS
###########

X_train = df_factors_X.to_numpy().astype(float)
y_train = df_factors_y.to_numpy().reshape(-1)

industry = df_nom['industry'].unique().tolist()
features = df_factors_X.columns.tolist()
non_industry_features = [f for f in features if f not in industry]

# Index of features not industry
model_idx_non_industry = np.array([i for i in range(len(features)) if features[i] in non_industry_features])
model_no_industry = sm.OLS(y_train, sm.add_constant(X_train[:, model_idx_non_industry])).fit()
sig_feature_idx_no_industry = print_coef_t_results(model_no_industry, critical_val=1.645, added_constant=True, suppress=True)

# Index of features of industry
model_idx_industry = np.array([i for i in range(len(features)) if features[i] not in non_industry_features])
model_w_industry = sm.OLS(y_train, X_train, hasconst=True).fit()
sig_feature_idx_w_industry = print_coef_t_results(model_w_industry, critical_val=1.645, added_constant=False, skip_i=model_idx_industry, suppress=True)

sig_feature_idx = np.array(list(set(sig_feature_idx_no_industry[1:]-1) | set(sig_feature_idx_w_industry)))
comb_idx = np.append(sig_feature_idx, model_idx_industry)
model_comb = sm.OLS(y_train, X_train[:, comb_idx], hasconst=True).fit()
pkl.dump(model_comb, open(f'./data/model_comb_{load_t}.pkl', 'wb'))
print(f'Saved model_comb_{load_t}.pkl')

df_X = df_factors_X.iloc[:, comb_idx]
df_y = df_factors_y
pkl.dump((df_X, df_y), open(f'./data/df_X__df_y_{load_t}.pkl', 'wb'))
print(f'Saved df_X__df_y_{load_t}.pkl')

###########
# Cos Sim
###########

X = X_train[:, comb_idx]
cos_sim_matrix = cosine_similarity(X, X)
pkl.dump(cos_sim_matrix, open(f'./data/cos_sim_matrix_{load_t}.pkl', 'wb'))
print(f'Saved cos_sim_matrix_{load_t}.pkl')