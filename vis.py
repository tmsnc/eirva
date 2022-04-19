import pandas as pd
import numpy as np
import pickle as pkl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Input, Output, dash_table
from data import log_return
import warnings
warnings.filterwarnings('ignore')

load_t = '20220318'
    
###########################
# Functions
###########################

def similar_stocks(cos_sim_matrix, df_X, df_info_raw, subject = '601318.SS'):

    row_index = np.argwhere(df_X.index == subject)[0][0]
    item_index = np.where(cos_sim_matrix[row_index, :]>0.65)
    df_cos_sim = pd.DataFrame({'Ticker': df_X.index[item_index],
                               'Name': df_info_raw.loc['shortName', df_X.index[item_index]],
                               'Industry': df_info_raw.loc['industry', df_X.index[item_index]],
                               'CosSim': cos_sim_matrix[row_index, item_index].reshape(-1)})
    
    df_cos_sim = df_cos_sim.sort_values('CosSim', ascending=False).reset_index().drop('index',axis=1).drop(0, axis=0)
    # > 0.65, max 10 rows
    if len(df_cos_sim) > 10:
        df_cos_sim = df_cos_sim.iloc[:10, :]

    return df_cos_sim
    
###########################
# Stock Price and Volatility
###########################

df_5y = pkl.load(open('./data/df_5y_{}.pkl'.format(load_t), 'rb'))
print(f'Loaded df_5y_{load_t}.pkl')

df_5y_ret = log_return(df_5y['Close'], fri_weekly=True)
df_5y_ret = df_5y_ret.drop(df_5y_ret.index[0], axis=0)

# rolling mean return and volatility
# 1y rolling mean weekly returns
df_weekly_return_rolling_1y_mean = df_5y_ret.rolling(52).mean()
df_weekly_return_rolling_1y_mean = df_weekly_return_rolling_1y_mean.loc[~df_weekly_return_rolling_1y_mean.isnull().any(axis=1), :]

# 1y rolling mean weekly vol annualized
df_weekly_return_rolling_1y_volann = df_5y_ret.rolling(52).std() * np.sqrt(52)
df_weekly_return_rolling_1y_volann = df_weekly_return_rolling_1y_volann.loc[~df_weekly_return_rolling_1y_volann.isnull().any(axis=1), :]

df_info_raw=pkl.load(open(f'./data/df_info_raw_{load_t}.pkl', 'rb'))
print(f'Loaded df_info_raw_{load_t}.pkl')

df_rolling_1y = pd.merge(df_weekly_return_rolling_1y_mean.iloc[-1,:].rename('mean_return'), df_weekly_return_rolling_1y_volann.iloc[-1,:].rename('ann_vol'), left_index=True, right_index=True, how='left')
df_rolling_1y = df_rolling_1y.merge(df_info_raw.loc['shortName'], how='left', left_index=True, right_index=True).reset_index()
df_rolling_1y['ticker_name'] = df_rolling_1y[['index', 'shortName']].astype(str).agg(' '.join, axis=1)

# 3m rolling mean weekly vol annualized
df_weekly_return_rolling_3m_volann = df_5y_ret.rolling(13).std() * np.sqrt(52)
df_weekly_return_rolling_3m_volann = df_weekly_return_rolling_3m_volann.loc[~df_weekly_return_rolling_3m_volann.isnull().any(axis=1), :]

############
# OLS model
############

# Load features matrix from Combination OLS regression
df_X, df_y = pkl.load(open(f'./data/df_X__df_y_{load_t}.pkl', 'rb'))
print(f'Loaded df_X__df_y_{load_t}.pkl')

model_comb = pkl.load(open(f'./data/model_comb_{load_t}.pkl', 'rb'))
print(f'Loaded model_comb_{load_t}.pkl')

df_feature_pct = (df_X.rank(pct=True).loc[:, df_X.columns[:(len(df_X.columns)-76)]])

##################
# Cosine Similarity
##################
cos_sim_matrix = pkl.load(open(f'./data/cos_sim_matrix_{load_t}.pkl', 'rb'))
print(f'Loaded cos_sim_matrix_{load_t}.pkl')

app = Dash(__name__)

###############
# MAIN LAYOUT # 
###############

app.layout = html.Div([
    html.Div(children=[
        
        html.H1('EIRVA'),
        html.H5('Equity Investment Recommendation and Visual Analysis'),
        html.P('This application is for traders to analyze stocks investment with interactive visualization.'),
        html.P('The following chart show the 1-year rolling weekly mean return and risk measured by standard deviation of returns.'),
        html.P('You may hover over the data to see the stock and values. Stocks with a higher return with lower amount of risk are preferred, and so the upper-left direction is the frontier of max return-over-risk possibilities.'),
        html.P('In this sample dataset, the universe is stocks in the CSI300 Index listed in Shanghai and Shenzhen, China.'),
        
        # CHART: Scatter Return/Risk
        html.Hr(), 
        dcc.Graph(id='graph-scatter'),

        # CONTROL
        # MULTI-SELECT
        html.H6('Stock Multi-Selection'),
        html.P('Please select the stock(s) you would like to highlight on the chart above. You may type the ticker or part of the name to find the stock. Within this list you will be able to perform additional analysis.'),
        html.P('Checkbox "Hide Unselected" will allow you to view only selected stocks and remove all others in the universe.'),
    
        dcc.Dropdown(df_rolling_1y.ticker_name,
                  [''],
                  multi=True, id='my-input'),
        dcc.Checklist([dict(label='Hide Unselected', value='hide_uns')], id='hide_uns'),
        html.Br(),
        
        # RADIO CHOOSE
        html.H6('Stock to Study'),
        html.P('Please choose one stock among selected to study.'),
        dcc.RadioItems(id='select_stock'),
        # html.P('Features Percentile (from low to high):'),
        
        dcc.Graph(id='graph_feature_pct'),
        
        html.Br(),
        html.P('Available studies:'),
        html.Ul([
            html.Li('Checkbox "Show History Path" will plot the historical path of its 1y rolling return & risk in the chart by color, up to 52 weeks.'),
            html.Li('Checkbox "Show Similar" will highlight other stocks that are similar to it with > 0.65 cosine similarity based on non-price features, max 10 stocks. \
            It is recommended to also "Hide Unselected" when using this function to reduce clutter.'),
            html.Li('View in a separate chart below of its stock price and volatility (standard deviation of reutrns) over time.'),
            # html.Li(''),
            ]
        ),
        html.P('Study controls:'),
        
        html.Table([
            html.Tr([
                html.Td([
                    dcc.Checklist([dict(label='Show History Path', value='show')],
                        id='show_hist_path')], style={'vertical-align': 'top'}),
                html.Td([
                    html.Label('Weeks History:')],
                    style={'vertical-align': 'top'}),
                html.Td([
                    dcc.Input(id='num_weeks', type='number', min=1, max=52, step=1, placeholder='10', size='2')],
                    style={'vertical-align': 'top'}),
                html.Td([
                    dcc.Checklist([dict(label='Show Similar', value='show_cluster')],
                        id='show_cluster')], style={'vertical-align': 'top'})  
            ])            
        ]),
        
        html.P('If "Show Similar" is checked, the below table will show stocks which have cosine similarity of higher than 0.65 versus \
        the selected stock, up to 10 stocks. An empty table means there are no stocks eligible that are similar enough at 0.65 threshold.'),
        
        dash_table.DataTable(columns=[{"name": i, "id": i} for i in ['Ticker','Name', 'Industry','CosSim']], data=[], id='cluster_table',
                             style_cell_conditional=[
                                 {
                                     'if': {'column_id': c},
                                     'textAlign':'left'
                                 } for c in ['Ticker','Name','Industry']
                             ],
                             style_cell={'padding': '5px'},
                             fill_width=False
                             # https://stackoverflow.com/questions/65625553/how-to-reduce-width-of-dash-datatable
                            ),
        # https://community.plotly.com/t/loading-pandas-dataframe-into-data-table-through-a-callback/19354
        
        # CHART: Selected Stock Price
        html.Hr(),
        dcc.Graph(id='stock_price')
        
        ], style=dict(padding= 10)
    )
], style={'display': 'flex', 'flex-direction': 'row'})

# multi select: update radio, update graph to highlight points
@app.callback(
    Output('select_stock', 'options'),
    Input(component_id='my-input', component_property='value'))
def multi_select_stocks(stocks):
    return [{'label': i, 'value': i} for i in stocks]

# radio: update graph to add historical path
@app.callback(
    Output('select_stock', 'value'),
    Input('select_stock', 'options'))
def select_stock(sel_stock_options):
    if len(sel_stock_options)==0:
        return ''
    else:
        return sel_stock_options[0]['value']

####################
# Feature Percentile
####################

@app.callback(
    Output('graph_feature_pct', 'figure'),
    Input('select_stock', 'value'))
def fig_feature_pct(sel_stock):
    fig = px.scatter(None)
    fig=go.Figure(fig)
    selected_stock_ticker=''
    if (sel_stock!='') and (sel_stock!=None):
        selected_stock_ticker = sel_stock.split(' ')[0]
        
        fig.add_traces(go.Scatter(
                         y=df_feature_pct.loc[selected_stock_ticker, :].rename('pct').reset_index()['index'], 
                         x=df_feature_pct.loc[selected_stock_ticker, :].rename('pct').reset_index()['pct'],
                         text=df_feature_pct.loc[selected_stock_ticker, :].rename('pct').reset_index()['pct'].map(lambda x:'{:.0%}'.format(x)),
                         mode='markers+text',
                         textfont_size=9,
                         textposition='top center',
                         cliponaxis=False, # https://community.plotly.com/t/need-help-on-python-dash-bar-text-label-is-cut-off-from-the-graph/14665
                         marker=dict(
                                    color='#851e3e',
                                    symbol='diamond',
                                    size=5
                                ),
                         ))
        # fig.update_traces(
        #                  )
    
    fig.update_yaxes(title='Feature', #tickformat='.1%',
                     showline=False, linecolor='black',
                     showgrid=True, gridcolor='#6497b1', gridwidth=1, 
                     zeroline=False, zerolinecolor='#dfe3ee', autorange=True) # y=0 line
    fig.update_xaxes(title='Percentile', tickformat='.0%',
                     showline=False, linecolor='black', # axis line
                     showgrid=False, gridwidth=0.2, gridcolor='#f7f7f7', autorange=True, # grid lines
                     range=(0,1),
                     rangemode='tozero'
                    )
    fig.update_layout(title=f'{selected_stock_ticker} Feature Percentile (from low to high)', transition_duration=500, 
                      paper_bgcolor='white',
                      plot_bgcolor='white',
                      width=500,
                      height=300,
                      margin_pad=10,
                      # margin_t=100
                     )    
    return fig

#####################
# CHART Return/Risk # 
#####################

@app.callback(
    Output(component_id='graph-scatter', component_property='figure'),
    Input(component_id='my-input', component_property='value'),
    Input(component_id='select_stock', component_property='value'),
    Input('show_hist_path', 'value'),
    Input('num_weeks','value'),
    Input('hide_uns','value'),
    Input('show_cluster','value')
)
def update_figure(stocks, sel_stock, show, n_weeks, hide_uns, show_cluster):
    
    if n_weeks==None:
        n_weeks=10
        
    # Base return / risk plot
    # -----------------------
        

    if show_cluster==['show_cluster'] and (sel_stock!='') and (sel_stock!=None):
        
        selected_stock_ticker = sel_stock.split(' ')[0]
        df_cos_sim = similar_stocks(cos_sim_matrix, df_X, df_info_raw, subject=selected_stock_ticker)
        cluster_list = df_cos_sim['Ticker'].tolist()
        

        if hide_uns:
            fig = px.scatter(None)
        else:
            fig = px.scatter(df_rolling_1y.loc[~(df_rolling_1y.ticker_name.isin(stocks) | df_rolling_1y['index'].isin(cluster_list))], 
                             y="mean_return", x="ann_vol", hover_name="ticker_name", 
                             log_x=False,
                             width=1000,
                             height=600
                            )
    else:
        if hide_uns:
            fig = px.scatter(None)
        else:
            fig = px.scatter(df_rolling_1y.loc[~df_rolling_1y.ticker_name.isin(stocks)], 
                         y="mean_return", x="ann_vol", hover_name="ticker_name", 
                         log_x=False,
                         width=1000,
                         height=600
                        )
        

    fig = go.Figure(fig)
    fig.update_traces(hovertemplate="<b>%{hovertext}</b><br><br>mean_return=%{y:.2%}<br>ann_vol=%{x:.2%}<extra></extra>",
                      marker=dict(
                          color='#005b96',
                          size=4
                      )
                     )

    # Multi-select points highlight 
    # -----------------------------
    fig.add_trace(go.Scatter(
                                y=df_rolling_1y.loc[df_rolling_1y.ticker_name.isin(stocks),'mean_return'],
                                x=df_rolling_1y.loc[df_rolling_1y.ticker_name.isin(stocks),'ann_vol'],
                                name='',
                                showlegend = False,
                                hovertemplate= "<b>%{hovertext}</b><br><br>mean_return=%{y:.1%}<br>ann_vol=%{x:.1%}<extra></extra>",
                                hovertext=df_rolling_1y.loc[df_rolling_1y.ticker_name.isin(stocks),'ticker_name'],
                                texttemplate="<b>%{hovertext}</b>",
                                textposition='bottom right',
                                textfont= {'color': '#8b9dc3'},
                                marker=dict(
                                    color='red',
                                    symbol='diamond',
                                    size=5
                                ),
                                mode='markers+text'
                               )
                          )               
    
    
    if (sel_stock=='') or (sel_stock==None):
        pass
    else:
        selected_stock_ticker=sel_stock.split(' ')[0]
        if show==['show']:
        
            # Historical path
            # ---------------
            
            fig.add_trace(dict(
                                        mode="markers",
                                        y= list(df_weekly_return_rolling_1y_mean.loc[:, selected_stock_ticker])[-n_weeks:],
                                        x= list(df_weekly_return_rolling_1y_volann.loc[:, selected_stock_ticker])[-n_weeks:],
                                        text= list(df_weekly_return_rolling_1y_mean.index)[-n_weeks:],
                                        name=selected_stock_ticker,
                                        showlegend=False,
                                        hovertemplate= "<b>" + selected_stock_ticker+ "<br>%{hovertext}</b><br><br>mean_return=%{y:.2%}<br>ann_vol=%{x:.2%}<extra></extra>",
                                        hovertext=list(df_weekly_return_rolling_1y_mean.index)[-n_weeks:],
                                        line=dict(
                                            color='red',
                                            dash='solid',
                                        ),
                                        marker=dict(
                                            symbol='diamond',
                                            color=np.linspace(0,1,n_weeks),
                                            colorscale='Agsunset_r',
                                            size=5
                                        ),
                                        textposition='bottom right',

                                   )
                              )
        
        
        if show_cluster==['show_cluster']:
            
            # cluster_list defined in main trace above            
            fig.add_trace(go.Scatter(
                                y=df_rolling_1y.loc[df_rolling_1y['index'].isin(cluster_list),'mean_return'],
                                x=df_rolling_1y.loc[df_rolling_1y['index'].isin(cluster_list),'ann_vol'],
                                name='',
                                showlegend = False,
                                hovertemplate= "<b>%{hovertext}</b><br><br>mean_return=%{y:.1%}<br>ann_vol=%{x:.1%}<extra></extra>",
                                hovertext=df_rolling_1y.loc[df_rolling_1y['index'].isin(cluster_list),'ticker_name'],
                                texttemplate="<b>%{hovertext}</b>",
                                textposition='bottom right',
                                textfont= {'color': '#8b9dc3'},
                                marker=dict(
                                    color='#2ab7ca',
                                    symbol='hexagram',
                                    size=5
                                ),
                                mode='markers+text'
                               )
                          )               
        
            
    
    fig.update_yaxes(title='Return (Mean)', tickformat='.1%',
                     showline=True, linecolor='black',
                     showgrid=True, gridcolor='#f7f7f7', gridwidth=0.2,
                     zeroline=True, zerolinecolor='#dfe3ee', autorange=True) # y=0 line
    fig.update_xaxes(title='Risk (Stdev)', tickformat='.0%',
                     showline=True, linecolor='black', # axis line
                     showgrid=True, gridwidth=0.2, gridcolor='#f7f7f7', autorange=True# grid lines
                    )
    fig.update_layout(title='1y Rolling Weekly Return / Risk', transition_duration=500, 
                      paper_bgcolor='white',
                      plot_bgcolor='white',
                      width=1000
                     )     
    return fig


# Update Cos Sim data table (called cluster since was using Kmeans previously)
@app.callback(
    Output('cluster_table', 'data'),
    Input('show_cluster', 'value'),
    Input('select_stock','value'))
def update_cluster_table(show_cluster, sel_stock):
    if show_cluster==['show_cluster'] and (sel_stock!='') and (sel_stock!=None):
        selected_stock_ticker = sel_stock.split(' ')[0]
        df_cos_sim = similar_stocks(cos_sim_matrix, df_X, df_info_raw, subject=selected_stock_ticker)
        df_cos_sim['CosSim'] = df_cos_sim['CosSim'].map(lambda x: '{:.4f}'.format(x))
        data = df_cos_sim.to_dict('records')
        return data
    else:
        return None
    

###########################
# CHART Price, Volatility # 
###########################
@app.callback(
    Output('stock_price', 'figure'),
    Input('select_stock', 'value'))
def update_stock_price_fig(sel_stock):
    if (sel_stock=='') or (sel_stock==None):
        df_empty = pd.DataFrame(np.zeros((df_5y.shape[0], 1)), index= df_5y.index).reset_index().rename({0: 'Price'}, axis=1)
        fig_px=px.scatter(df_empty, x='Date', y='Price', log_x=False)
        fig_px.update_yaxes(title='Price',
                         showline=True, linecolor='black',
                         showgrid=False, gridcolor='#f7f7f7', gridwidth=0.2,
                         zeroline=True, zerolinecolor='black') # y=0 line
        fig_px.update_xaxes(
                        title='Date', tickformat='%d-%b-%Y', ticklen=3,
                         showline=True, linecolor='black', # axis line
                         showgrid=False, gridwidth=0.2, gridcolor='#f7f7f7',
                            rangeslider_thickness = 0.1)
        fig_px.update_layout(title='Stock Price', transition_duration=500, 
                             paper_bgcolor='white',
                             plot_bgcolor='white')
        return fig_px
    else:
        
    # Price chart
        stk=sel_stock.split(' ')[0]
        fig_px = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing = 0.3,
                              subplot_titles=['Price','3M Volatility'],
                              row_heights=[800, 200],
                              column_widths=[1000])
        
        fig_px.add_trace(go.Candlestick(x=df_5y.index,
                                        open=df_5y[('Open',stk)],
                                        high=df_5y[('High',stk)],
                                        low=df_5y[('Low',stk)],
                                        close=df_5y[('Close',stk)], 
                                        xaxis='x', yaxis='y', visible=True, name='Price', 
                                        ),
                           row=1, col=1)
    # Add range slider
        fig_px.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1m",
                             step="month",
                             stepmode="backward"),
                        dict(count=6,
                             label="6m",
                             step="month",
                             stepmode="backward"),
                        dict(count=1,
                             label="YTD",
                             step="year",
                             stepmode="todate"),
                        dict(count=1,
                             label="1y",
                             step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ][::-1]) #reverse
                ),
                rangeslider=dict(
                    autorange=True,
                    visible=True,
                    range= [df_5y.index.min(), df_5y.index.max()]
                ),
                type="date"
            ),
            yaxis=dict(
                anchor='x',
                autorange=True,
                side='left',
                title='Price'
                ),
            yaxis2=dict(
                anchor='x',
                autorange=True,
                side='left',
                title='Volatility',
                tickformat='.0%'),
        )        

        fig_px.update_xaxes(row=1,col=1,
                        title='', tickformat='%d-%b-%Y', ticklen=3, showticklabels=True,
                         showline=True, linecolor='black', # axis line
                         showgrid=False, gridwidth=0.2, gridcolor='#f7f7f7',
                        rangeslider_thickness = 0.1)
        
        fig_px.update_yaxes(row=1,col=1,
                         showline=True, linecolor='black',
                         showgrid=True, gridcolor='#f7f7f7', gridwidth=0.2,
                         zeroline=True, zerolinecolor='black') # y=0 line
    
    # Volatility Chart
        fig_px.add_trace(go.Scatter(x=df_weekly_return_rolling_3m_volann.index,
                                    y=df_weekly_return_rolling_3m_volann[stk],
                                    yaxis='y2',
                                    name='Volatility',
                                    marker=dict(color='#2ab7ca')
                                    ),
                        row=2,col=1
                        )        
        

        fig_px.update_layout(title=f'{sel_stock}', transition_duration=500, 
                            paper_bgcolor='white',
                            plot_bgcolor='white',
                             hovermode="x unified",
                             showlegend=False,
                             width=1000,
                             height=1000
                            )

        fig_px.update_xaxes(row=2,col=1,
                            title='', tickformat='%d-%b-%Y', ticks='outside', tickson='boundaries', ticklen=3,
                         showline=True, linecolor='black', # axis line
                         showgrid=False, gridwidth=0.3, gridcolor='#f7f7f7') # grid lines

        fig_px.update_yaxes(row=2,col=1,
                         showline=True, linecolor='black',
                         showgrid=True, gridcolor='#f7f7f7', gridwidth=0.2,
                         zeroline=True, zerolinecolor='black') # y=0 line
        
        return fig_px
    

if __name__ == '__main__':
    # For Web Browser: the port can be changed
    # Go to http://127.0.0.1:8050 in browser
    app.run_server(debug=False, host='127.0.0.1', port=8050)
