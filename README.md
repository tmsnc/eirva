# EIRVA: Equity Investment Recommendation and Visualization Analysis.
A detailed report blog can be viewed at:
https://medium.com/@tschoi/eirva-ba05362a11bb

EIRVA is a Python tool for stock investors to analyze and visualize returns and risk of a stock universe for investment selection.
The features are:
1. Risk Factor decomposition of an input portfolio of stocks: for better understanding of factor exposures of a portfolio
2. Clustering of stocks in the portfolio by unsupervised learning: to understand if there might be themes or similarities among these stocks that explains such clustering
3. Recommendation of other stocks not in the portfolio that have similar characteristics as those inside: to identify potential substitute investments that have higher risk-return.
4. Visualization of data to explain the analysis performed, and implement interactivity as a way to allow user to adjust parameters of analysis: this will enable user to perform variations of analysis without changing code.

**Table of Contents**

To follow along the report blog, details of the code implementation can be read from the Jupyter notebooks, recommended in Jupyter Lab environment:
* _data.ipynb_: Data Cleaning and Analysis
* _vis.ipynb_: Plotly Dash app for web-hosted interactive visualization

To run the application directly, there are 3 components:
* _download_data.py_: this can be scheduled daily to download the necessary data from Yahoo Finance into a pickle file for further processing.
* _data.py_: this will take the pickled Yahoo Finance data and perform all data preparation and processing steps and expore pickle files ready for visualization.
* _vis.py_: this runs the Plotly Dash flask web server to host the interactive application that can be viewed in browser.

Sample data is contained in folder
_/data_

**Quick start**

Run _data.py_ and then _vis.py_
Then browse to http://127.0.0.1:8050 in browser to view application

**Requirements** 

dash==2.3.1\
numpy==1.21.5\
pandas==1.4.0\
plotly==4.14.3\
scikit_learn==1.0.2\
statsmodels==0.13.2\
yfinance==0.1.63
