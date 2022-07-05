# Univariate time series application: Stock price prediction
We built a simple stock-price predictor using LSTM and GRU neural networks. The data used is obtained from the stock price history, using the publicly available dataset. A detailed example is provided on one sample stock. 

The core functions used are stored in ```core``` folder. The ```Data``` folder contains the datasets of the detailed example, as well as the other samples on which experiments were conducted. The folder ```trained_models``` contains the trained weights for both LSTM and GRU networks. They are appropriately named, along with the respective stock name tags. 

## 1. Statistical Distance Dissimilarity (SDD)
The SDD-vs-performance data is obtained from the an intermediate step of StaDRo computation. The obtained values are stored in csv format in ```SDD>data``` folder. The notebook ```Performance_SDD_visualization_updated.ipynb``` visualizes the SDD vs performance behavior. 

## 2. StaDRo
A detailed example of StaDRo is provided in the notebook ```Reliance_CurveFit_Robustness_Example.ipynb```. The notebook ```StaDRo_other_datasets_results.ipynb``` displays the results of StaDRo on other stocks. 

## 3. StaDRe
A detailed example of StaDRe is provided in the notebook ```Reliance_StaDRe_Example.ipynb```. The notebook ```ReliabilityEstimation_visualization_updated.ipynb``` displays the results of StaDRe on other stocks. 