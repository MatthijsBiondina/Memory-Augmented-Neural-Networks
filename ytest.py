from data.stock_dataset import StockDataset

dataset = StockDataset()

# import warnings
# import pandas as pd
# from matplotlib import pyplot as plt
# from pandas import DataFrame
# from pandas.plotting import autocorrelation_plot
# from sklearn.metrics import mean_squared_error
# from statsmodels.tsa.arima_model import ARIMA
# from tqdm import tqdm
#
# from utils.tools import listdir, poem
#
# warnings.filterwarnings('ignore')
#
# df = pd.read_csv(listdir('./res/in/stocks')[0],
#                  header=0,
#                  parse_dates=[0],
#                  index_col=0,
#                  squeeze=True,
#                  date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%d"))['Adj Close']
# print(df.head())
# df.plot()
# plt.title("Adj Close")
# plt.show()
#
# autocorrelation_plot(df)
# plt.title("Autocorrelation")
# plt.show()
#
# model = ARIMA(df, order=(10,1,0))
# model_fit = model.fit(disp=0)
# print(model_fit.summary())
#
# residuals = DataFrame(model_fit.resid)
# residuals.plot()
# plt.title("Residuals")
# plt.show()
# residuals.plot(kind='kde')
# plt.title("Residuals KDE")
# plt.show()
# print(residuals.describe())
#
# X = df.values
# train, test = X[:-35], X[-35:]
# history = [x for x in train]
# predictions=[]
# for t in tqdm(range(len(test)), desc=poem("rolling forecast"), leave=False):
#     model = ARIMA(history, order=(10,1,0))
#     model_fit = model.fit(disp=0)
#     output = model_fit.forecast()
#     yhat=output[0]
#     predictions.append(yhat)
#     obs=test[t]
#     history.append(obs)
# error = mean_squared_error(test,predictions)
# print(f"Test MSE: {error:.3f}")
# plt.plot(test)
# plt.plot(predictions, color='red')
# plt.title("predictions")
# plt.show()