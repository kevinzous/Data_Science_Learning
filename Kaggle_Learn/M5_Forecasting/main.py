# data manipulation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 500)
import datetime
# ploting
from matplotlib import style
style.use('ggplot')
#import seaborn as sns
import plotly.express as px

# modeling
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.tsa.arima_process as sta
import statsmodels.graphics.tsaplots as sgt
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.tsa.statespace as sts

import warnings
warnings.filterwarnings(action='once')
import sys

# data reading
sales_train_validation = pd.read_csv('data/sales_train_validation.csv')
calendar = pd.read_csv('data/calendar.csv',parse_dates=[0])
sell_prices = pd.read_csv('data/sell_prices.csv')

#reduce memory usage
sales_train_validation = sales_train_validation.sample(frac=0.2)
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

sales_train_validation = reduce_mem_usage(sales_train_validation)
calendar = reduce_mem_usage(calendar)
sell_prices = reduce_mem_usage(sell_prices)


# missing values examination
print("sales_train_validation nan values: ",sales_train_validation.isna().sum().sum())
print("sell_prices nan values: ", sell_prices.isna().sum().sum())
print("calendar nan values: ", calendar.isna().sum().sum())
print() #line break

print(calendar.isna().sum())
print() #line break

# data sparcity check
percentage_zero=(sales_train_validation==0).sum(axis=1)/1913 ## 1913 is the number of days
print(percentage_zero.describe())

# unpivoting the columns
value_vars=sales_train_validation.columns.to_list()[6:]  # remove the first 6 variables ie 'id','item_id', 'dept_id', 'cat_id', 'store_id','state_id'
id_vars=['id','item_id', 'dept_id', 'cat_id', 'store_id','state_id']
sales = pd.melt(sales_train_validation,id_vars=id_vars, value_vars=value_vars,var_name='d', value_name='sales_count')
# joining with calendar but memomry run out with this method
sales=pd.merge(sales,calendar,left_on='d',right_on='d',how="left")

# plot
plotted_sales=sales[sales['item_id'].isin([sales.item_id.values[:2]])]
fig = px.line(plotted_sales, x="date", y="sales_count",# color="continent", line_group="country", hover_name="country",
        line_shape="spline", render_mode="svg",color="id")
fig.show()

sales[sales['item_id'].isin([sales.item_id.values[:2]])]