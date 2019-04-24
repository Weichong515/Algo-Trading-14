# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 03:05:40 2019

@author: 51667
"""
import csv
import datetime
import os
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# Dict_=np.load('data.npy').item()
COMMISSION = 0
TAX = 0
SNP = pd.read_csv("^GSPC.csv")
SNP.set_index(pd.to_datetime(SNP.Date).map(lambda x: x.date()), inplace=True)

date_list = [
    '20181001', '20181002', '20181003', '20181004', '20181005', '20181008',
    '20181009', '20181010', '20181011', '20181012', '20181015', '20181016',
    '20181017', '20181018', '20181019', '20181022', '20181023', '20181024',
    '20181025', '20181026', '20181029', '20181030', '20181031', '20181101',
    '20181102', '20181105', '20181106', '20181107', '20181108', '20181109',
    '20181112', '20181113', '20181114', '20181115', '20181116', '20181119',
    '20181120', '20181121', '20181123', '20181126', '20181127', '20181128',
    '20181129', '20181130', '20181203', '20181204', '20181206', '20181207',
    '20181210', '20181211', '20181212', '20181213', '20181214', '20181217',
    '20181218', '20181219', '20181220', '20181221', '20181224', '20181226',
    '20181227', '20181228', '20181231'
]
rebalance_date = ['20181016', '20181101', '20181116', '20181203', '20181217',
                  '20181231']

rebalance_formal_date = [datetime.datetime.strptime(date, "%Y%m%d").date() \
                         for date in rebalance_date]

ALL_STOCK_COLUMN = ["date", "time", "O", "H", "L", "C", "V",
                    "splits", "earnings", "dividends"]

portfolio_file = [
    "benchmark model.txt",
    'top5_nn_model_combined.txt',
    'top5_nn_model_combined_ls.txt',
    'top10_nn_model_combined.txt',
    'top10_nn_model_combined_ls.txt',
    'top10_nn_model1.txt',
    'top10_nn_model1_ls.txt',
    'top10_nn_model2.txt',
    'top10_nn_model2_ls.txt',
    'top20_nn_model1.txt',
    'top20_nn_model1_ls.txt',
    'top20_nn_model2.txt',
    'top20_nn_model2_ls.txt'
]
# file_name = "2018-11-01.csv"

SNP["Open_std"] = SNP["Open"] / SNP["Open"][rebalance_formal_date[0]]

class action:
    symbol = ""
    datetime = datetime.datetime.now()
    pos = 0
    patience = 0
    
class trade:
    symbol = ""
    decision_time = None
    ask0 = 0.0
    bid0 = 0.0  # bid, ask price at order release time
    execution_time = None
    ask = 0.0
    bid = 0.0  # bid, ask price at order filled time
    price = 0.0
    volume = 0
    
def split_quote_data(file_name, output_path=os.getcwd()):
    """
    The function is to split the whole quote data csv into csv with respect to 
    each stock.
    Be careful that the input file_name is used to name the split files.
    """
    with open(file_name) as csvfile:
        file_ = file_name.split(".csv")[0]
        reader = csv.reader(csvfile, delimiter=',')
        i = 0
        old_symbol = "A"
        data = []
        for line in reader:
            if i == 0:
                column = line  # keep the column name
                for i, col in enumerate(column):
                    if col == "SYM_ROOT":
                        SYM_ROOT = i
            else:
                symbol = line[SYM_ROOT]
                if symbol != old_symbol:
                    output = pd.DataFrame(data, columns=column)
                    if output_path:
                        output.to_csv("\\".join([output_path, f"{file_}_{old_symbol}.csv"]),
                                      index=False)
                    else:
                        output.to_csv(f"{file_}_{old_symbol}.csv",
                                      index=False)
                    data = []
                    old_symbol = symbol
                else:
                    data.append(line)
            i += 1
        output = pd.DataFrame(data, columns=column)
        if output_path:
            output.to_csv("\\".join([output_path, f"{file_}_{old_symbol}.csv"]),
                          index=False)
        else:
            output.to_csv(f"{file_}_{old_symbol}.csv",
                          index=False)


"""
trade1 = {
    "symbol": "AAPL",
    "volume": 1,
    "trade_price": 200,
    "trade_time": datetime.datetime(2018,10,11,9,45),
    "decision_time": datetime.datetime(2018,10,11,9,39),
    "decision_price": 201,
}
trade2 = {
    "symbol": "AAPL",
    "volume": -1,
    "trade_price": 200,
    "trade_time": datetime.datetime(2018,11,11,11,45),
    "decision_time": datetime.datetime(2018,11,11,11,30),
    "decision_price": 201,
}
trades = [trade1,trade2]
"""


def read_portfolio(file):
    with open(file) as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        i = 0
        if "ls" in file:
            short = True
        else:
            short = False
        if not short:
            portfolio = []
            for line in reader:
                if i % 2:
                    portfolio_day = line
                    portfolio.append(portfolio_day)
                i += 1
            return {"long": portfolio}
        else:
            long_portfolio = []
            short_portfolio = []
            for line in reader:
                if i % 3 == 1:
                    portfolio_day = line
                    line[0] = line[0].split("Long:")[-1]
                    long_portfolio.append(portfolio_day)
                elif i % 3 == 2:
                    portfolio_day = line
                    line[0] = line[0].split("Short:")[-1]
                    short_portfolio.append(portfolio_day)
                i += 1
            return {
                "long": long_portfolio,
                "short": short_portfolio
            }


def naive_trade(portfolio):
    # rebalance at openning, assuming we can trade all volume at open price
    capital = 1000000
    trade_list = []
    old_portfolio = set(portfolio["long"][0])
    if "short" in portfolio:
        old_portfolio.union(set(portfolio["short"][0]))
    old_allocation = {stock: 0 for stock in old_portfolio}
    total_rebalance_day = len(rebalance_date)
    pos_data = {}
    side = len(portfolio)
    for i, date in enumerate(rebalance_date):
        if i <= total_rebalance_day - 2:  # if it is not the last trading day.
            p = portfolio["long"][i]
            new_allocation = {stock: capital / (len(p)) / side for stock in p}
            if "short" in portfolio:
                for stock in portfolio["short"][i]:
                    new_allocation[stock] = -capital / (len(p)) / side
        else:
            new_portfolio = portfolio["long"][-1]
            if "short" in portfolio:
                new_portfolio.extend(portfolio["short"][-1])
            new_allocation = {stock: 0 for stock in new_portfolio}
        new_portfolio = set(new_allocation.keys())
        change_portfolio = old_portfolio.union(new_portfolio)
        for stock in change_portfolio:
            root_path = os.getcwd()
            file_path_suffix = f"allstocks_{date}"
            csv_name = f"table_{stock.lower()}.csv"
            total_path = "\\".join([root_path, file_path_suffix, csv_name])
            backtest_data = pd.read_csv(total_path, header=None)
            backtest_data.columns = ALL_STOCK_COLUMN
            backtest_data["datetime"] = \
                pd.to_datetime(backtest_data.date.map(str) + " " + backtest_data.time.map(str),
                               format='%Y%m%d %H%M')
            backtest_data.set_index("datetime", inplace=True)

            for dt, row in backtest_data.iterrows():
                if dt.time() >= datetime.time(9, 30):
                    open_dt = dt
                    open_data = row
                    break
            pos_change = int(new_allocation.get(stock, 0) / open_data["O"]) \
                         - pos_data.get(stock.upper(), 0)
            if pos_change:
                trade = {
                    "symbol": stock.upper(),
                    "volume": pos_change,
                    "trade_price": open_data["O"],
                    "trade_time": open_dt,
                    "decision_time": open_dt,
                    "decision_price": open_data["O"]
                }
                if stock.upper() in pos_data:
                    pos_data[stock.upper()] += pos_change
                else:
                    pos_data[stock.upper()] = pos_change
                trade_list.append(trade)
        old_portfolio = new_portfolio
    trade_list = sorted(trade_list, key=lambda x: x["symbol"])
    return trade_list


def calculate_pos(trades):
    # calculate position data w.r.t. only one symbol
    position = [{
        "datetime": datetime.datetime(2018, 10, 1, 9, 30),
        "pos": 0,
        "cash": 0,
    }]
    i = 0
    for trade in trades:
        old_pos = position[i]
        new_pos = {
            "datetime": trade["trade_time"],
            "pos": old_pos["pos"] + trade["volume"],
            "cash": old_pos["cash"] - trade["volume"] * trade["trade_price"] \
                    - abs(trade["volume"]) * trade["trade_price"] * COMMISSION
        }
        position.append(new_pos)
        i += 1
    symbol = trade["symbol"].upper()
    position[0]["symbol"] = symbol
    return position


def calculate_pnl(position):
    # to convert position data into pnl data
    symbol = position[0]["symbol"]
    i = 0
    pnl_list = []
    max_length = len(position)
    for date in date_list:
        root_path = os.getcwd()
        file_path_suffix = f"allstocks_{date}"
        csv_name = f"table_{symbol.lower()}.csv"
        total_path = "\\".join([root_path, file_path_suffix, csv_name])
        backtest_data = pd.read_csv(total_path, header=None)
        backtest_data.columns = ALL_STOCK_COLUMN
        backtest_data["datetime"] = \
            pd.to_datetime(backtest_data.date.map(str) + " " + backtest_data.time.map(str),
                           format='%Y%m%d %H%M')
        backtest_data.set_index("datetime", inplace=True)

        open_dt = datetime.datetime.strptime(date + " 930", '%Y%m%d %H%M')
        open_data = backtest_data.loc[open_dt]

        dividend = open_data["dividends"]
        for dt, row in backtest_data.iterrows():
            if dt.time() < datetime.time(9, 30):
                pass
            elif dt.time() > datetime.time(16, 0):
                pass
            else:
                temp_pos = position[i]
                if dt == open_dt:  # calculate dividend
                    pnl_minute = {"dt": dt,
                                  "pos": temp_pos["pos"],
                                  "dividend": dividend * temp_pos["pos"],
                                  "pnl": temp_pos["cash"] + temp_pos["pos"] * row["O"]
                                  }
                else:
                    pnl_minute = {"dt": dt,
                                  "pos": temp_pos["pos"],
                                  "dividend": 0,
                                  "pnl": temp_pos["cash"] + temp_pos["pos"] * row["O"]
                                  }
                pnl_list.append(pnl_minute)
                if i < max_length - 1 and dt >= position[i + 1]["datetime"]:
                    i += 1
    pnl = pd.DataFrame(pnl_list)
    pnl.set_index("dt", inplace=True)
    pnl["cum_dividend"] = pnl.dividend.cumsum()
    pnl[symbol] = pnl["cum_dividend"] + pnl["pnl"]
    return pnl[symbol]


def calculate_daily_pnl(position, capital=0):
    # to convert position data into daily pnl data
    symbol = position[0]["symbol"]
    i = 0
    pnl_list = []
    max_length = len(position)
    for date in date_list:
        root_path = os.getcwd()
        file_path_suffix = f"allstocks_{date}"
        csv_name = f"table_{symbol.lower()}.csv"
        total_path = "\\".join([root_path, file_path_suffix, csv_name])
        backtest_data = pd.read_csv(total_path, header=None)
        backtest_data.columns = ALL_STOCK_COLUMN
        backtest_data["datetime"] = \
            pd.to_datetime(backtest_data.date.map(str) + " " + backtest_data.time.map(str),
                           format='%Y%m%d %H%M')
        backtest_data.set_index("datetime", inplace=True)

        for dt, row in backtest_data.iterrows():
            if dt.time() >= datetime.time(9, 30):
                open_dt = dt
                open_data = row
                break

        dividend = open_data["dividends"]
        temp_pos = position[i]
        pnl_daily = {"dt": open_dt,
                     "pos": temp_pos["pos"],
                     "dividend": dividend * temp_pos["pos"],
                     "pnl": temp_pos["cash"] + temp_pos["pos"] * open_data["O"]
                     }
        pnl_list.append(pnl_daily)
        if i < max_length - 1 and open_dt >= position[i + 1]["datetime"]:
            i += 1
    pnl = pd.DataFrame(pnl_list)
    pnl["date"] = pnl.dt.map(lambda x: x.date())
    pnl.set_index("date", inplace=True)
    pnl["cum_dividend"] = pnl.dividend.cumsum()
    pnl[symbol] = pnl["cum_dividend"] + pnl["pnl"]
    pnl["capital"] = pnl[symbol] + capital
    return pnl[symbol]


def split_trade(trade_list):
    trade_split = []
    old_symbol = trade_list[0]["symbol"]
    sub_list = []
    for trade in trade_list:
        if trade["symbol"] == old_symbol:
            sub_list.append(trade)
        else:
            trade_split.append(sub_list)
            sub_list = [trade]
            old_symbol = trade["symbol"]
    return trade_split


def draw_individual_stock(trade_stock, model_name):
    symbol = trade_stock[0]["symbol"]
    trade = trade_stock[0]
    trade_date = trade["trade_time"].date()
    if trade["volume"] > 0:
        direction = "LONG"
    else:
        direction = "SHORT"
    inspect_window_start = trade_date - datetime.timedelta(days=15)
    inspect_window_end = trade_date + datetime.timedelta(days=15)
    daily_data = Dict_[symbol.lower()]
    daily_data.set_index(pd.to_datetime(daily_data["Date"]).map(lambda x: x.date()), inplace=True)
    open_price = daily_data["Open"][inspect_window_start:inspect_window_end]

    plt.figure(figsize=(13, 6))
    plt.scatter(trade_date, open_price[trade_date], s=50, color="r")
    plt.ylabel("Open")
    title = f"{model_name}_{trade_date.isoformat()}_{symbol.upper()}__{direction}"
    plt.title(title)
    open_price.plot()
    plt.savefig(f'./figs/{model_name}/{title}.png')


backtest_result = []
if "figs" not in os.listdir():
    os.mkdir("figs")
for model_file in portfolio_file:
    # model_file = portfolio_file[i]
    portfolio = read_portfolio(model_file)
    model_name = model_file.split(".txt")[0]
    trade_list = naive_trade(portfolio)
    trade_list_time1 = sorted(trade_list, key=lambda x: x["trade_time"])
    trade_split = split_trade(trade_list)
    
    total_position = [calculate_pos(trade_stock) for trade_stock in trade_split]
    total_pnl = [calculate_daily_pnl(position) for position in total_position]

    all_pnl = pd.DataFrame(total_pnl).T
    sum_pnl = pd.DataFrame(total_pnl).T.sum(axis=1)
    sum_pnl_trading = sum_pnl[sum_pnl.index >= rebalance_formal_date[0]]

    backtest_result.append(all_pnl)
    """
    plt.figure(figsize=(13,6))
    plt.scatter(rebalance_formal_date, sum_pnl[rebalance_formal_date], s=50,color="r")
    sum_pnl_trading.plot()
    plt.ylabel("profit")
    title = f"{model_name}_PNL"
    plt.title(title)
    
    if model_name not in os.listdir("./figs"):
        os.mkdir(f"./figs/{model_name}")
    plt.savefig(f'./figs/{model_name}/{title}.png')
    
    for trade_stock in trade_split:
        draw_individual_stock(trade_stock, model_name)   
    """

def implementation_shortfall(action):
    cwd = os.getcwd()

    symbol = action.symbol
    dt = action.datetime
    date = dt.date()
    pos = action.pos
    patience = action.patience
    decision_price = None
    """
    col_name = ["date", "time", "O", "H", "L", "C", "V","unknown1", "unknown2", "unknown3"]
    trade_file_path = cwd+"\\"+f"allstocks_{date}"+"\\"+f"table_{symbol.lower()}.csv"
    trade_data = pd.read_csv(trade_file_path, header = None)
    trade_data.columns = col_name
    """
    quote_data = pd.read_csv(f"{date}_quote_{symbol.upper()}.csv")
    quote_data["ISOFORMAT"] = quote_data.DATE.map(str) + " " + quote_data.TIME_M
    quote_data["DATETIME"] = pd.to_datetime(quote_data["ISOFORMAT"])
    remaining_pos = pos
    trade_list = []

    for i, row in quote_data.iterrows():
        if row["time"] >= dt:
            if not decision_price:
                decision_price = (row["BID"] + row["ASK"]) / 2
            if patience == 0:
                if pos > 0:
                    trade_price = row["ASK"]
                    trade_volume = min(pos, row["ASKSIZ"])
                    remaining_pos -= trade_volume
                    trade_dt = row["DATETIME"]
                    break
                elif pos < 0:
                    trade_price = row["BID"]
                    trade_volume = min(-pos, row["ASKSIZ"])
                    remaining_pos += trade_volume
                    trade_dt = row["DATETIME"]
                    break
            else:
                pass  # how to do it?
        trade = {
            "symbol": symbol,
            "volume": trade_volume,
            "trade_price": trade_price,
            "trade_time": trade_dt,
            "decision_time": datetime,
            "decision_price": decision_price
        }
        trade_list.append(trade)
    return trade_list


def transaction_cost_analysis(trades):
    result = []
    for trade in trades:
        fair_price = (trade.bid0 + trade.ask0) / 2
        single_dict = {
            "comission": COMMISSION * trade.price * abs(trade.volume),
            "delay_cost": 0,  # assuming we are doing algotrading and trading immediately.
            "splippage": (trade.price - fair_price) * trade.volume,
            # dynamic_strategy? Because we will use implementation shortfall
        }
        result.append(single_dict)
    return result


start = rebalance_formal_date[0]
end = rebalance_formal_date[-1]
long_only = [0, 1, 3, 5, 7, 9, 11]
long_short = [0, 2, 4, 6, 8, 10, 12]

# long-only pnl
plt.figure(figsize=(10, 7))
for i in long_only:
    model_name = portfolio_file[i].split(".txt")[0]
    all_pnl = backtest_result[i]

    pnl = all_pnl.sum(axis=1)
    (pnl[start:end] / 1000000).plot(label=f"{model_name}")
plt.axvline(rebalance_formal_date[-2], linewidth=4)
(SNP["Open_std"] - 1)[start:end].plot(style="-.", label="S&P500")
title = "Portfolio Performance Only Long"
plt.title(title)
plt.ylabel("profit_rate")
plt.legend()
plt.savefig(f"{title}.png")

# long and short pnl
plt.figure(figsize=(10, 7))
for i in long_short:
    model_name = portfolio_file[i].split(".txt")[0]
    all_pnl = backtest_result[i]

    pnl = all_pnl.sum(axis=1)
    (pnl[start:end] / 1000000).plot(label=f"{model_name}")
plt.axvline(rebalance_formal_date[-2], linewidth=4)
(SNP["Open_std"] - 1)[start:end].plot(style="-.", label="S&P500")
title = "Portfolio Performance Long & Short"
plt.title(title)
plt.ylabel("profit_rate")
plt.legend()
plt.savefig(f"{title}.png")

# all sample
start = rebalance_formal_date[0]
end = rebalance_formal_date[-1]
all_p_result = []
for i in long_only:
    p_result = {}
    model_name = portfolio_file[i].split(".txt")[0]
    all_pnl = backtest_result[i]

    pnl = all_pnl.sum(axis=1)[start:end]

    draw_down = pnl - pnl.rolling(min_periods=1, window=len(pnl), center=False).max()
    pnl -= pnl[start]
    pnl_d1 = pnl.diff()

    os_pnl = pnl_d1.dropna()
    os_rate = os_pnl / 1000000

    sharpe_ratio = np.mean(os_rate) / np.std(os_rate) * (250) ** 0.5
    p_result["profit"] = pnl[-1] / 1000000
    p_result["sharpe_ratio"] = sharpe_ratio
    p_result["draw_down"] = draw_down.min()
    all_p_result.append(p_result)
pd.DataFrame(all_p_result).to_csv("all_long.csv")

# out-of-sample
start = rebalance_formal_date[-2]
end = rebalance_formal_date[-1]
all_p_result = []
for i in long_only:
    p_result = {}
    model_name = portfolio_file[i].split(".txt")[0]
    all_pnl = backtest_result[i]

    pnl = all_pnl.sum(axis=1)[start:end]

    draw_down = pnl - pnl.rolling(min_periods=1, window=len(pnl), center=False).max()
    pnl -= pnl[start]
    pnl_d1 = pnl.diff()

    os_pnl = pnl_d1.dropna()
    os_rate = os_pnl / 1000000

    sharpe_ratio = np.mean(os_rate) / np.std(os_rate) * (250) ** 0.5
    p_result["profit"] = pnl[-1] / 1000000
    p_result["sharpe_ratio"] = sharpe_ratio
    p_result["draw_down"] = draw_down.min()
    all_p_result.append(p_result)
pd.DataFrame(all_p_result).to_csv("out_long.csv")

# all sample
start = rebalance_formal_date[0]
end = rebalance_formal_date[-1]
all_p_result = []
for i in long_short:
    p_result = {}
    model_name = portfolio_file[i].split(".txt")[0]
    all_pnl = backtest_result[i]

    pnl = all_pnl.sum(axis=1)[start:end]

    draw_down = pnl - pnl.rolling(min_periods=1, window=len(pnl), center=False).max()
    pnl -= pnl[start]
    pnl_d1 = pnl.diff()

    os_pnl = pnl_d1.dropna()
    os_rate = os_pnl / 1000000

    sharpe_ratio = np.mean(os_rate) / np.std(os_rate) * (250) ** 0.5
    p_result["profit"] = pnl[-1] / 1000000
    p_result["sharpe_ratio"] = sharpe_ratio
    p_result["draw_down"] = draw_down.min()
    all_p_result.append(p_result)
pd.DataFrame(all_p_result).to_csv("all_ls.csv")

# out-of-sample
start = rebalance_formal_date[-2]
end = rebalance_formal_date[-1]
all_p_result = []
for i in long_short:
    p_result = {}
    model_name = portfolio_file[i].split(".txt")[0]
    all_pnl = backtest_result[i]

    pnl = all_pnl.sum(axis=1)[start:end]

    draw_down = pnl - pnl.rolling(min_periods=1, window=len(pnl), center=False).max()
    pnl -= pnl[start]
    pnl_d1 = pnl.diff()

    os_pnl = pnl_d1.dropna()
    os_rate = os_pnl / 1000000

    sharpe_ratio = np.mean(os_rate) / np.std(os_rate) * (250) ** 0.5
    p_result["profit"] = pnl[-1] / 1000000
    p_result["sharpe_ratio"] = sharpe_ratio
    p_result["draw_down"] = draw_down.min()
    all_p_result.append(p_result)
pd.DataFrame(all_p_result).to_csv("out_ls.csv")
