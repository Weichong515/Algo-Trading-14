# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 06:25:52 2019

@author: 51667
"""

import datetime

class action:
    symbol = ""
    datetime = datetime.datetime.now()
    pos = 0
    patience = 0
    
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
    quote_data = pd.read_csv(f"{date}_{symbol.upper()}.csv")
    trade_list = []
    
    for i, row in quote_data.iterrows():
        if row["time"] >= dt:
            if not decision_price:
                decision_price = (row["BID"] + row["ASK"])/2
            if patience == 0：:
                if pos > 0:
                    trade_price = row["ASK"]
                    break
                elif pos < 0:
                    trade_price = row["BID"]
                    break
            else:
                pass  # how to do it?
    trade = {"trade_price": trade_price,
             "decision_price": decision_price}
    return trade