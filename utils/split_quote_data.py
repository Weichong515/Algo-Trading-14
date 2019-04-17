# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 03:10:19 2019

@author: 51667
"""

import numpy as np
import pandas as pd
import csv
import os

# file_name = "2018-11-01.csv"
def split_quote_data(file_name, output_path=os.getcwd()):
    """
    The function is to split the whole quote data csv into csv with respect to 
    each stock.
    Be careful that the input file_name is used to name the split files.
    """
    with open(file_name) as csvfile:
        date = file_name.split(".csv")[0].split("\\")[-1]
        reader = csv.reader(csvfile, delimiter=',')
        i = 0
        old_symbol = "A"
        data = []
        for line in reader:
            if i == 0:
                column = line  # keep the column name
            else:
                symbol = line[-2]
                if symbol != old_symbol:
                    output = pd.DataFrame(data,columns=column)
                    if output_path:
                        output.to_csv("\\".join([output_path,f"{date}_{old_symbol}.csv"]), 
                                      index=False)
                    else:
                        output.to_csv(f"{date}_{old_symbol}.csv",
                                      index=False)
                    data = []
                    old_symbol = symbol
                else:
                    data.append(line)  
            i += 1    
        output = pd.DataFrame(data,columns=column)
        if output_path:
            output.to_csv("\\".join([output_path,f"{date}_{old_symbol}.csv"]), 
                          index=False)
        else:
            output.to_csv(f"{date}_{old_symbol}.csv",
                          index=False)

if __name__ == "__main__":
    split_quote_data("2018-11-01.csv")