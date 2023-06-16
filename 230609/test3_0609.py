import numpy as np
import pandas as pd

def SeriesType():
    ser = pd.Series([1,-3,2,10])
    #print(ser)
    #print(ser.values)
    #print(ser.index)

    ser = pd.Series(['One', 'Two', 'Three'],
                    index = ['c', 'a', 'd'])
    #print(ser)

    dict2 = {'One':'c', 'Two':'a', 'Three':'d'}
    ser2 = pd.Series(dict2)
    print(ser2)

    dict = {'Bab':30, 'Anna':12, 'Maria':27}
    ser = pd.Series(dict)
    #print(ser)

    ser.name = 'Family'
    ser.index.name = 'Name'
    print(ser)

def DataFrameType():
    data = {"Name": ['Bab', 'Anna', 'Maria'],
            "Age" : [30, 12, 27],
            "Student": [3,1,2]}
    df = pd.DataFrame(data)
    

    # print(df.index)
    # print(df.values)
    # print(df.columns)

    df = pd.DataFrame(data, columns=["Student", "Name", "Age", "Score"])
    #print(df)

    df = pd.DataFrame(data, columns=["Student", "Name", "Age", "Score"],
                      index=['1st', '2nd', '3rd'])
    #print(df)

    #value = df['Name']
    #value = df.Name
    value = df[['Name', 'Score']]    
    print(f"value : \n{value}\nType : {type(value)}")


if __name__ == "__main__":
    #SeriesType()
    DataFrameType()