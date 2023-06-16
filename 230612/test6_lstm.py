import numpy as np
import pandas as pd

def DataFrame2() : 
    data = {"Name" :['Bab', 'Anna', 'Maria'],
            "Age" : [30,12,27],
            "Student":[3,1,2]}
    df = pd.DataFrame(data, 
                      columns=['Student', 'Name', 'Age', 'Score'],
                      index=['1st','2nd','3rd'])
    
    #df['Score']=100
    df['Score'] = [100, 90, 80]

    df['Count'] = np.arange(3)

    ser = pd.Series(['True','False'], index=['1st', '2nd'])
    df['Pass'] = ser
    print(df)

def DataFrameCal():
    dict = {'Name':['Alice', 'Bab', 'Judy'],
           'Number':[1,2,3],
           'Score':[40,46,35]}
    df = pd.DataFrame(dict, index=['One', 'Two', 'Three'])
    df['Add_Point'] = df['Score']*0.1
    df['Exam'] = df['Score']>40
    del df['Number']

    #data = df.loc['Three']
    #data = df.loc['Two':'Three','Exam']

    df.loc['Four']=['Maria', 43, 4.3, True]

    data = df.iloc[0:2, 1:3]
    data = df.iloc[[0,2],[1,2]]

    data = df.loc[df['Score']>40, :]
    data = df.loc[(df['Score']>40)&(df['Score']<45), :]

    print(data)

if __name__ == "__main__":
    #DataFrame2()
    DataFrameCal()