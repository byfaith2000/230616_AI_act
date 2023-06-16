import numpy as np
import pandas as pd

def DataFrame1() : 
    data = {"Name" :['Alice', 'Bab', 'Judy', 'Maria'],
            "Score" : [40,46,35,43]}
    df = pd.DataFrame(data, 
                      columns=['Name', 'Score'],
                      index=['One','Two','Three', 'Four'])
    df['Add_Point'] = df['Score']*0.1
    df['Exam'] = df['Score']>40

    #1
    df.loc[df['Exam']==False, 'Add_Point'] = 0

    #2
    df['Result'] = df['Score'] + df['Add_Point']
    print(df)

    #3
    grade = []
    for value in df['Result']:
        val = int(value)
        if(val >= 45):
            grade.append('A')
        elif((val>=40)&(val<45)):
            grade.append('B')
        else:
            grade.append('C')

    #print(grade)
    df['grade'] = grade
    print(df)


if __name__ == "__main__":
    DataFrame1()
