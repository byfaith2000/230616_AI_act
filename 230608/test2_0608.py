import numpy as np

def list_test1():
    ary = [
        [1,2], 
        [3,4]
        ]
    #print(ary)

    ary2 = [
        [1,2,3,4,5], 
        [6,7,8,9,10],
        [11,12,13,14,15]
        ]
    #print(f"rows(row): {len(ary2)}, cols(column): {len(ary2[0])}")
    print(ary2[2][3])

def list_test2(rows, cols):
    ary = []
    for r in range(rows):
        ary += [[0] * cols]
    print(ary)

def list_test3():
    ary1 = [1,2,3]
    ary2 = [2,4,6]

    print(ary1 + ary2)

    for i in zip(ary1, ary2):
        print(f"{i[0] + i[1]}", end=" ")

def list_test4():
    a1 = [
        [1,2,3], 
        [1,2,3]
        ]
    a2 = [
        [4,5], 
        [6,7], 
        [8,9]
        ]

    npa1 = np.array([
        [1,2,3], 
        [1,2,3]
        ])
    
    npa2 = np.array([
        [4,5], 
        [6,7], 
        [8,9]
        ])
    
    npa3 = np.array([
        [1,2,3], 
        [1,2,3],
        [1,2,3]
        ])
    
    npa4 = np.array([
        [4,5,0], 
        [6,7,1], 
        [8,9,2]
        ])

    #print(npa1 @ npa2)
    print(npa3 * npa4)



if __name__ == "__main__":
    list_test4()
    #list_test2(3,4)
    #print(np.__version__)
