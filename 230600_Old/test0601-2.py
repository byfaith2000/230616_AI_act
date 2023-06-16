def integer(num):
    print("="*30)
    types = ["Dec", "Bin", "Oct", "Hex"]
    print(f"{types[0]:^4} {types[1]:^8} {types[2]:^6} {types[3]:^8}")
    print("="*30)

    for i in range(0, num+1):
        print(f"{i:4d} {i:08b} {i:06o} {i:#06x}")

if __name__=="__main__":
    try:
        typing = input("Input a Number")
        integer(int(typing))
    except:
        print("Not a number?")





    typing = input("Input a Number:")
    integer(int(typing))