def temp(num):
    if num == 0:
        print("num == 0")
        return
    
    print(f"{num} | 0000000")
    temp(num -1)
    print(f"{num} | 1111111")

def temp2(num1, num2):
    if num1 == 0:
        print(f"temp_num1({num1}, {num2}) | Finish 11")
        return
    if num2 == 0:
        print(f"temp_num2({num1}, {num2}) | Finish 22")
        return
    
    print(f"temp2({num1} - 1, {num2}) | 0000")
    temp2(num1 - 1, num2 - 1)
    print(f"temp2({num1}, {num2} - 1) | 1111")
    temp2(num1, num2 - 1)
    print(f"{num1}| {num2} | 2222")

def temp3(num1, num2, cnt):
    cnt = cnt + 1
    if num1 == 0:
        print(f"{'          '*cnt}temp_num1({num1}, {num2}) Finish")
        return
    if num2 == 0:
        print(f"{'          '*cnt}temp_num2({num1}, {num2}) Finish")
        return
    
    print(f"{'          '*cnt}temp3({num1} - 1, {num2}) | 0000")
    temp3(num1 - 1, num2, cnt)
    print(f"{'          '*cnt}temp3({num1}, {num2} - 1) | 1111")
    temp3(num1, num2 - 1, cnt)
    print(f"{'          '*cnt}{num1} | {num2} | 2222")

def fact(num):
    if num == 1:
        return 1
    return num * fact(num - 1)

# def fact2(num):
#     res = 1
#     for i in range(1, num+1):

def fib(num):
    if num == 0:
        return 0
    elif num == 1:
        return 1
    else:
        return fib(num - 1) + fib(num - 2)

def hanoi(num, a, b, c):
    if num == 1:
        print(f"plate1 {a} to {c} move")
    else:
        #print(f"1_")
        hanoi(num - 1, a, c, b)
        print(f"plate {num} {a} to {c} move.")
        hanoi(num )


if __name__ == "__main__":
    #temp2(1, 0)
    #temp3(2,2,1)
    #print(fact(5))
    # for i in range(1, 15):
    #     print(fib(i), end=" ")