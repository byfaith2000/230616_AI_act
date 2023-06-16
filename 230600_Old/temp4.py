def add(x, y):
    return x + y

def substract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    return x / y

print("choice the mathmatics")
print("1.add")
print("2.substract")
print("3.multiply")
print("4.divide")
print("0.exit")

while(True):
    choice = input("choice(1/2/3/4/0):")
    if choice =='0':
        print('The end')
        break
        
    num1 = float(input('choice first number:'))
    num2 = float(input('choice second number:'))  

    if choice == '1':
        print(num1, "+", num2, "=", add(num1, num2))

    elif choice =='2':
        print(num1, "-", num2, "=", abstract(num1, num2))

    elif choice =='3':
        print(num1, "*", num2, "=", multiply(num1, num2))

    elif choice =='4':
        print(num1, "/", num2, "=", divide(num1, num2))

    else:
        print("incorrect input")

  
    print()