import random
number = random.randint(1,100)
print(number)

while True:
    try:
        guess = int(input('Input number :'))
        if guess == number:
            print('OK')
            break
        elif guess > number:
            print('Lower')
        else:
            print('bigger')
    except:
        print('ddd')
