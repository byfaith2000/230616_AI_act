def guess_num(num, cnt):
    tries = 0
    while tries != cnt:
        try:
            guess = int(input("Input: "))
            tries += 1
            if guess < num:
                print("lower")
            elif guess > num:
                print("bigger")
            else:
                print(f"{tries}th try")

        except:
            print("only number")


if __name__=="__main__":
    guess_num(3,3)