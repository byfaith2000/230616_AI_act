def min_max(arr) : # arr = 
    alphas_arr = []
    digits_arr = []
    for i in arr:
        if str(i).isalpha():
            alphas_arr.append(i)
        elif i < 0:
            digits_arr.append(i)
        if str(i).isdigit():
            digits_arr.append(i)
    print(f"alphas_min: {min(alphas_arr)}, ")

def round_ex(num, nd):
    print(round(num, nd))
if __name__ == "__main__":
    round_ex(1.2278, 2)
    round_ex(1.2228, 2)

def zip_ex(arr1, arr2):
    for x in zip(arr1, arr2):
        print(x)
if __name__ == "__main__":
    arr1 = [1,2,3]
    arr2 = [4,5,6]
    zip_ex(arr1, arr2)

if __name__ == "__main__":
    s1 = "hello world hi ai-hub"
    s2 = s1.split('-')
    print(s2)

def find_ex(str1, f_word):
    str1.find(f_word)
    print(str1.find(f_word))

if __name__ == "__main__":
    str1 = "hello world hi ai-hub"
    fword = "hi"
    find_ex(str1, fword)

if __name__ == "__main__":
    multi_num = lambda x : x * 1500
    print(multi_num(3))

    func_range = list(map(lambda x : x*1500, (range(1, 6))))
    print(func_range)

