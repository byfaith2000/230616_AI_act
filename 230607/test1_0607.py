# def dict2():
#     list_dict = dict()
#     str = open('Write_text1.txt', mode='r', encoding="UTF-8")
#     for line in lines:
#         lists = line.list(lines)
#         for list in lists:
#             list_lower = list.lower()
#             if list_lower not in list_dict:
#                 list_dict[list_lower] = 1
#             else:
#                 list_dict[list_lower] += 1
#         print(list_dict)

# if __name__ == "__main__":

#     dict2()

#Answer
def dict_char():
    char_dict = dict()  #generate dic
    lines = open('Write_text1.txt', mode='r', encoding="UTF-8") 

    for line in lines:
        words = line.split()
        for word in words:
            for char in word:
                char_lower = char.lower()
                if char_lower not in char_dict:
                    char_dict[char_lower] = 1
                else:
                    char_dict[char_lower] += 1
    s_char_dict = dict(sorted(char_dict.items()))
    print(s_char_dict)

if __name__ == "__main__":

    dict_char()