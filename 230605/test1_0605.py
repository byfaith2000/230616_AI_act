def FileWrite():
    lines = open('Write_text2.txt', mode='a')
    #lines.write('Write_text!!!')
    lines.writelines(['\n'.join(['aaa', 'bbb', 'ccc'])])
    # => aaa\nbbb
    lines.close()

def FileRead():
    lines = open('Write_text2.txt', mode='r')
    # text = lines.read(2)
    # text = lines.readline()
    # print(text)

    # before = lines.tell()
    # text = lines.readlines()
    # after = lines.tell()
    # print(f'Text : {text}\nNow Pos : {before}\nAfter : {after}')

    lines.seek(2)
    text = lines.readlines()
    print(text)

def AppendText():
    alltext = list()

    text1 = open('Write_text1.txt', mode='r')
    alltext.append(text1.read())
    text1.close()

    text2 = open('Write_text2.txt', mode='r')
    for line in text2:
        alltext.append(line)
    text2.close()

    lines = open('Write_Append.txt', mode='w')
    lines.writelines(alltext)
    lines.close()

    #print(alltext)

def dict1():
    typing = input("Input Text : ")
    word_lower = typing.lower()
    eng_dict = {"one":"1", "two":"2", "three":"3"}
    print(eng_dict.get(word_lower, "No"))

def dict2():
    word_dict = dict()
    lines = open('Write_text1.txt', mode='r', encoding="UTF-8")
    for line in lines:
        words = line.split()
        for word in words:
            word_lower = word.lower()
            if word_lower not in word_dict:
                word_dict[word_lower] = 1
            else:
                word_dict[word_lower] += 1
        print(word_dict)

if __name__ == "__main__":
    # FileWrite()
    #FileRead()

    #AppendText()
    dict2()