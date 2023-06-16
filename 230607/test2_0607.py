class dict_class:
    def __init__(self, f_name):
        self.f_name = f_name

    def word_cnt(self):
        #print("function of check of the same word")
        self.word_dict = dict()
        lines = open(self.f_name, mode='r', encoding="UTF-8") 
        for line in lines:
            words = line.split()
            for word in words:
                word_lower = word.lower()
                if word_lower not in self.word_dict:
                    self.word_dict[word_lower] = 1
                else:
                    self.word_dict[word_lower] += 1

    def char_cnt(self):
        print("function of check of the same char")

    def create_eng_dict(self):
        #print("function of generating of English dic")
        self.eng_dict = {"one": 1, "two": 2, "three":3}

    def find_in_eng(self, d):
        #print("interpreter throuh English dic")
        for i in d:
            if self.eng_dict.get(i) != None:
                print(f"{i} is {self.eng_dict.get(i)}")

if __name__ == "__main__":
    d = dict_class('test.txt')
    d.word_cnt()
    print(f"Word : {d.word_dict}")

    d.char_cnt()
    d.create_eng_dict()
    d.find_in_eng()