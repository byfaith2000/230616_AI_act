student1 = 0
student2 = 0

def plus1(num):
    global student1
    student1 += num
    return student1

def plus2(num):
    global student2
    student2 += num
    return student2

class Operation:
    def __init__(self):
       self.result = 2

    def plus(self, num):
        self.result += num
        return self.result
    
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self): 
        print(f"Name : {self.name}")
        print(f"Age : {self.age}")


if __name__=="__main__":
    # print(f"student1 = {plus1(3)}")
    # print(f"student2 = {plus2(4)}")

    # print(f"student1 = {plus1(5)}")
    # print(f"student2 = {plus2(5)}")

    # std1 = Operation()
    # std2 = Operation()

    # print(std1.plus(3))
    # print(std1.plus(5))

    # print(std2.result)

    std1 = Person('John', 25)
    std1.introduce()

    std2 = Person('Anna', 30)
    std2.introduce()

    Person.birth = '0602'
    print(std1.birth)
    print(std2.birth)


