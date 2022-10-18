class Person:
    def __init__(self, name):
        super(Person, self).__init__()
        self.name = name
        print("__init__"+"hello " + name)

    def hello(self):
        print("hello", self.name)


if __name__ == '__main__':
    person = Person("zhang")
    person.hello()
