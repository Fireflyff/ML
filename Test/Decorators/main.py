from my_decorator_classifier import ComponentMeta
import inspect

my_classify = ComponentMeta("LR", "cd")


@my_classify.bind_param
def func_param():
    print("param")


# @my_classify.bind_runner.on_host
def fun_host():
    print("host")


# @my_classify.bind_runner.on_guest
def fun_guest():
    print("guest")


# @my_classify.bind_runner.on_arbiter
def fun_arbiter():
    print("arbiter")


# print(inspect.getmembers(my_classify))
# print(inspect.getmembers(my_classify, inspect.isclass))
# print(inspect.getmembers(my_classify, inspect.isfunction))
# print(inspect.getmembers(my_classify, inspect.ismethod))
# # print(inspect.getmembers(my_classify, inspect.ismodule))
# print(ComponentMeta.name)
fun_host()
fun_guest()
print(my_classify.bind_runner.roles)
print(my_classify.bind_runner._meta)

