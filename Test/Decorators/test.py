import inspect
class _RunnerDecorator:
    def __init__(self, meta: "A") -> None:
        self._roles = {"1"}
        self._meta = meta

    @property
    def roles(self):
        print("role_", self._roles)
        return self

    @property
    def on_guest(self):
        self._roles.add("guest")
        print("guest_", self._roles)
        return self

    @property
    def on_host(self):
        self._roles.add("host")
        print("host_", self._roles)
        return self

    @property
    def on_arbiter(self):
        self._roles.add("arbiter")
        print("arbiter", self._roles)
        return self

    @property
    def on_local(self):
        self._roles.add("local")
    def __call__(self, cls):
        return cls

class A:
    def __init__(self):
        pass
    @property
    def fun2(self):
        return _RunnerDecorator(self)

class C_fun_G:
    def __init__(self):
        pass
class C_fun_H:
    def __init__(self):
        pass
class C_fun_A:
    def __init__(self):
        pass

my_test = A()
@my_test.fun2.on_guest
def fun_G():
    return C_fun_G

@my_test.fun2.on_host
def fun_H():
    return C_fun_H

@my_test.fun2.on_arbiter
def fun_A():
    return C_fun_A
print("S"*10)
J = fun_G
inspect.getmembers(J)