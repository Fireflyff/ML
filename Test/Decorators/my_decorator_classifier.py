import inspect
import typing


class _RunnerDecorator:
    def __init__(self, meta: "ComponentMeta") -> None:
        self._roles = set()
        self._meta = meta

    @property
    def roles(self):
        print("role_", self._roles)
        return self._roles

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
        return self

    @property
    def on_local(self):
        self._roles.add("local")
        return self

    def __call__(self, cls):
        if inspect.isclass(cls):
            for role in self._roles:
                self._meta._role_to_runner_cls[role] = cls
        elif inspect.isfunction(cls):
            for role in self._roles:
                self._meta._role_to_runner_cls_getter[role] = cls
        else:
            raise NotImplementedError(f"type of {cls} not supported")

        return cls


class ComponentMeta:
    __name_to_obj: typing.Dict[str, "ComponentMeta"] = {}

    def __init__(self, name, *others) -> None:
        if len(others) > 0:
            self._alias = [name, *others]
            self._name = "|".join(self._alias)
        else:
            self._alias = [name]
            self._name = name
        self._role_to_runner_cls = {}
        self._role_to_runner_cls_getter = {}  # lazy
        self._param_cls = None
        self._param_cls_getter = None  # lazy

        for alias in self._alias:
            self.__name_to_obj[alias] = self

    @property
    def name(self):
        return self._name

    @property
    def alias(self):
        return self._alias

    @property
    def bind_runner(self):
        return _RunnerDecorator(self)

    @property
    def bind_param(self):
        def _wrap(cls):
            if inspect.isclass(cls):
                self._param_cls = cls
            elif inspect.isfunction(cls):
                self._param_cls_getter = cls
            else:
                raise NotImplementedError(f"type of {cls} not supported")
            return cls

        return _wrap
