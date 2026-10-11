DEFAULT_NAME = "world"


class Greeter:
    def __init__(self, name=None):
        self.name = (name if name is not None else DEFAULT_NAME)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    def greet(self):
        return "Hello!" if not self._name.strip() else f"Hello, {self._name}!"

    def shout(self):
        if not self._name.strip():
            return "HELLO, WORLD!"
        greeting = "HELLO"
        name = self._name.upper()
        return f"{greeting}, {name}!"

    def whisper(self):
        return self.greet().lower()

    def farewell(self):
        return "Goodbye!"
