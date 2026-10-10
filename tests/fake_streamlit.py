"""LOCAL TEST double for the few Streamlit calls ui/chat.py makes. It records WHAT is rendered and WHERE (container
path), so UI structure and the chip -> agent flow can be tested without a browser."""


class _Ctx:
    def __init__(self, st, key):
        self.st, self.key = st, key

    def __enter__(self):
        self.st.stack.append(self.key)
        return self

    def __exit__(self, *a):
        self.st.stack.pop()


class _Slot:
    def __init__(self, st):
        self.st = st

    def markdown(self, body, **kw):
        self.st.events.append(("markdown", body, tuple(self.st.stack)))


class FakeSt:
    def __init__(self, typed=None):
        self.session_state, self.stack, self.events, self.buttons, self.typed = {}, [], [], {}, typed

    def container(self, key=None, **kw):
        self.events.append(("container", key, tuple(self.stack)))
        return _Ctx(self, key)

    def markdown(self, body, **kw):
        self.events.append(("markdown", body, tuple(self.stack)))

    def button(self, label, key=None, on_click=None, args=(), **kw):
        self.buttons[key] = (label, on_click, args)
        self.events.append(("button", key, tuple(self.stack)))
        return False

    def chat_input(self, *a, **k):
        self.events.append(("chat_input", None, tuple(self.stack)))
        return self.typed

    def caption(self, *a, **k):
        pass

    def empty(self):
        return _Slot(self)

    def of(self, kind):
        return [e for e in self.events if e[0] == kind]
