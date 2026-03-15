from textual.widgets import Static, Log
from textual.app import ComposeResult

class GlobalLogWidget(Static):
    def compose(self) -> ComposeResult:
        yield Log(id="main-global-log")
