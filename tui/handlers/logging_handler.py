import logging


class TextualLogHandler(logging.Handler):
    def __init__(self, log_widget):
        super().__init__()
        self.log_widget = log_widget

    def emit(self, record):
        log_entry = self.format(record)
        self.log_widget.app.call_from_thread(self.log_widget.write, log_entry)


def attach_textual_log_handler(chem_agent, log_widget, log_level_name: str):
    for handler in list(chem_agent.logger.handlers):
        if isinstance(handler, TextualLogHandler) and handler.log_widget is log_widget:
            chem_agent.logger.removeHandler(handler)

    tui_handler = TextualLogHandler(log_widget)
    tui_handler.setLevel(getattr(logging, log_level_name.upper(), logging.INFO))
    tui_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    chem_agent.logger.addHandler(tui_handler)
