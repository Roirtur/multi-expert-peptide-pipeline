from tui.app import PeptideApp
from logger import configure_logging

if __name__ == "__main__":
    configure_logging(level="INFO")
    PeptideApp().run()
