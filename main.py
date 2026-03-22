import sys
import subprocess
from logger import configure_logging

if __name__ == "__main__":
    configure_logging(level="INFO")
    print("Starting Streamlit Application...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app/app.py"])

