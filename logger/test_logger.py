import logging
from logger import configure_logging

def test_logger():
    # Configure the logging system
    configure_logging(log_file="test_log.log", enable_console=True, enable_file=True, level="DEBUG")

    # Create loggers for different modules
    root_logger = logging.getLogger()
    chemist_logger = logging.getLogger("peptide_pipeline.chemist")
    biologist_logger = logging.getLogger("peptide_pipeline.biologist")

    chemist_logger.setLevel(logging.INFO)
    biologist_logger.setLevel(logging.DEBUG)

    # Log messages at different levels
    root_logger.debug("This is a DEBUG message from the root logger.")
    root_logger.info("This is an INFO message from the root logger.")
    root_logger.warning("This is a WARNING message from the root logger.")
    root_logger.error("This is an ERROR message from the root logger.")
    root_logger.critical("This is a CRITICAL message from the root logger.")

    chemist_logger.debug("This is a DEBUG message from the chemist logger.")
    chemist_logger.info("This is an INFO message from the chemist logger.")
    chemist_logger.notice("This is a NOTICE message from the chemist logger.")
    chemist_logger.warning("This is a WARNING message from the chemist logger.")
    chemist_logger.error("This is an ERROR message from the chemist logger.")


    biologist_logger.debug("This is a DEBUG message from the biologist logger.")
    biologist_logger.info("This is an INFO message from the biologist logger.")
    biologist_logger.notice("This is a NOTICE message from the biologist logger.")
    biologist_logger.warning("This is a WARNING message from the biologist logger.")
    biologist_logger.error("This is an ERROR message from the biologist logger.")

if __name__ == "__main__":
    test_logger()