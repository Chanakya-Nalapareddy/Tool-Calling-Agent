import logging

# Configure the root logger
logging.basicConfig(
    filename="agent.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)
