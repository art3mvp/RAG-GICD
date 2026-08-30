from src.config.loader import load_settings
from src.utils.logger import get_logger


def main() -> None:
    settings = load_settings()
    logger = get_logger("main", settings.logs_dir)
    logger.info("Project loaded: %s", settings.project_name)


if __name__ == "__main__":
    main()
