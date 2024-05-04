import logging

LOGGER_NAME = "logger"
LOGGER_FORMAT = "%(asctime)s\t%(levelname)s %(filename)s:%(lineno)s in %(funcName)s -- %(message)s"
LOGGER_LEVEL = "info"
LOGGER_LEVEL_CHOICES = ["debug", "info", "warning", "error", "critical"]


class Formatter(logging.Formatter):
    def format(self, record):
        if "metrics" in record.__dict__.keys():
            data = record.__dict__["metrics"]
            if isinstance(data, dict):
                for k, v in data.items():
                    record.msg += f" {k}: {v:.4f},"
                record.msg = record.msg[:-1]
        return super().format(record)


def get_logger(logger_name=LOGGER_NAME, logging_level=LOGGER_LEVEL):
    logger = logging.getLogger(logger_name)
    assert logging_level in LOGGER_LEVEL_CHOICES

    logging_level = logging.getLevelName(logging_level.upper())
    logger.setLevel(logging_level)

    handler = logging.StreamHandler()
    handler.setLevel(logging_level)
    handler.setFormatter(Formatter(LOGGER_FORMAT))
    logger.addHandler(handler)

    return logger
