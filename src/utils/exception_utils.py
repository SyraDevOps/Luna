def log_exception(e: Exception, context: str = "") -> None:
    logger.error(f"Error in context [{context}]: {str(e)}")
    logger.debug(traceback.format_exc())