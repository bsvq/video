import logging
 
# Настройка логгера
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler = logging.FileHandler('livetalking.log')  # Вы можете изменить его на StreamHandler для вывода на консоль или использовать несколько Handlers в комбинации и т. д.
fhandler.setFormatter(formatter)
fhandler.setLevel(logging.INFO)
logger.addHandler(fhandler)

# handler = logging.StreamHandler()
# handler.setLevel(logging.DEBUG)
# sformatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# handler.setFormatter(sformatter)
# logger.addHandler(handler)