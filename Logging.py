import datetime as dt
import os

class Logging():
    #PUBLIC
    def __init__(self, path: str = None):
        self.__date = dt.date.today()

        if (path is None or path == ''):
            self.__path_to_file = str('Logs/') + str('LOG ') + str(self.__date) + '.txt'
            try:
                os.mkdir('Logs')
            except FileExistsError:
                pass
        else:
            self.__path_to_file = path

        self.__txt_file = None

    #PRIVATE
    def push_log(self, log: str):
        current_time = str(dt.datetime.now().time())[:8]
        standard_log = f'[{self.__date} {current_time}]'
        final_log = standard_log + '-> ' + log + '\n'

        with open(file = self.__path_to_file, mode = 'a', encoding='UTF-8') as self.__txt_file:
            self.__txt_file.write(final_log)
