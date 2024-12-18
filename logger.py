import time
import sys


class Logger(object):
    def __init__(self, filename=f'log_{time.time()}.log') -> None:
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass