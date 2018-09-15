import os

from codebase.utils.all_utils import mkdir_p

class Log():

    def __init__(self, output_dir):
        self.allow_print = True
        self.log_dir = os.path.join(output_dir, 'log')
        mkdir_p(self.log_dir)
        log_path = os.path.join(self.log_dir, 'log.txt')
        self.log_file = open(log_path, 'w')

    def disallow_print(self):
        self.allow_print = False

    def add(self, string='', pr=True):
        if pr and self.allow_print:
            print(string)
        self.log_file.write(str(string) + "\n")