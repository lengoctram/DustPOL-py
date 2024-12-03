import threading
import sys
from functools import wraps

mydata = threading.local()

__all__ = ['auto_refresh']


def auto_refresh(f):

    @wraps(f)
    def wrapper(*args, **kwargs):

        if 'refresh' in kwargs:
            refresh = kwargs.pop('refresh')
        else:
            refresh = True

        # The following is necessary rather than using mydata.nesting = 0 at the
        # start of the file, because doing the latter caused issues with the Django
        # development server.
        mydata.nesting = getattr(mydata, 'nesting', 0) + 1

        try:
            return f(*args, **kwargs)
        finally:
            mydata.nesting -= 1
            if hasattr(args[0], '_figure'):
                if refresh and mydata.nesting == 0 and args[0]._figure._auto_refresh:
                    args[0]._figure.canvas.draw()

    return wrapper

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

# def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r", line=0):
#     """
#     Call in a loop to create terminal progress bar
#     @params:
#         iteration   - Required  : current iteration (Int)
#         total       - Required  : total iterations (Int)
#         prefix      - Optional  : prefix string (Str)
#         suffix      - Optional  : suffix string (Str)
#         decimals    - Optional  : positive number of decimals in percent complete (Int)
#         length      - Optional  : character length of bar (Int)
#         fill        - Optional  : bar fill character (Str)
#         printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
#         line        - Optional  : line number to print the bar on (Int)
#     """
#     percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
#     filledLength = int(length * iteration // total)
#     bar = fill * filledLength + '-' * (length - filledLength)
#     # Move cursor to the specified line
#     sys.stdout.write(f"\033[{line}F")  # Move up to line
#     print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
#     sys.stdout.flush()
#     # Reset cursor to next line for further outputs
#     sys.stdout.write(f"\033[{line}E")  # Move back down to original position

class bcolors:
    black  = '\033[30m'
    red    = '\033[31m'
    green  = '\033[32m'
    yellow = '\033[33m'
    blue   = '\033[34m'
    purple = '\033[35m'
    cyan   = '\033[36m'
    HEADER = '\033[1;95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'