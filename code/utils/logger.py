
from datetime import datetime 
import sys

class logger():
    
    def info(string):
        print(' | ',datetime.now().replace(microsecond=0), '|   '+string, flush=True)
        #sys.stdout.flush()

