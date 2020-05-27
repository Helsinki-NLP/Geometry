
from datetime import datetime 

class logger():
    
    def info(string):
        print(' | ',datetime.now().replace(microsecond=0), '|   '+string)

