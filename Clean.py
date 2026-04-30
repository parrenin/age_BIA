import os
import re
import sys

dir = sys.argv[1]
inclusive = True

import os
import re
import sys

dir = sys.argv[1]
inclusive = True

def clean(pattern):
    regexObj = re.compile(pattern)
    for root, dirs, files in os.walk(dir, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            if bool(regexObj.search(path)) == bool(inclusive):
                os.remove(path)
        for name in dirs:
            path = os.path.join(root, name)
            if len(os.listdir(path)) == 0:
                os.rmdir(path)
    
list_pattern = ['.pdf', 'output.txt', '.svg']
for pattern in list_pattern:
    clean(pattern)
