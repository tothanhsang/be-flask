import os

cwd = os.getcwd()
print(cwd)
dir = os.path.join(cwd,"python")
if not os.path.exists(dir):
    os.mkdir(dir)