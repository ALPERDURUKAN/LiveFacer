import glob
import os

for f in glob.glob('start_portable_*.bat'):
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    content = content.replace(', torch', '')
    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)
print('Done')
