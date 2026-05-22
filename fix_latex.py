import re

with open('Relatorio.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Resolve Git conflicts: always take the '=======' to '>>>>>>>' part
# The regex looks for <<<<<<< HEAD\n(.*?)\n=======\n(.*?)\n>>>>>>> [a-f0-9]+
# and replaces it with group(2)
content = re.sub(r'<<<<<<< HEAD\n.*?\n=======\n(.*?)\n>>>>>>> [a-f0-9]+', r'\1', content, flags=re.DOTALL)

# Also handle the one at the very top which might not have \n before =======
content = re.sub(r'<<<<<<< HEAD\n=======\n(.*?)\n>>>>>>> [a-f0-9]+', r'\1', content, flags=re.DOTALL)
content = re.sub(r'<<<<<<< HEAD\n=======\n(.*?)\n>>>>>>> [a-f0-9]+', r'\1', content, flags=re.DOTALL)

# Handle the one at the end
content = re.sub(r'<<<<<<< HEAD\n=======\n(.*?)\n>>>>>>> [a-f0-9]+', r'\1', content, flags=re.DOTALL)

with open('Relatorio.tex', 'w', encoding='utf-8') as f:
    f.write(content)
