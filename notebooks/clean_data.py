import string 

with open('eminem dataset/ALL_eminem.txt', 'r', encoding='utf-8') as f:
    text = f.read()

allowed = set(string.ascii_letters + string.digits + " \n.,!?:'\"()-[]")
clean_text = ''.join(c for c in text if c in allowed)

with open('eminem dataset/ALL_eminem_clean.txt', 'w', encoding='utf-8') as f: 
    f.write(clean_text)

print("done")