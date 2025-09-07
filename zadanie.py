import random
import string

literki = string.ascii_letters
print("Ile literek wylosować?")
ile = int(input())
for i in range(ile):
    print(random.choice(literki))