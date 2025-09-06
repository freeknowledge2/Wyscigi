import random
import string

literki = string.ascii_letters + "  \n"
print("Ile literek wylosować?")
odpowiedz = input()

    ile = int(odpowiedz)
    for i in range(ile):
        print(random.choice(literki), end="")