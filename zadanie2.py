import random
literki = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
print("Ile literek wylosować?")

ilośc = int(input())
for i in range(ilośc):
    print(random.choice(literki))