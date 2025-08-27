import random
kierowcy = ["jig","jan","bizon","john","owal","dummy","komin","silnik"]
#wyniki = [[Imie , czas],[Imie , czas]]
wyniki = []
def czas():
    
    time=(round(random.random()*100,3))
    return time


for kierowca in kierowcy:
    mala = [kierowca, czas()]
    wyniki.append(mala)


def wypisz_wyniki():
    print("TABELA WYNIKÓW : ")
    print("Kierowca   czas")
    print("---------------")
    for wynik in wyniki:
        print(wynik[0]+"  =>  "+str(wynik[1])+" s")




wypisz_wyniki()