# Gra Wyścigowa z AI - Instrukcja Obsługi

## Opis Gry
Gra wyścigowa z 10 agentami sterowanymi przez sieci neuronowe. Każdy agent uczy się jazdy po torze za pomocą algorytmu ewolucyjnego.

## Sterowanie
- **SPACJA** - Przyspiesz ewolucję (przejdź do następnej generacji)
- **R** - Restart gry (nowa populacja)
- **ESC/X** - Zamknij grę

## Jak to działa

### Sieci Neuronowe
- Każdy agent ma własną sieć neuronową (8 wejść, 16 neuronów ukrytych, 3 wyjścia)
- **Wejścia**: 8 sensorów wykrywających odległość do krawędzi toru
- **Wyjścia**: [przyspieszenie, skręt w lewo, skręt w prawo]

### Fizyka Pojazdów
- **Masa i bezwładność**: Pojazdy mają realistyczną fizykę ruchu
- **Przyspieszenie**: Maksymalna prędkość 8 jednostek/klatkę
- **Tarcie**: Pojazdy zwalniają gdy nie przyspieszają
- **Skręcanie**: Możliwość skrętu zależy od prędkości

### System Punktacji
- **+0.1 punktu** za każdą jednostkę przebytej odległości
- **-100 punktów** za wypadnięcie z toru (dyskwalifikacja)
- **Cel**: Przejechać jak najdalej bez wypadnięcia

### Algorytm Ewolucyjny
1. **Selekcja**: Najlepsze 30% (3 agentów) przechodzi do następnej generacji
2. **Reprodukcja**: Pozostałe 7 agentów to zmutowani potomkowie najlepszych
3. **Mutacja**: Losowe zmiany w wagach sieci neuronowej
4. **Czas**: Maksymalnie 30 sekund na generację

## Interfejs

### HUD (Lewy górny róg)
- Numer aktualnej generacji
- Pozostały czas w rundzie
- Najlepszy wynik wszech czasów
- Liczba aktywnych agentów

### Ranking (Prawy górny róg)
- Lista wszystkich agentów posortowana według wyniku
- ✓ = agent aktywny, ✗ = zdyskwalifikowany
- Kolory odpowiadają kolorom agentów na torze

## Cechy Zaawansowane

### Sensory
Każdy agent ma 8 sensorów rozmieszczonych wokół pojazdu:
- Promień 100 pikseli
- Wykrywanie krawędzi toru
- Znormalizowane wartości (0.0 - 1.0)

### Tor
- Kształt ósemki z ostrymi zakrętami
- Szerokość 80 pikseli (40 w każdą stronę od środka)
- Żółta linia środkowa
- Białe krawędzie
- Czerwona linia mety

## Parametry do Dostosowania

W kodzie można łatwo zmienić:
- `FPS = 60` - Płynność gry
- `max_time = 30 * FPS` - Czas na rundę
- `mutation_rate = 0.3` - Częstość mutacji
- `mutation_strength = 0.5` - Siła mutacji
- `max_speed = 8.0` - Maksymalna prędkość pojazdów
- `track_width = 40` - Szerokość toru

## Rozszerzenia

Możliwe ulepszenia:
1. **Checkpoint System**: Dodaj punkty kontrolne dla lepszej oceny postępu
2. **Więcej sensorów**: Dodaj sensory prędkości, przyspieszenia
3. **Większe sieci**: Eksperymentuj z większymi sieciami neuronowymi
4. **Różne tory**: Generuj losowe tory dla każdej generacji
5. **Zapisywanie**: Zapisuj najlepsze sieci do pliku
6. **Statystyki**: Zbieraj dane o wydajności generacji

## (NOWE) Gra 3D w Ursina

Dodano plik `race3d_ursina.py` z wersją 3D gry spełniającą wymagania:

- Silnik: Ursina (pip install ursina)
- Tor 3D: Eliptyczny, zbudowany z segmentowych barierek (Entity z collider='box').
- Samochody (N konfigurowalne) startują automatycznie i są sterowane przez sieci neuronowe uczone algorytmem ewolucyjnym.
- Fizyka: proste przyspieszenie, tarcie, skręt zależny od prędkości, kolizje z bandami -> dyskwalifikacja (auto znika / wyłączone).
- Meta: wykrycie pełnego okrążenia (progress >= 0.99) – pierwszy kończący wygrywa.
- Kamera: domyślnie widok z góry; klawisz `C` przełącza na tryb śledzenia lidera.
- Uczenie: po zakończeniu generacji (czas lub brak żywych aut) następuje selekcja, mutacje i kolejna generacja.

### Uruchomienie 3D

1. Zainstaluj zależności:
```
pip install ursina numpy
```
2. Uruchom:
```
python race3d_ursina.py
```

### Klawisze 3D
- C – przełącz kamera (follow/top)
- N – wymuś przejście do kolejnej generacji
- ESC – wyjście

### Parametry do zmiany (u góry pliku)
- `NUM_CARS`, `GEN_TIME_SECONDS`, `TRACK_WIDTH`, `TURN_SPEED`, `MUTATION_RATE`, `MUTATION_STRENGTH` itd.

### Sensory i AI (3D)
- 10 promieni (raycast) 360° w płaszczyźnie poziomej.
- Wejścia: znormalizowane odległości (0..1) do bariery lub maks. zasięgu.
- Wyjścia sieci: [przyspiesz/hamuj, skręt_lewo, skręt_prawo].
- Algorytm: prosty elitarny ewolucyjny – elita zachowana, reszta mutacje.

### Możliwe rozszerzenia 3D
1. Checkpointy i czas okrążenia.
2. Wielookrążeniowe wyścigi.
3. Różne generatory torów (proceduralnie zakręty, wzniesienia – dodając różne Y).
4. Sieci głębsze lub PPO / DQN (wymagałoby dyskretyzacji lub wrappera środowiska).
5. Zapisywanie champion.brain do pliku (pickle / npz) i wczytywanie przy starcie.

