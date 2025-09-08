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
