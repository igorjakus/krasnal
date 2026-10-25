# Czy modele kopiują ruchy Stockfisha?

Ewaluacja diagnostyczna na tym samym teście: 9 666 partii, 652 712 ruchów
ludzkich. Nie trenowano żadnego modelu. Dla każdego ruchu model widział
prawdziwą podpowiedź Stockfisha (teacher forcing) albo — w ramieniu kontrolnym —
stałą gwiazdkę `4080`; następnie porównano jego najwyżej oceniony token ruchu
ludzki z ruchem człowieka i podpowiedzią SF.

| Model / warunek wejściowy | Top-1 ruchu ludzkiego* | Predykcja równa SF (wszystkie pozycje) | Predykcja równa SF, gdy SF ≠ człowiek |
|---|---:|---:|---:|
| Fine-tune 100M, prawdziwe SF | 54.956% | 50.309% | 29.422% |
| Scratch, prawdziwe SF | 52.830% | 52.462% | 32.810% |
| Scratch, gwiazdka zamiast SF | 52.685% | 44.821% | 24.732% |

Na 39.217% pozycji ruch człowieka i SF był taki sam. Na pozostałych 396 739
pozycjach scratch trenowany z prawdziwym SF wybierał ruch SF w 32.810% przypadków;
scratch z gwiazdką — w 24.732%. Sparowana różnica wynosi **+8.077 pp**,
95-procentowy przedział bootstrapu klastrowanego po partiach: **[+7.944,
+8.220] pp** (4 000 replikacji). To mocny sygnał, że scratch model nauczył się
warunkować wybór na tożsamości ruchu SF, a nie tylko na obecności dodatkowego
tokena.

Nie oznacza to bezmyślnego kopiowania: scratch-SF nie wybierał ruchu SF w
67.190% przypadków rozbieżności. Nie dowodzi też „mądrego” użycia ani poprawy
siły gry. Wcześniejszy pełny test scratch-SF kontra gwiazdka pokazał zaledwie
+0.151 pp standardowego top-1, przy gorszym NLL o 0.0114 dla SF. Zatem
podpowiedź zmienia zachowanie, ale na razie brak dowodu, że ten efekt poprawia
przewidywanie ludzi lub jakość szachową.

Wynik fine-tune 100M podano osobno. Nie ma odpowiadającego mu ramienia
„100M + gwiazdka”, więc jego częstość zgody z SF nie jest przyczynowym
porównaniem wpływu SF.

\* Top-1 ograniczono do słownika tokenów ruchów ludzkich; nie stosowano maski
legalności ruchu na planszy. Ruch zgodny z SF nie musi być legalny ani dobry w
szerszym sensie pozycyjnym — wskaźnik mierzy podobieństwo wyboru, nie siłę gry.
Format wejścia odpowiada treningowi tych modeli (bez `effort:0`). Test zawierał
rzeczywiste, teacher-forced ruchy SF; raport nie mierzy swobodnego generowania
całej partii ani Elo.

## Pochodzenie

- Test JSONL SHA-256: `0cfd3bf2dcc42e156745576b132dd22555c54be31d4cae97c80c975e730f1885`
- Checkpointy: fine-tune 100M `0605bf201b31397e50ed36705d4a5d3b8ae7f86607da1b45493f18b1c1544f33`; scratch-SF `62b9ee761d61fc5322d46b8feadbffbe4f0030ab76b6132e5171bbbb94945751`; scratch-star `035840de972a256364dd18d39d2bc96b3f201cb577ce218a746ce61a3a8a4b50`.
- Obliczenia float32, wyłącznie CPU na UWR; job 19178. Kod modelu, tokenizacji i kodowania interleave miał SHA-256 zgodny z manifestami treningów.
- JSON wynikowy: [`copy-agreement.json`](copy-agreement.json), SHA-256 `5afdf9fa58ee09a930b432fa64628e6a97036cc96971e467260c9e5e6e6ab6cd`.
