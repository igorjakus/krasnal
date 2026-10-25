# Wyniki na seminarium — 9 października 2026

**W tej konfiguracji dodanie najlepszych ruchów Stockfisha nie poprawiło
przewidywania ruchów ludzkich względem dotrenowania na samych ruchach ludzkich.**
Oba modele poprawiły wynik względem bazowego pretraina.

| Model | Kroki dotrenowania | Test top-1 | Test top-5 | Test NLL |
| --- | ---: | ---: | ---: | ---: |
| Pretrain | 0 | 50,497% | 86,176% | 1,6465 |
| Kontrola: ruchy ludzkie | 10 000 | 53,743% | 90,132% | 1,4292 |
| Stockfish → człowiek, osobne tokeny SF | 20 000 | 53,342% | 89,791% | 1,4480 |

Treatment względem kontroli: **−0,401 punktu procentowego top-1** oraz
**+0,0188 NLL**. Wyższe top-1 i niższe NLL są lepsze. To wynik jednego
eksperymentu, bez przedziałów ufności; nie dowodzi, że informacja Stockfisha
nie może pomóc przy innym sposobie treningu.

Oba dotrenowania zaczynały z tego samego pretraina. Dodano osobne identyfikatory
ruchów SF, zachowując oryginalne wagi i identyfikatory; oba modele mają taką samą
rozszerzoną architekturę (102,3M parametrów). Treatment ma format
`S W S B …`, gdzie S obliczono przed prawdziwym ruchem partii. Pozycja zmienia
się wyłącznie przez ruch człowieka. Ewaluacja podaje prawdziwe ruchy SF
i prawdziwą historię człowieka; mierzymy wyłącznie predykcję następnego ruchu
ludzkiego. Stockfish 17.1: 30 tys. węzłów, 1 wątek.

Trening użył 1 156 769 partii (78 044 747 oryginalnych ruchów ludzkich).
Kontrola zobaczyła 309 343 284 ludzkie targety, treatment 318 245 709
(około 2,9% więcej; to nie jest porównanie identycznego kosztu obliczeń).
Walidacja: maj 2024, 9677 partii. Test: czerwiec 2024,
9666 partii i 652 712 ruchów ludzkich. Identyfikatory i miesiące splitów
są rozłączne; nakładanie z korpusem oryginalnego pretraina jest nieznane.
Pośrednie checkpointy były ustalone z góry, a wynik końcowy nie został wybrany
według najlepszego testu.

Treningi wykonały wszystkie zaplanowane kroki. Trening kontroli trwał
około 2 h 45 min, treatmentu 3 h 3 min. Treatment użył 2× RTX 3080, kontrola
jednej RTX 3080. Pełny dataset 7 035 726 partii treningowych również jest
gotowy na UWR, ale nie był użyty w tym nocnym dotrenowaniu.

Wyniki są już na tym Macu. Sprawdzono SHA-256 bazowego wyniku i obu końcowych
checkpointowych ewaluacji względem manifestu transferu. Szczegółowe metryki,
manifesty, podsumowania treningów i wyniki pośrednie są w tym katalogu;
`RAPORT.md` zawiera pełną tabelę. Checkpointy pozostają na UWR.

Zdanie do opowiedzenia: „Sprawdziliśmy, czy podawanie podpowiedzi Stockfisha
pomaga przewidywać ruchy człowieka. Po dotrenowaniu z pretraina na 1,16 mln
partii oba modele poprawiły się, ale model z podpowiedziami SF był o 0,40 pp
słabszy niż kontrola. W tej konfiguracji hipoteza nie uzyskała potwierdzenia”.
