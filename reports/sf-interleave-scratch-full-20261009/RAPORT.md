# Stockfish interleave trenowany od zera

Losowa inicjalizacja ~102,28M parametrów; żadnych wag pretraina nie wczytano. Pretrain 100M występuje wyłącznie jako punkt odniesienia w ewaluacji.

Trening od zera: 7 035 726 tych samych partii, dwa przejścia; loss tylko na ruchach ludzkich.

| Metryka / checkpoint | Partie | Ludzkie top-1 | Ludzkie top-5 | Ludzkie NLL |
| --- | ---: | ---: | ---: | ---: |
| test-pretrain100m | 9666 | 51.518% | 87.053% | 1.5775 |
| test-selected | 9666 | 52.823% | 89.119% | 1.4803 |
| val-final | 9677 | 53.020% | 89.246% | 1.4742 |
| val-pretrain100m | 9677 | 51.725% | 87.126% | 1.5706 |
| val-step-10000 | 9677 | 48.035% | 84.072% | 1.7463 |
| val-step-15000 | 9677 | 49.126% | 85.568% | 1.6731 |
| val-step-20000 | 9677 | 49.911% | 86.468% | 1.6271 |
| val-step-25000 | 9677 | 50.521% | 87.127% | 1.5938 |
| val-step-30000 | 9677 | 51.109% | 87.738% | 1.5631 |
| val-step-35000 | 9677 | 51.593% | 88.106% | 1.5414 |
| val-step-40000 | 9677 | 52.036% | 88.482% | 1.5200 |
| val-step-45000 | 9677 | 52.337% | 88.758% | 1.5024 |
| val-step-5000 | 9677 | 46.080% | 81.038% | 1.8937 |
| val-step-50000 | 9677 | 52.703% | 88.987% | 1.4890 |
| val-step-55000 | 9677 | 52.863% | 89.177% | 1.4791 |

Wybrany checkpoint: experiments/sf-interleave-scratch-full-20261009-humanloss/model.pt

Test względem pretraina: top-1 +1.305 pp, NLL -0.0971.

Checkpoint wybierany wyłącznie według ludzkiego NLL na walidacji. Test porównuje wybrany checkpoint z pretrainem na tych samych partiach. Rozłączne miesiące i identyfikatory splitów dotrenowania; nakładanie z korpusem pierwotnego pretraina jest nieznane. Powtarzanie danych nie zwiększa liczby unikalnych partii. Jedna próba, bez przedziałów ufności.
