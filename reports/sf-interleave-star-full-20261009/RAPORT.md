# Kontrola compute: człowiek + pusta gwiazdka, od zera

Losowa inicjalizacja ~102,28M parametrów; żadnych wag pretraina nie wczytano. Pretrain 100M występuje wyłącznie jako punkt odniesienia w ewaluacji.

Trening od zera: 7 035 726 tych samych partii, dwa przejścia; loss tylko na ruchach ludzkich. Wszystkie wejścia SF zastąpione stałym tokenem ★ (4080), także w walidacji i teście. Ten sam seed, architektura, długości, batch i harmonogram co scratch SF; bez informacji SF.

| Metryka / checkpoint | Partie | Ludzkie top-1 | Ludzkie top-5 | Ludzkie NLL |
| --- | ---: | ---: | ---: | ---: |
| test-pretrain100m | 9666 | 51.518% | 87.053% | 1.5775 |
| test-selected | 9666 | 52.673% | 89.498% | 1.4689 |
| val-final | 9677 | 52.948% | 89.557% | 1.4627 |
| val-pretrain100m | 9677 | 51.725% | 87.126% | 1.5706 |
| val-step-10000 | 9677 | 45.728% | 83.905% | 1.7730 |
| val-step-15000 | 9677 | 47.575% | 85.636% | 1.6862 |
| val-step-20000 | 9677 | 48.757% | 86.656% | 1.6344 |
| val-step-25000 | 9677 | 49.735% | 87.401% | 1.5936 |
| val-step-30000 | 9677 | 50.515% | 87.981% | 1.5614 |
| val-step-35000 | 9677 | 51.071% | 88.422% | 1.5367 |
| val-step-40000 | 9677 | 51.667% | 88.755% | 1.5135 |
| val-step-45000 | 9677 | 52.093% | 89.101% | 1.4937 |
| val-step-5000 | 9677 | 41.718% | 79.734% | 1.9679 |
| val-step-50000 | 9677 | 52.522% | 89.311% | 1.4791 |
| val-step-55000 | 9677 | 52.802% | 89.507% | 1.4678 |

Wybrany checkpoint: experiments/sf-interleave-star-full-20261009-humanloss/model.pt

Test względem pretraina: top-1 +1.155 pp, NLL -0.1085.

SF minus gwiazdka: top-1 +0.151 pp; NLL +0.0114. Kroki SF/gwiazdka: 59631/59631; równy końcowy budżet: True.

Checkpoint wybierany wyłącznie według ludzkiego NLL na walidacji. Test porównuje wybrany checkpoint z pretrainem na tych samych partiach. Rozłączne miesiące i identyfikatory splitów dotrenowania; nakładanie z korpusem pierwotnego pretraina jest nieznane. Powtarzanie danych nie zwiększa liczby unikalnych partii. Jedna próba, bez przedziałów ufności.
