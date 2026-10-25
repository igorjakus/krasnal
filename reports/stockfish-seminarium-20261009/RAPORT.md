# Stockfish a przewidywanie ruchów ludzkich

Aktualizacja: 2026-10-25T16:34:49+01:00

Hipoteza: informacja o najlepszym ruchu Stockfisha przed ruchem człowieka poprawia predykcję ruchu człowieka.

Pretrain 100,7M parametrów; oba dotrenowania mają 102,3M po rozszerzeniu słownika. Osobne tokeny SF, sekwencja S–człowiek–S–człowiek. Stockfish 17.1: 30 tys. węzłów, 1 wątek. Trening: 1 156 769 partii, 78 044 747 ludzkich ruchów. Walidacja: maj 2024 (9677 partii); test: czerwiec 2024 (9666 partii).

Ewaluacja podaje prawdziwe ruchy SF przy prawdziwym prefixie partii. Wyniki dotyczą predykcji ruchów ludzkich; wyższe top-1/top-5 i niższe NLL są lepsze.

| Wynik / checkpoint | Partie | Ruchy ludzkie | Top-1 | Top-5 | NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| test-augmented-step10000 | 9666 | 652712 | 53.133% | 89.658% | 1.4560 |
| test-augmented-step2000 | 9666 | 652712 | 50.947% | 87.985% | 1.5569 |
| test-augmented-step20000 | 9666 | 652712 | 53.342% | 89.791% | 1.4480 |
| test-augmented | 9666 | 652712 | 53.342% | 89.791% | 1.4480 |
| test-human-step1000 | 9666 | 652712 | 53.535% | 89.973% | 1.4389 |
| test-human-step10000 | 9666 | 652712 | 53.743% | 90.132% | 1.4292 |
| test-human-step5000 | 9666 | 652712 | 53.680% | 90.103% | 1.4309 |
| test-human | 9666 | 652712 | 53.743% | 90.132% | 1.4292 |
| test-pretrain-early | 9666 | 652712 | 50.497% | 86.176% | 1.6465 |
| test-pretrain | 9666 | 652712 | 50.497% | 86.176% | 1.6465 |
| val-augmented-step10000 | 9677 | 654407 | 53.346% | 89.758% | 1.4490 |
| val-augmented-step2000 | 9677 | 654407 | 51.102% | 88.107% | 1.5498 |
| val-augmented-step20000 | 9677 | 654407 | 53.550% | 89.892% | 1.4409 |
| val-augmented | 9677 | 654407 | 53.550% | 89.892% | 1.4409 |
| val-human-step1000 | 9677 | 654407 | 53.788% | 90.065% | 1.4321 |
| val-human-step10000 | 9677 | 654407 | 54.015% | 90.191% | 1.4225 |
| val-human-step5000 | 9677 | 654407 | 53.975% | 90.191% | 1.4241 |
| val-human | 9677 | 654407 | 54.015% | 90.191% | 1.4225 |
| val-pretrain-early | 9677 | 654407 | 50.809% | 86.269% | 1.6396 |
| val-pretrain | 9677 | 654407 | 50.809% | 86.269% | 1.6396 |

Test, kontrola 1000 / treatment 2000 kroków: zmiana top-1 = -2.589 punktu procentowego; zmiana NLL = +0.1180.

Test, kontrola 5000 / treatment 10000 kroków: zmiana top-1 = -0.547 punktu procentowego; zmiana NLL = +0.0251.

Test, kontrola 10000 / treatment 20000 kroków: zmiana top-1 = -0.401 punktu procentowego; zmiana NLL = +0.0188.

Uwagi do seminarium: porównanie 2× kroków treatmentu przy tej samej globalnej liczbie sekwencji daje zbliżoną ekspozycję na ludzkie targety, nie identyczny koszt obliczeń. Splity mają rozłączne miesiące i identyfikatory partii. Nakładanie z korpusem oryginalnego pretraina jest nieznane. Brak przedziałów ufności; to wstępny eksperyment. Kolejne checkpointy są raportowane jawnie, nie wybierane według najlepszego wyniku testu.

## Ostatni stan jobów

```text
JOBID                   NAME ST       TIME       NODELIST(REASON)
```
