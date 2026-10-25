# Pełny Stockfish interleave z pretraina 100M

Foundation: human-pretrain-100m-20260730; checkpoint przypięty SHA-256. 100M partii treningowych, około 100,76M parametrów przed dodaniem tokenów SF.

Dotrenowanie: 7 035 726 oryginalnych partii, dwa przejścia przez pełne bloki danych. Osobne tokeny SF; loss tylko na ruchach ludzkich. SF zawsze podawany jako prawdziwa podpowiedź przy prefixie partii. Bez nowego treningu kontrolnego.

| Metryka / checkpoint | Partie | Ludzkie top-1 | Ludzkie top-5 | Ludzkie NLL |
| --- | ---: | ---: | ---: | ---: |
| test-pretrain100m | 9666 | 51.518% | 87.053% | 1.5775 |
| test-selected | 9666 | 54.946% | 90.804% | 1.3808 |
| val-final | 9677 | 55.222% | 90.942% | 1.3734 |
| val-pretrain100m | 9677 | 51.725% | 87.126% | 1.5706 |
| val-step-10000 | 9677 | 54.718% | 90.661% | 1.3925 |
| val-step-15000 | 9677 | 54.895% | 90.743% | 1.3860 |
| val-step-20000 | 9677 | 54.966% | 90.809% | 1.3825 |
| val-step-25000 | 9677 | 55.071% | 90.866% | 1.3794 |
| val-step-30000 | 9677 | 55.148% | 90.885% | 1.3774 |
| val-step-35000 | 9677 | 55.131% | 90.891% | 1.3765 |
| val-step-40000 | 9677 | 55.193% | 90.903% | 1.3754 |
| val-step-45000 | 9677 | 55.194% | 90.918% | 1.3748 |
| val-step-5000 | 9677 | 54.303% | 90.398% | 1.4086 |
| val-step-50000 | 9677 | 55.223% | 90.933% | 1.3740 |
| val-step-55000 | 9677 | 55.230% | 90.949% | 1.3736 |

Wybrany checkpoint: experiments/sf-interleave-100m-full-20261009-humanloss/model.pt

Test względem pretraina: top-1 +3.428 pp, NLL -0.1967.

Checkpoint wybierany wyłącznie według ludzkiego NLL na walidacji. Test porównuje wybrany checkpoint z pretrainem na tych samych partiach. Rozłączne miesiące i identyfikatory splitów dotrenowania; nakładanie z korpusem pierwotnego pretraina jest nieznane. Powtarzanie danych nie zwiększa liczby unikalnych partii. Jedna próba, bez przedziałów ufności.
