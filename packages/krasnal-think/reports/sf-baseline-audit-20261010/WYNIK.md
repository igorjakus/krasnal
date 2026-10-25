# Audyt baseline — pełny test, SHA-256 zweryfikowane

9666 tych samych partii, 652712 ruchów ludzkich. Niezależny evaluator CPU float32.

| Wariant | Top-1 | NLL |
| --- | ---: | ---: |
| Pretrain bez effort0 (błędny format) | 51.5304% | 1.577603 |
| Pretrain z effort0 (poprawny format) | 54.1772% | 1.408680 |
| Dotrenowanie SF (wcześniejsza ewaluacja GPU bf16) | 54.9458% | 1.380821 |

Dotrenowanie minus poprawny baseline: +0.7686 pp; NLL -0.027859. To porównanie różnych precyzji; dla granicznego efektu należy powtórzyć oba w tej samej precyzji. Nie rozstrzyga SF versus gwiazdka. Stare wyniki zachowane bez nadpisywania.
