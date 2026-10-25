# 100M on-policy DAgger — wyniki po dwóch rundach

Stan sprawdzony 2026-10-10 00:46 UTC (02:46 Europe/Warsaw). Kontener
`kt-curriculum-100m-onpolicy-v1-20261009` działa, bez OOM. Dwie pełne rundy
ukończone; SFT trzeciej ukończony, generowanie 96 884 kontekstów trzeciej
rundy zakończone, następnie relabeling CPU. Nie zmieniano protokołu ani kodu.

Foundation: `human-pretrain-100m-20260730`, SHA256
`b660b1f8eef7104123dd6551eff5c63ab3b665b72a8f1a8b2d17ec6fd71c7e81`.
Start z istniejącego human pretrain, nie ponowny pretrain. 9 688 373 teacher
records podzielonych na 10 disjoint shards; po każdym SFT DAgger 96 884
przykładów z zachowanym własnym trace i loss tylko na final expert choice,
teacher replay 75/25. Fixed eval identyczny między etapami. Pretrain bez
thinking. Historyczne 34M/v4 nie są matched comparison.

Wszystkie metryki poniżej w procentach; SF eval 1000 pozycji, raw format 100.

| Etap | Human top1 | SF direct | SF thinking | Recall | Selection | Selection przy obecnym best | Błąd formatu |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pretrain | 53.70 | 44.10 | — | — | — | — | — |
| SFT 1 | 52.28 | 50.90 | 39.00 | 71.20 | 47.00 | 54.49 | 0.00 |
| DAgger 1 | 52.00 | 50.60 | 45.20 | 73.70 | 53.10 | 60.52 | 0.00 |
| SFT 2 | 51.51 | 50.70 | 39.20 | 72.70 | 49.50 | 53.92 | 0.00 |
| DAgger 2 | 51.22 | 51.80 | 44.10 | 72.80 | 55.00 | 59.89 | 0.00 |
| SFT 3 | 50.93 | 51.20 | 42.60 | 72.60 | 52.30 | 58.68 | 0.00 |

DAgger poprawia thinking względem bezpośrednio poprzedzającego SFT:
+6.2 i +4.9 pp. Thinking nadal słabszy od direct: po rundzie 2 o 7.7 pp.
Human top1 spada o 2.48 pp względem foundation. SFT 2 częściowo cofnął
poprawę thinking po DAgger 1. To obserwacje, nie dowód przyczynowy ani wynik
siły gry. 0/100 błędów formatu nie oznacza gwarancji bezbłędności.

Pierwszy DAgger checkpoint zapisany 9 października 15:12 CEST, drugi 22:02
CEST: odstęp ~6 h 50 min. Ekstrapolacja tego odstępu na pozostałe osiem
rund daje zakończenie około 12 października 04:45 CEST, z niepewnością kilku
godzin lub większą (zmienna długość generowania i obciążenie CPU). Generowanie
trace i CPU Stockfish dominują nad krótkim treningiem GPU. Gry automatycznie
dopiero po wszystkich rundach, do 15 minut. Brak wyników gier na tym etapie.

## Małe artefakty i SHA256

Kopie w `reports/dagger-100m-20261010-0046-artifacts/`; sumy kopii sprawdzone
z sumami źródeł na Adrianie:

| Plik | SHA256 |
|---|---|
| report.md | 79e0c4e5cff01ce9ac23fb44a9cd3f8fc7005437130b0d0f89547be557877135 |
| status.json | 616d177645a7f9db600320453d166698cdb4082fe7be1fff9e8059e2ee6a06bd |
| run-audit.json | 8200452e27a1d2f2e32ec35b55453ed89472728bb31a60d9664cf839a1ca61cc |
