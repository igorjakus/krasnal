# Diagnostyka thinking — 10 października 2026

Status: wykonano etap A (dwa ramiona SFT i fixed eval). Etap B ma wyłącznie jednokrokowy smoke test; nie ma wyników pełnego mieszanego treningu. Etap C/RL ani gry nie zostały wykonane. Seria nie osiągnęła całego celu. Automatyzację usunięto po wyczerpaniu budżetu; nie wznowiono starego curriculum.

## Metodologia

Wspólny start: DAgger3 z curriculum100M, SHA256 `086781ca9846cb4216e2b1d48781f9a7bba890fa4f13bcc5f70f2d578d684025`. Foundation: human-pretrain-100m-20260730, 29-class head, SHA256 `b660b1f8eef7104123dd6551eff5c63ab3b665b72a8f1a8b2d17ec6fd71c7e81`. Nie użyto retired34M.

Dwa niezależne ramiona kontynuowały identyczny stan optymalizatora; oba resetowały RNG seed42 (nie odtwarzały historycznego RNG checkpointu). Ten sam next-disjoint shard `data-000/round-03.jsonl`, limit 73728 pozycji, effort0/2, batch64, 1000 aktualizacji: nominalnie 64000 przykładów na ramię. Liczby unikalnych pozycji faktycznie wykorzystanych oraz token exposures nie zapisano oddzielnie. LR5e-6→5e-7, warmup50. Normalne SFT stosuje istniejący ważony loss per-example dla compact teacher trace; final-only zachowuje cały teacher trace jako wejście, ale maskuje wszystkie cele poza ostatnim ruchem. Generator nie jest zamrożony.

Fixed eval: te same 1000 pozycji, effort2 compact, zamrożone gold, Stockfish30000 nodes/4 CPU workers; human: val.bin, seed1729, 30 batches, 926144 cele ruchowe. Raw format: 100 pozycji, bez naprawy formatu. To nie są gry ani pomiar Elo.

## Wyniki

Wszystkie kolumny poza CP-gap są procentami. Niższy CP-gap jest lepszy.

| Checkpoint | Human top1 | SF direct top1 | SF thinking top1 | Recall | Selection | Conditional selection | Błędy formatu | CP-gap direct | CP-gap thinking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DAgger3, baseline | 50.58 | 51.1 | 46.1 | 71.6 | 56.6 | 64.11 | 0/100 | 174.72 | 159.83 |
| Teacher trace SFT, 1000 updates | 50.60 | 50.5 | 41.7 | 71.8 | 51.1 | 57.94 | 0/100 | 166.68 | 194.13 |
| Teacher final-only, 1000 updates | 50.58 | 50.3 | 40.7 | 68.7 | 56.3 | 59.24 | 1/100 | 166.69 | 184.88 |

Selection oznacza wybór najlepszego sampled root; conditional selection to trafienie globalnego gold, gdy ten występuje w kandydatach. Nie są to zamienne metryki.

Usunięcie PV supervision z loss nie poprawiło thinking top1: oba ramiona spadły względem startu. Final-only zachowało lepszą selection niż normalne SFT, lecz miało niższy recall. To nie potwierdza hipotezy, że jedyną przyczyną jest loss PV. Human pozostaje praktycznie bez zmian w tej krótkiej ablacji.

Ważne: niższe top1 thinking nie oznacza automatycznie gorszej jakości każdego ruchu. Baseline thinking ma niższy średni CP-gap niż direct. Top1 nie rozróżnia ruchów bliskich jakościowo, a średni CP-gap jest wrażliwy na skrajności/maty. Nie przeprowadzono paired uncertainty analysis, kilku seedów ani wiarygodnej ewaluacji jakości delt. Wnioski są diagnostyczne, nie dowodzą siły gry ani jednego root issue.

## Czas, awarie i niewykonane etapy

Pierwszy dispatch: 14:26:52 CEST; niezmienny deadline: 20:26:52 CEST. Pierwszy proces zatrzymał się na pominiętej zależności pipeline.py podczas human eval. Dodano tę zależność i wznowiono zachowane artefakty, bez zmiany deadline. Wznowiony kontener A działał 16:25:56–16:40:09 CEST i zakończył się kodem0; logi treningu pokazują około70s na 1000 updates każdego ramienia. Reszta to przygotowanie/eval/CPU scoring, nie ciągły trening GPU. Między zakończeniem A a kolejnym wywołaniem automatyzacji nie działał autonomiczny etap B/C — to brak wykonania planu, nie dowód ich nieskuteczności.

Smoke B wykonał 1 update, batch4, loss1.2299 i zapisał osobny smoke checkpoint. Pool: 16384 deduplicated onpolicy i 8192 teacher; human train.bin, nie val.bin. Oględziny czasu plików wskazują smoke około20:53 CEST, już po globalnym deadline: naruszenie limitu przy przygotowaniu, nie wynik pilota. Nie uruchomiono kolejnego treningu. B wymaga pełnego audytu leak/exposure oraz fixed eval; C wymaga implementacji/zweryfikowania RL i nowej zgody na budżet. Nie wybieramy checkpointu do RL na podstawie nieistniejących wyników B.

Stan na 20:55 CEST: sterownik NVIDIA595.99.02 działa; GPU bez procesów. CPU-only `kt-sf-cpu-extra-20261009` nadal działa i nie został naruszony. Checkpointy pozostają na Adrianie; na Mac pobrano tylko JSON/logi. Nie zmieniano UWR ani cudzych zadań.

## Provenance

Worker root: `/home/ijakus/projects/krasnal-think/experiments/thinking-diagnostic-20261010`, pinned code/config, kontener A `kt-thinking-diagnostic-a-20261010`. Oryginalne źródła i dane są zahashowane w plan.json; naprawa importu w repair-import-audit.json. Kod mieszany jest przygotowaniem po etapie A, nie kodem użytym do jego wyników. Checkpointy A:

- Teacher trace: `8a69668f6716c2faa50ca895a81e3a935672f82f4adec5df24638c707cbf74e2`.
- Final-only: `de3c4fec40c990331531df44b35f29afc6bfe02fe86c269b6d1a76e6e55cbba1`.

Małe artefakty: `reports/thinking-diagnostic-20261010-artifacts/`; poniższe SHA256 porównano na Adrianie i Macu, zgodne:

| Plik | SHA256 |
|---|---|
| plan.json | a194fb4d74b02b7c4a4693e583df4180c052b5c03ede0009cdb7d812c0ab7388 |
| repair-import-audit.json | cfd0154d9b1e54b30603d2956b79ba7f2e674a673bc5e893328fac97b01bec79 |
| stage-a-complete.json | 79f0721b99d2115af9ba198035c95395c09c8ab2840485ba58e15ca05d1361c5 |
| baseline/eval/result.json | 3d7c531034969938be4905c0dd1603fb39408d2e734284b4c87f65daa1e5d9df |
| teacher-trace/eval/result.json | 0b233434d31fc1f15511ba85988fab42ed345c5303aae9fe43b9e74ed01dfb59 |
| final-only/eval/result.json | 8880c8ebb9b943c1807adb6c3a03c2f632fcebeaf5238d5e3ad8cd8db82c05e4 |
