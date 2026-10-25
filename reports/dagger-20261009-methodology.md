# Nocny eksperyment SFT / DAgger — 8–9 października 2026

Status dokumentu: metodologia i plan; wyniki należy uzupełnić po odczycie workera.
Raport wyników zaplanowany na 08:30 Europe/Warsaw 9 października.

## Cel i hipoteza

Sprawdzić, czy końcowy wybór ruchu poprawia się, gdy DAgger uczy wyboru po
rzeczywistym thinking modelu zamiast po trace poprawionym przez nauczyciela.
Ukończenie pipeline nie oznacza potwierdzenia hipotezy.

## Protokół

- Worker: Adrian, pojedyncza RTX 3090, Docker, bez konkurencyjnego treningu.
- Konfiguracja: `configs/curriculum-clean.json`; nowy katalog
  `experiments/curriculum-onpolicy-v1-20261008`.
- Start: istniejący human pretrain `runs/pretrain/model.pt`, SHA256 przypięty
  w `foundation.json`. Nie jest to ponowne wykonanie pretrain ani start z SFT02.
- Dane: 10 jawnych źródeł teacher, poprzednio 9 688 373 unikalnych rekordów
  treningowych po walidacji/deduplikacji. Bieżący manifest jest rozstrzygający.
- Holdout: 1000 testowych i 1000 walidacyjnych historii; 10 rozłącznych shardów.
  Deduplikacja po historii ruchów, nie po FEN; nie dowodzi niezależności od pretrain.
- Każda runda: pełna epoka SFT na jej shardzie, fixed eval, generacja DAgger,
  CPU relabeling Stockfish 30k nodes, epoka DAgger z teacher replay, fixed eval.
- DAgger: ceil(0.1 * liczba rekordów shardu), około 96.9k kontekstów na rundę.
  Brak wstrzykiwania najlepszego globalnego ruchu. Model proponuje do 3 korzeni;
  Stockfish wybiera najlepszy z zaproponowanych korzeni.
- Nowy objective: zachowujemy dokładne effort/PV/delta/closing tokens modelu;
  wszystkie są context-only. Loss dotyczy tylko końcowego ruchu wybranego przez
  eksperta. Nie uczymy wygenerowanych błędów jako targetów ani nie zastępujemy
  trace trace'em Stockfisha. Eksperckie PV pozostają w rekordzie diagnostycznym.
- Teacher replay: 3 rekordy teacher na jeden DAgger (75/25 w tej epoce).
  Łączny udział DAgger w ekspozycjach całego przebiegu około 7.14%.
- Teacher SFT: direct/thinking effort 0/2, do 3 PV po 3 plies; 4 klasy delty;
  wagi PV 0.5, 0.25, 0.125; struktura/delta 0.1, końcowy ruch 1.
- Adam state przenoszony między etapami, cosine LR 5e-6 do 5e-7 na etap.
  Wznowienie etapu przywraca optimizer, RNG i cursor z atomic latest checkpoint.

## Dane na obu workerach

Adrian: 6.7 GiB katalogu SFT (obejmuje również starsze/powielone pliki);
UWR: 2.1 GiB SFT. Osiem dużych źródeł teacher w incoming ma identyczne SHA256
na obu workerach: cztery po 250k z 2019 i cztery po 1M z 2020–2023.
Nie liczymy ich dwukrotnie ani nie kopiujemy ponownie. UWR ma dodatkowe dane
SF-interleave w innym formacie oraz aktywne eksperymenty; nie mieszamy tych
danych z teacher SFT i nie ingerujemy w te zadania.

## Ewaluacja i ograniczenia

21 wierszy: pretrain oraz SFT i DAgger każdej z 10 rund. Stałe 1000 pozycji
z zamrożonym globalnym SF gold; human top1 na 30 batchach ze stałym seed;
raw format na 100 pozycjach. Pretrain: tylko human i direct SF top1, thinking
i format oznaczone jako nie dotyczy. Pozostałe etapy: human/direct/thinking
top1, candidate recall, selection, conditional selection i raw format error.

To wariant DAgger dla wyboru po on-policy trace w kontekstach teacher, nie
DAgger z pełnych rozgrywek i nie RL. Budżet i objective zmienione względem
v4, więc porównanie v4 nie izoluje wpływu pojedynczej zmiany. 1000 pozycji
i krótki ekran gier nie wystarczą do precyzyjnego Elo. Rano należy podać
faktycznie ukończone etapy; nie przedstawiać etapów trwających jako finalnych.

## Weryfikacja techniczna

Lokalnie: 20 testów zakończonych sukcesem, 1 GPU test pominięty; dodatkowy test
sprawdza zachowanie trace i maskowanie wszystkich targetów poza końcowym ruchem.
Test pełnego pipeline na Adrianie obejmuje start, resume i append nowych danych:
`kt-onpolicy-smoke-20261008`, exit 0, 1 passed / 3 deselected, 180.91 s;
cztery małe rundy ukończone. Duży pipeline uruchomiony około 23:21 CEST
8 października jako `kt-curriculum-onpolicy-v1-20261008`; start od walidacji
i deduplikacji pełnego zbioru, następnie automatycznie baseline/trening/evale.
Po ukończeniu automatyczny ekran gier ma limit 15 minut; porównuje nowy
final direct/thinking z pretrain, wcześniejszymi SFT02/SFT08 oraz SF1600
na stałych otwarciach ze zmianą kolorów. Pełny status treningu należy uzupełnić rano.

Archiwum protokołu sprzed zmiany na Adrianie:
`experiments/protocol-before-onpolicy-20261008.tar`, SHA256
`299c7a226be92befc4ba33629abd2aa6cdd1d52ae7d03262bb1d110cf8e31c91`.
Wcześniejsze dane, v4 i checkpointy pozostają zachowane.
