# Podsumowanie DAggera — 10 października 2026, 12:06 CEST

Monitor `pilnuj-nocnego-daggera-na-adrianie` na polecenie użytkownika ustawiony
na PAUSED i zweryfikowany. Nie zatrzymywaliśmy kontenera w tym sprawdzeniu:
przebieg sam zakończył się kodem 1 o 11:31 CEST, bez OOM, po około 26 h 37 min.

## Historia i metodologia

Wcześniejszy curriculum-v4 ukończył 10 rund na starej foundation 34M.
Nocny onpolicy-v1 z 8 października również wystartował z 34M, ukończył jedną
pełną rundę i SFT drugiej; został zatrzymany 9 października na polecenie
użytkownika, a dane/checkpointy zachowano. Nie jest częścią wyniku 100M.

Poprawny przebieg `curriculum-100m-onpolicy-v1-20261009` wystartował
9 października 08:54 CEST z istniejącej foundation
`human-pretrain-100m-20260730`, SHA256
`b660b1f8eef7104123dd6551eff5c63ab3b665b72a8f1a8b2d17ec6fd71c7e81`.
Nie trenowaliśmy ponownie pretrainu. Plan: 10 disjoint shardów z 9 688 373
teacher records, SFT jeden epoch na shard, następnie 96 884 przykładów
on-policy-selection DAgger i teacher replay 75/25. Własny dokładnie wygenerowany
trace jest kontekstem, loss DAggera tylko na final move wybranym przez SF30k
spośród sampled roots. Teacher compact SFT ma zwykły ważony loss trace/final.
Nie jest to pełny game-state-rollout DAgger ani RL. Fixed eval po każdym etapie,
pretrain tylko human/direct. Kod i konfiguracja przypięte per run.

## Wszystkie ukończone ewaluacje 100M

Wartości w %. SF eval 1000 pozycji, raw format 100 przykładów.

| Etap | Teacher exposures | DAgger exposures | Human top1 | SF direct | SF thinking | Recall | Selection | Selection przy obecnym best | Format error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pretrain | 0 | 0 | 53.70 | 44.10 | — | — | — | — | — |
| SFT 1 | 968838 | 0 | 52.28 | 50.90 | 39.00 | 71.20 | 47.00 | 54.49 | 0.00 |
| DAgger 1 | 1259490 | 96884 | 52.00 | 50.60 | 45.20 | 73.70 | 53.10 | 60.52 | 0.00 |
| SFT 2 | 2228328 | 96884 | 51.51 | 50.70 | 39.20 | 72.70 | 49.50 | 53.92 | 0.00 |
| DAgger 2 | 2518980 | 193768 | 51.22 | 51.80 | 44.10 | 72.80 | 55.00 | 59.89 | 0.00 |
| SFT 3 | 3487818 | 193768 | 50.93 | 51.20 | 42.60 | 72.60 | 52.30 | 58.68 | 0.00 |
| DAgger 3 | 3778470 | 290652 | 50.58 | 51.10 | 46.10 | 71.60 | 56.60 | 64.11 | 0.00 |
| SFT 4 | 4747307 | 290652 | 50.30 | 51.80 | 42.50 | 72.30 | 53.40 | 58.78 | 0.00 |

## Wnioski

- Trzy pełne rundy z ewaluacjami i SFT czwartej ukończone; DAgger 4 nie
  wytrenowany. Jego 96 884 konteksty wygenerowano i oznaczono SF, zachowane.
- Po DAgger 3 direct poprawia się o 7.0 pp vs foundation, human spada
  o 3.12 pp. Thinking 46.1% pozostaje 5.0 pp poniżej direct 51.1%.
- DAgger poprawia thinking względem poprzedzającego SFT o 6.2, 4.9 i 3.5 pp,
  ale kolejny SFT cofa część poprawy. To sygnał konfliktu lub zapominania
  pomiędzy etapami, nie rozstrzygnięta przyczyna bez kontrolowanego ablation.
- Recall ~72% i conditional selection 64.11% wskazują jednocześnie problem
  pokrycia kandydatów i wyboru spośród nich. Brak dowodu, że sama ilość danych
  jest główną przyczyną. Format nie wygląda na dominujący problem na badanej
  próbce; 0/100 nie daje gwarancji bezbłędności.
- Gry nowego 100M nie zostały uruchomione: automatycznie dopiero po 10 rundach.
  Historyczne gry i wyniki 34M/v4 nie są matched comparison ani wynikiem 100M.

## Zatrzymanie techniczne

`round-004/dagger/train.log`: GPT.load -> torch.load(map_location=cuda)
kończy się `torch.cuda.is_available() is False`. Host `nvidia-smi` zgłasza
Driver/library version mismatch (biblioteka 595.99, wcześniej odczytany
załadowany kernel driver 595.91.07). Problem pojawił się po systemowej
aktualizacji NVIDIA; istniejący proces generowania jeszcze działał, ale kolejny
proces treningowy już nie zainicjalizował CUDA. Nie jest to błąd formatu danych
ani OOM. Nie zmieniono protokołu, źródeł, sterownika ani nie restartowano hosta.

Przed wznowieniem trzeba naprawić zgodność NVIDIA w uzgodnionym oknie,
uwzględniając nadal działający osobny CPU job. Następnie zweryfikować CUDA
i wznowić od zachowanego SFT 4 oraz gotowych danych DAggera 4, bez ponownego
generowania wcześniejszych rund. Poprzednie ETA nie obowiązuje podczas postoju.

Źródła: docker inspect/logs, status.json, report.md oraz aktywne logi z root
`/home/ijakus/projects/krasnal-think/experiments/curriculum-100m-onpolicy-v1-20261009`
na Adrianie, odczyt na żywo 10 października 12:06 CEST. Szczegółowy audyt
foundation/protokołu: `reports/dagger-100m-20261009-run-audit.json`.
