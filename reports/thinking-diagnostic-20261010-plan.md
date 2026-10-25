# Ograniczona diagnostyka thinkingu 100M

Etap A uruchomiony 10 października 2026. Adrian: kontener
`kt-thinking-diagnostic-a-20261010`, root `experiments/thinking-diagnostic-20261010`.
Pinned code w root/code; nie wznawia starego curriculum.

Wspólny checkpoint DAgger3 SHA256
`086781ca9846cb4216e2b1d48781f9a7bba890fa4f13bcc5f70f2d578d684025`.
Foundation100M SHA256b660b1f8eef7104123dd6551eff5c63ab3b665b72a8f1a8b2d17ec6fd71c7e81
sprawdzana przed dispatch treningu.

A: baseline eval oraz dwa niezależne ramiona po1000updates/batch64,
ten sam pierwszy podzbiór73728records next shard round-03.jsonl,
efforts0/2 seed42, optimizer kontynuowany z identycznego checkpointu,
RNG identycznie resetowany seed42. LR5e-6→5e-7 warmup50.
Teacher trace CE normalizowany wagami per-example vs final-only CE.
Final-only maskowanie odbywa się w loss, więc wejścia/kolejność/bucketing
pozostają identyczne. Nie zamrażamy parametrów modelu. Smoke na128records,
effort2,batch4,1update przeszedł CUDA; loss0.3276=final move loss,
thinking diagnostic NaN oczekiwane przy braku thinking targets.
Nie dodano unit testów. Python compilation i source hash comparison przeszły.

Fixed eval: niezmieniony gold1000pozycji, human30batches seed1729,
rawformat100, SF30k. CPU label4workers, bez przerywania istniejącego CPUjobu.
Każde ramię max30min treningu, globalny deadline6h zapisany raz w plan.json;
wznowienie nie resetuje deadline. Kod/config/data/foundation SHA w plan.json.

Etapy B (mix50/40/10) i C (mały full-trace RL pilot) nie są jeszcze
zaimplementowane/uruchomione przez driver A. Automatyzacja
`test-sft-on-policy-mix-pilot-rl` kontynuuje po odczycie wyników A.
Żadnego rozszerzania budżetu ani automatycznego wielodniowego treningu.

## Naprawa techniczna 10 października 16:24 CEST

Pierwszy start zakończył się o14:30CEST przed treningiem przez brak
`scripts/pipeline.py` w snapshot (import curriculum_eval→curriculum→pipeline).
Uzupełniono zależność SHA256
`01ee63f5fff1ef39ef55ff3f775824d258c3050e717dbec709df04764b261ab0`.
Container import smoke wszystkich entrypoints przeszedł. Driver wznowiony
w tym samym kontenerze; gotowa kolekcja baseline1000 pozostaje bez nadpisania,
oryginalny plan/deadline pozostaje niezmieniony. Repair audit zapisuje aktualne
source SHA obok pierwotnego planu. Nie zmieniono loss, danych ani budżetu.
