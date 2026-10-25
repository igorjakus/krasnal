# DAgger — ponowny start z właściwej foundation 100M

9 października 2026, Europe/Warsaw. Na wyraźne polecenie użytkownika zatrzymano
`kt-curriculum-onpolicy-v1-20261008`. Stary przebieg dotyczył foundation 34M,
SHA256 `52929453aaac2e62bb8ce1b61c084d7491e2dac462d10a1d3b64218e415b7f84`.
Jego dane, checkpointy, częściowo wygenerowane dane i raporty nie są usunięte.

## Nowy przebieg

- Foundation ID: `human-pretrain-100m-20260730`.
- Canonical path: `runs/foundations/human-pretrain-100m-20260730/model.pt`.
- Zweryfikowany SHA256:
  `b660b1f8eef7104123dd6551eff5c63ab3b665b72a8f1a8b2d17ec6fd71c7e81`.
- 100 000 000 training games, 100 756 992 parametrów, 29-class eval head,
  checkpoint step 106 439 (registry).
- Root: `experiments/curriculum-100m-onpolicy-v1-20261009`.
- Config: `configs/curriculum-100m-onpolicy-v1-20261009.json`.
- Kod przypięty w `experiments/curriculum-100m-onpolicy-v1-20261009/code`;
  uruchomienie używa tego snapshotu, nie współdzielonych edytowanych scripts/src.
- GPU count: 1, RTX 3090 na Adrianie; brak konkurencyjnego treningu GPU.

## Niezmieniony protokół treningu

10 rozłącznych shardów z 9 688 373 teacher przykładami; każdy shard pełne SFT,
fixed eval, 96 884 contexts DAgger, SF30k relabeling, DAgger + teacher replay,
fixed eval. Teacher replay 75/25 (290 652 teacher na rundę); końcowy udział
DAggera ~7.14% ekspozycji. Effort sampling teacher 0/2; generacja greedy effort2,
do 3 PV po 3 plies, delty 4 klasy. Teacher weighted CE jak wcześniej;
on-policy DAgger zachowuje dokładny własny trace i nadzoruje tylko final choice
najlepszego sampled root. Nie jest to pełny game-rollout DAgger ani RL.

Użyto kopii **tylko niezmiennych, zwalidowanych teacher shardów/ledger i fixed SF
gold** poprzedniego przebiegu: identyczne historie/holdout/hashy; nie przeniesiono
żadnych SFT/DAgger wag ani wygenerowanych danych DAggera. Pozwala uniknąć
ponownej walidacji milionów przykładów. Nowy launcher weryfikuje hashe źródeł
i rozpoznaje manifest przygotowania z identycznymi parametrami.

Baseliny i wszystkie ewaluacje nowego modelu liczymy od początku; wyników
historycznego 34M nie mieszamy z wynikami 100M. Pretrain baseline tylko
human/direct, bez thinking. Automatic games po ukończeniu (limit15min)
porównują final direct/thinking z **właściwą foundation100M** oraz SF1600,
nie ze starymi SFT02/SFT08. Kod, config, dataset i foundation mają manifesty;
launcher zapisuje foundation ID/hash/training_games/GPU count.

## Czas i status

Po zmianie foundation należy ponownie zmierzyć czasy. Dotychczasowy kolektor
przetwarzał ~6.1 contexts/s; przy tym samym protokole orientacyjny czas całości
pozostaje 55–60h od nowego startu, nie jedna noc.

Test techniczny `kt-curriculum-100m-smoke-retry1`: exit0, 1 passed,
3 deselected, 185.24s; cztery małe rundy oraz resume/append zakończone.
Pierwsza próba snapshotu nie rozpoczęła treningu z powodu brakującego helpera
`pipeline.py`; dodano dokładny plik kanoniczny i powtórzono test w nowym root.
Pełny kontener `kt-curriculum-100m-onpolicy-v1-20261009` uruchomiony około
08:54 CEST; automatycznie reuse przygotowanych danych, nowy baseline100M,
pełne10rund i bounded games. Monitor przepięty na nowy kontener; starego
przebiegu nie wolno wznawiać. Source hashes / sampling / GPU count i audyt
restartu: `reports/dagger-100m-20261009-run-audit.json`, kopia `run-audit.json`
na nowym worker root. Zmieniony po smoke nieużywany moduł interleave został
zsynchronizowany ze współbieżnie aktualizowanego canonical checkout przed
pełnym startem; algorytm SFT/DAgger był identyczny jak w smoke.

Nowy baseline100M zakończony: human top1 **53.70%**, direct SF top1 **44.10%**,
czas42s. Pipeline przeszedł do SFT rundy1; thinking pretrain nie oceniany.
