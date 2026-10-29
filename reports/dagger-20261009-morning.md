# SFT + on-policy DAgger — raport poranny

> Raport historyczny: ten przebieg korzystał z wycofanej foundation **34M gier**
> (SHA `52929453…415b7f84`), nie z 100M. Na polecenie użytkownika został
> przerwany 9 października. Jego wyniki nie są wynikami nowego przebiegu 100M;
> dane, checkpointy i oryginalne artefakty pozostają zachowane.

Stan na 9 października 2026, około 08:30–08:35 Europe/Warsaw.

## Podsumowanie

**Trening nie jest zakończony.** Ukończono pretrain baseline, SFT rundy 1,
DAgger rundy 1 oraz SFT rundy 2. Trwa generacja danych DAggera rundy 2:
około 58.5 tys. z 96 884 kontekstów. Kontener działa bez OOM/awarii, bez
konkurencyjnego zadania GPU na Adrianie. Nie zatrzymano ani nie zmieniono protokołu.

Pierwszy DAgger poprawił wybór końcowy po thinking: selection +7.9 pp,
thinking SF top1 +5.8 pp, recall +0.1 pp względem SFT tej samej rundy.
To obiecujący sygnał, nie końcowy dowód skuteczności. Human top1 spadł
o 0.30 pp, direct o 0.10 pp, raw format error wzrósł z 0/100 do 1/100.
Thinking nadal ma niższy SF top1 niż direct. Nie ma jeszcze gier nowego modelu.

**Wcześniejsza estymacja czasu była zbyt optymistyczna.** Zmierzony koszt
generacji on-policy wynosi około 4.4 godziny/rundę, CPU relabelingu około
50 minut, SFT około 18 minut, treningu DAggera około 7 minut; eval około
3.4 minuty/etap. Przy utrzymaniu tempa pełne 10 rund to około 55–60 godzin
od startu, czyli orientacyjnie niedziela 11 października rano (06:00–11:00).
To ekstrapolacja pierwszej rundy, nie gwarancja. Drugi pełny wynik DAggera
orientacyjnie dziś około 11:00–12:00.

## Metodologia

### Model i start

Start z ukończonego human pretrain, **nie z wybranego checkpointu SFT**.
Pretrain nie został ponownie trenowany tej nocy; importujemy jego artefakt.
Model: około 100.7M parametrów, 14 warstw, 12 heads, embedding 768,
context 1024, vocabulary 2112; foundation step 73 799.

SHA256 foundation: `52929453aaac2e62bb8ce1b61c084d7491e2dac462d10a1d3b64218e415b7f84`.

Worker: Adrian, RTX 3090 24 GiB, Docker `kt-curriculum-onpolicy-v1-20261008`,
start 8 października o 23:21:26 CEST. Root:
`/home/ijakus/projects/krasnal-think/experiments/curriculum-onpolicy-v1-20261008`.
Public config `configs/curriculum-clean.json`; config, code fingerprints,
foundation i źródła przypięte w manifestach workera.

### Dane i rozłączne rundy

Walidacja legalności historii/PV oraz długości context, deduplikacja po dokładnej
historii ruchów w SQLite. Rozłączne historie między shardami i holdout;
nie gwarantuje rozłącznych FEN/transpozycji ani gier i niezależności od pretrain.

| Licznik przygotowania | Rekordy |
|---|---:|
| Wejściowe z 10 jawnych źródeł | 9 788 260 |
| Duplikaty | 97 887 |
| Odrzucone | 0 |
| Fixed test | 1 000 |
| Teacher validation | 1 000 |
| SFT train | 9 688 373 |
| Rozłączne shardy | 10 po 968 837–968 838 |

Adrian ma 6.7 GiB w katalogu SFT, UWR 2.1 GiB; to rozmiary katalogów,
nie liczba unikalnych przykładów. Osiem dużych incoming teacher plików
(2019 cztery po 250k, 2020–2023 cztery po 1M) ma identyczne SHA256 na obu
workerach. Nie kopiowano ich ponownie ani nie liczono dwukrotnie.
UWR ma dodatkowe dane SF-interleave w innym formacie; nie włączono ich do tego
protokołu. Aktywne inne zadania UWR pozostawiono bez zmian.

### SFT i DAgger

Każda runda: pełna epoka SFT na nowym shardzie → fixed eval → wygenerowanie
DAgger contexts przez bieżący model → Stockfish relabel → epoka mieszana
DAgger + teacher replay → fixed eval → kolejny shard.

Teacher SFT: effort 0/direct i 2/thinking; do 3 różnych korzeni PV po 3 plies;
cztery klasy delty (≤5, 6–40, 41–180, >180 cp). Teacher trace zawiera najlepszy
korzeń i do dwóch alternatyw w losowej kolejności. Wagi CE: final move 1,
PV 0.5/0.25/0.125, grammar/delta 0.1; normalizacja per przykład.

DAgger: `ceil(0.1 × shard_count)` = **96 884 kontekstów na rundę**.
Model generuje korzenie, krótkie PV i delty greedy; zachowujemy dokładne
effort/PV/delta/closing tokens jako context. Stockfish 17.1, 30k nodes,
wybiera najlepszy z zaproponowanych korzeni. Nie wstrzykujemy globalnego
najlepszego ruchu spoza kandydatów. Trace nie jest zastępowany trace'em
Stockfisha; **loss DAggera dotyczy tylko expert final move**.
Eksperckie PV pozostają w dodatkowych polach diagnostycznych rekordu.
Maskowanie trace nie zamraża parametrów modelu — gradient końcowego wyboru
nadal przepływa przez reprezentację context.

Teacher replay: 290 652 rekordy na rundę, czyli 3 teacher na 1 DAgger.
W mixed pass udział DAgger to 25%; w całym przebiegu około 7.14% ekspozycji.
SFT i DAgger w tej samej rundzie celowo korzystają z historii bieżącego shardu,
nie z przyszłych rund ani testu. Terminal/empty-root contexts są zastępowane
kolejnymi z tego shardu; w rundzie 1 pominięto 51 takich contexts.

To **candidate-selection DAgger na on-policy trace w teacher contexts**,
nie pełne self-play/state-rollout DAgger ani RL. Teacher SFT/replay dalej
uczą PV/delt; nowy DAgger uczy wykorzystania własnego trace przy final choice.

Trening: batch 64, Adam state przenoszony między etapami; cosine LR 5e-6
do 5e-7, warmup 50 na etap. Atomic checkpoints pozwalają przywrócić optimizer,
RNG i batch cursor. Nie zmieniono LR/modelu/budżetu w trakcie przebiegu.

## Stała ewaluacja

Te same 1000 test histories po każdym SFT i DAgger; global SF gold zamrożony
raz, Stockfish 17.1, 30k nodes, hash 32 MiB/thread 1/clear hash per position.
Gold SHA256: `b347129b66483430289dde08f3fca0020c8b7b5158168fd6aad69f64f5ded319`,
identyczny jak v4. Human top1: stałe 30 batchy, seed 1729, validation stream
pretrain. Format: unmasked generation na stałych pierwszych 100 pozycjach.
Pretrain oceniany tylko human/direct: thinking i format są nie dotyczy.

- SF direct/thinking top1: zgodność końcowego ruchu z zamrożonym global best.
- Recall: global best obecny w wygenerowanych korzeniach.
- Selection: final move zgadza się z najlepszym SF-ocenionym kandydatem.
- Conditional selection: global best wybrany, gdy znajdował się w kandydatach.
- Format error: udział surowych generacji z błędem protokołu.

## Wyniki wszystkich ukończonych etapów

| Etap | Teacher exposures | DAgger exposures | Human top1 | SF direct | SF thinking | Recall | Selection | Conditional selection | Format error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pretrain | 0 | 0 | 52.98% | 41.60% | — | — | — | — | — |
| SFT 1 | 968 838 | 0 | 51.70% | 47.50% | 35.80% | 68.30% | 45.40% | 52.12% | 0.00% |
| DAgger 1 | 1 259 490 | 96 884 | 51.40% | 47.40% | 41.60% | 68.40% | 53.30% | 60.38% | 1.00% |
| SFT 2 | 2 228 328 | 96 884 | 50.91% | 49.20% | 41.50% | 71.30% | 51.50% | 58.06% | 0.00% |

Exposures obejmują ponowne użycie teacher replay, nie tylko unikalne dane.
Z planowanych 21 wierszy raport ma 4. Brak wyniku DAgger 2 i dalszych etapów.

## Porównanie z historycznym v4

| Historyczny etap v4 | Human | SF direct | SF thinking | Recall | Selection | Format error |
|---|---:|---:|---:|---:|---:|---:|
| DAgger 1 (10k) | 51.67% | 47.90% | 35.70% | 68.90% | 45.20% | 0.00% |
| SFT 2 | 51.05% | 49.10% | 41.90% | 71.80% | 52.10% | 0.00% |
| DAgger 2 (20k cumulative) | 51.00% | 48.50% | 42.10% | 70.50% | 55.30% | 0.00% |
| Final DAgger 10 | 47.84% | 50.10% | 42.60% | 71.80% | 52.00% | 0.00% |

Nowy DAgger 1 vs v4 DAgger 1: thinking +5.9 pp, selection +8.1 pp,
recall −0.5 pp, human −0.27 pp, direct −0.5 pp. Budżet DAggera jest jednak
9.69× większy i zmieniono objective, więc **nie izoluje to wpływu nowej
metodologii**. Po kolejnym SFT 2 thinking jest zbliżony do v4 (−0.4 pp),
nie ma jeszcze dowodu trwałej przewagi. Final v4 pokazano jako odniesienie,
nie jako dopasowane porównanie do ukończonego nowego treningu.

## Zasoby, czas i bottleneck

| Czynność rundy 1 | Zmierzony/obserwowany koszt |
|---|---|
| Przygotowanie pełnych danych i baseline | około 22 min od startu do SFT |
| SFT | około 18.3 min; GPU około 96%, 19.5 GiB |
| Eval SFT | 203 s |
| Generowanie 96 884 on-policy contexts | około 4.4 h; około 6.1 contexts/s, GPU 24–28%, 2.2 GiB |
| Stockfish relabel | 2975 s = 49.6 min; 32.56 contexts/s, CPU 12 workers |
| Trening DAgger/replay | około 7.3 min; GPU obserwowane około 84%, 15.7 GiB |
| Eval DAgger | 204 s |

Generacja jest sekwencyjna, kontekst po kontekście, i dominuje nad uczeniem.
Niska zajętość GPU to realna niewydajność kolektora; nie oznacza awarii
ani konkurencyjnego zadania. CPU relabeling z bezczynnym GPU to oczekiwany
etap obecnego pipeline. Dalsza optymalizacja powinna dotyczyć batched generation
i zachowania równoważnych masek/context/trace, a nie ukrytego obcięcia budżetu.
Nie wdrażano zmiany kolektora do aktywnego przebiegu przy normalnym postępie.

## Gry, ograniczenia i następne kroki

Nowych gier jeszcze nie rozegrano — automatyczny ekran startuje dopiero po
ukończeniu treningu, z limitem 15 min i bez konkurencyjnego GPU. Wyników gier
historycznego v4 nie przedstawiamy jako wyników nowej wersji.

Jedna pełna runda nie wystarcza do oceny trwałego efektu. Test 1000 pozycji
nie jest precyzyjnym Elo; raw format 100 pozycji ma małą próbę (1% to jeden
błąd, 0% nie dowodzi bezbłędności). Dedupe po historii nie rozwiązuje pełnej
kontaminacji gier/pretrain. Dwa równoczesne zmienione czynniki (objective i
budżet) wymagają w przyszłości dopasowanej ablation przed wnioskiem przyczynowym.

Trening i półgodzinny monitor pozostają aktywne. Czekamy na DAgger 2 oraz
kolejne rundy, sprawdzając selection, human retention i format. Nie uruchomiono
drugiego eksperymentu i nie ingerowano w aktywne zadania UWR.

## Reprodukowalność i artefakty

Lokalnie 20 tests passed / 1 skipped oraz kontrola maskowania trace; realny
GPU smoke start/resume/append: exit 0, 180.91 s, cztery małe rundy ukończone.
Stare protokoły/dane/checkpointy zachowane. Źródłem wyników są raporty i
manifesty workera, nie ręcznie estymowane metryki.

Dziewięć małych plików pobrano do `reports/dagger-20261009-artifacts/`.
SHA256 każdego porównano z Adrianem; lista źródeł i hashy w
`sha256-manifest.json`. Nie pobierano zbiorów treningowych ani checkpointów.
Pełniejsze szczegóły implementacji: `docs/curriculum.md`;
pierwotny opis planu: `reports/dagger-20261009-methodology.md`.
