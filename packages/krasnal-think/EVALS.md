# Ewaluacje

Aktywny pipeline v4 i jego stały protokół ewaluacji opisuje
[docs/curriculum.md](docs/curriculum.md). Poniższe wyniki i bramki v2/v3 są
historyczne; v4 wykonuje wszystkie 10 rund, zatrzymując się przy błędzie lub
human-top1 poniżej 40%, a nie przy plateau thinking.

Ewaluacje mają odpowiadać na trzy pytania: czy model zachował umiejętność
przewidywania ludzi, czy nauczył się poprawnie myśleć oraz czy thinking faktycznie
poprawia jakość ruchów. Tanie metryki liczymy na stałym zbiorze walidacyjnym w
trakcie treningu. Drogą drabinkę gier uruchamiamy dopiero po przejściu tych testów.

| Metryka | Co mierzy | Idealny wynik | Koszt |
| --- | --- | ---: | --- |
| `human-top1-acc` | Odsetek pozycji, w których najbardziej prawdopodobny ruch jest dokładnie ruchem człowieka z partii. | 100% | niski |
| `human-nll` | Negative Log Likelihood ruchów ludzkich. Uwzględnia całą przewidywaną dystrybucję, a nie tylko argmax. | 0 | niski |
| `stockfish-top1` | Odsetek pozycji, w których najbardziej prawdopodobny ruch jest dokładnie pierwszym wyborem Stockfisha. | 100% | niski, jeśli używamy zapisanych anotacji |
| `legal-move-ppb` | Łączna masa prawdopodobieństwa przypisana ruchom legalnym przed nałożeniem maski legalności. | 100% | niski |
| `format-error-rate` | Odsetek generacji, w których wystąpił przynajmniej jeden błąd formatu. Każdy przykład daje 0 albo 1. | 0 | niski |
| `avg-centipawn-thinking-improvement` | Średnio o ile centypionów ruch po thinking jest lepszy od ruchu bez thinking dla tej samej pozycji i tego samego modelu. Wynik ujemny oznacza szkodliwy thinking. | im większy dodatni, tym lepiej | niski lub średni |
| `pv-diversity-first-move` | Średni odsetek par PV różniących się pierwszym ruchem. | zakres [0, 1] | niski |
| `pv-diversity-second-move` | Średni odsetek par PV różniących się drugim ruchem, niezależnie od pierwszego. | zakres [0, 1] | niski |
| `pv-diversity-third-move` | Średni odsetek par PV różniących się trzecim ruchem, niezależnie od wcześniejszych. | zakres [0, 1] | niski |

`human-*` służy do oceny modelowania ludzkiej gry i jako kontrola zapominania po
SFT/RL. `stockfish-*` służy do oceny siły gry. Nie należy porównywać wartości tych
dwóch rodzin tak, jakby mierzyły ten sam cel.

Domyślny guardrail SFT zatrzymuje trening dopiero poniżej 40%
`human-top1-acc`. Spadek względem pretraina nadal raportujemy, ale nie blokuje on
eksperymentu, jeśli jednocześnie wyraźnie rośnie jakość względem Stockfisha.

## Błędy formatu

Oprócz zbiorczego `format-error-rate` raportujemy średnią każdego binarnego typu
błędu jako `format-error-<typ>`. Jedna generacja może mieć kilka błędów, dlatego
ich wartości nie muszą sumować się do `format-error-rate` ani do 1.

Podstawowe typy:

- `format-error-budget-compliance` — thinking przekracza maksymalny budżet
  zadany przez `<effort:...>`. Budżet jest limitem, nie wymogiem wykorzystania
  dokładnej liczby tokenów lub PV;
- `format-error-think-opening` i `format-error-think-closing` — brak lub błędne
  użycie `<think>` / `</think>`;
- `format-error-pv-opening` i `format-error-pv-closing` — niepoprawne granice PV;
- `format-error-pv-count` — liczba PV jest niedozwolona dla danego budżetu;
- `format-error-pv-length` — PV przekracza długość dozwoloną przez budżet;
- `format-error-evaluation` — brak, nadmiar lub zły rodzaj tokenu oceny PV;
- `format-error-token-order` — tokeny występują w kolejności niezgodnej z
  gramatyką.

Legalność ruchów wewnątrz PV raportujemy oddzielnie jako
`format-error-illegal-pv-move`. Liczymy ją na surowej generacji, zanim zewnętrzna
szachownica lub maskowanie skoryguje zachowanie modelu.

## Różnorodność PV

Raportujemy osobno różnicę na pierwszym, drugim i trzecim ruchu. Dla każdej
głębokości metryka jest średnią po wszystkich nieuporządkowanych parach PV, które
mają ruch na tej głębokości. Różnica na wcześniejszym ruchu nie wpływa na wynik
dla późniejszego ruchu. Każda metryka mieści się w `[0, 1]`: 0 oznacza ten sam
ruch we wszystkich porównanych PV, a 1 — różne ruchy w każdej parze. Jeśli nie ma
co najmniej dwóch PV odpowiedniej długości, wynik jest `N/A`.

Sama różnorodność nie jest celem: losowe, słabe PV mogą mieć wynik 1. Na razie
nie definiujemy `pv-quality`; wrócimy do niej dopiero, gdy będziemy mieli prostą,
jednoznaczną metrykę, która pomaga podejmować decyzje.

## Pomiar podczas treningu

- Używamy stałych, wersjonowanych próbek walidacyjnych i tych samych pozycji dla
  wszystkich checkpointów.
- Tanie metryki teacher-forced liczymy przed SFT oraz regularnie w trakcie
  treningu, zapisując
  `step`, czas, loss i wszystkie wyniki w prostym pliku tabelarycznym. Na tej
  podstawie rysujemy przebiegi metryk w czasie.
- Metryki formatu wymagają osobnej surowej generacji. Liczymy je co kilka tysięcy
  kroków na małej, stałej próbce, a nie w każdym batchu treningowym.
- `avg-centipawn-thinking-improvement` porównuje thinking i brak thinking parami
  na dokładnie tych samych pozycjach.
- Raportujemy liczebność próby. Dla metryk stochastycznych zapisujemy seed i, gdy
  ma to znaczenie, przedział ufności.

## Kolejność decyzji

1. `human-top1-acc` i `human-nll`: czy SFT nie zniszczył pretraina.
2. Metryki formatu i `legal-move-ppb`: czy model opanował protokół bez pomocy
   masek.
3. `stockfish-top1`: czy ruchy są mocniejsze.
4. `avg-centipawn-thinking-improvement`: czy poprawa rzeczywiście pochodzi z
   thinking.
5. Metryki `pv-diversity-*`: czy model rozważa różne linie.
6. Dopiero po dobrych tanich wynikach uruchamiamy kosztowną i wolną drabinkę gier.

Priorytety eksperymentu pozostają równoległe: możliwie dobre dane, szybki i
stabilny trening, wysoka jakość uczenia oraz wykorzystanie jak największej ilości
danych. Nie maksymalizujemy liczby przykładów kosztem ich jakości ani szybkości
kosztem braku wiarygodnej ewaluacji.

## Bramka automatycznego pipeline'u

Każdy paired eval korzysta z tych samych pozycji dla direct i thinking i
jednocześnie zapisuje on-policy roots do następnego shardu DAgger. Stockfish
niezależnie wyznacza globalny najlepszy ruch oraz relabeluje wszystkie delty.
Raport zawiera: direct/thinking top-1, recall globalnego best w distinct roots,
liczbę unikalnych roots oraz selection accuracy (czy ruch końcowy jest najlepszym
z faktycznie wygenerowanych kandydatów).

DAgger nie dostarcza przykładów direct ani effort 0. Shard jest skończonym
strumieniem i może zostać użyty najwyżej w 1--2 epokach; początkowy miks podczas
wspólnej części strumieni wynosi 75% teacher / 25% DAgger. Każdy checkpoint jest
porównywany parą z poprzednim na stałej próbce. Regresja thinking, direct,
selection, formatu lub human-top1 zatrzymuje iterację zamiast automatycznie
uruchamiać następny cykl.

Game ladder pozostaje zablokowany, dopóki thinking nie przebija direct w
powtarzalnych paired evalach, raw format jest praktycznie bezbłędny, candidate
recall i selection działają, a human-top1 pozostaje co najmniej 40%.

Bezstronny baseline dla iteracji pipeline-v2 to sft-v4 na 1,000 pozycjach przy
30k nodes i bez best-root injection: direct 49.4%, thinking 37.0%, candidate
recall 73.7%, selection 42.8%, średnio 3.958 distinct roots. PV diversity na
plies 1/2/3 wynosi 100% / 64.65% / 80.41%. Human-top1 wynosi 50.565%. Dla raw
format effort 1/2/4 mają 0% błędów, effort 3 ma 17%. Wcześniejszy wynik recall
78.5% był obciążony 30% injection i pozostaje wyłącznie diagnostyczny. Następny
checkpoint musi być oceniony na tej samej próbce i przebić bezstronne wartości
bez naruszenia guardraila.

## Teacher SFT v5 (2026-09-15)

Jedna epoka na 1,132,101 nowych poprawnych rekordach teacher (165,576 extra oraz
966,525 rekordów z czterech shardów UWR) nie przebiła baseline'u. Wyniki paired
eval na tych samych 1,000 pozycjach i przy 30k nodes:

| Checkpoint | Direct top1 | Thinking top1 | Recall | Selection | Distinct roots |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sft-v4` baseline | 49.4% | 37.0% | 73.7% | 42.8% | 3.958 |
| teacher-v5 step 5,000 | 47.5% | 37.6% | 70.7% | 44.2% | 4.313 |
| teacher-v5 final | 45.1% | 32.1% | 61.3% | 41.7% | 4.061 |

PV diversity dla step 5,000 wyniosła 100% / 65.20% / 80.53%, a dla finalnego
checkpointu 100% / 60.19% / 80.25% na plies 1/2/3. Human-top1 podczas treningu
pozostał bezpiecznie nad guardrailem 40% (finalnie 50.84%), więc regresja nie
wynika z katastroficznego zapominania mierzonego tym guardrailem. Więcej teacher
SFT poprawiało teacher-forced NLL, ale pogarszało zachowanie on-policy.

Decyzja: nie promujemy żadnego checkpointu v5. Step 5,000 jest wynikiem mieszanym,
nie stabilnym zwycięzcą; pełny v5 jest regresją. Następna iteracja ma startować od
`runs/sft-v4/model.pt` i celować w generowanie/selekcję on-policy, z częstym paired
evalem. DAgger oraz game ladder pozostają zablokowane do czasu czystej poprawy.
## Pipeline v3 schedule (2026-09-15)

Clean root: `runs/pretrain/model.pt`; `runs/sft-v4/model.pt` is comparison-only.
Teacher stage is capped at 5,000 steps. Each DAgger round uses 10,000 fresh
on-policy contexts and one 500-step 75/25 teacher/DAgger pass. Paired metrics are
collected before every mixed SFT, so the next round is automatically stopped on
regression. The game ladder remains gated until thinking reliably beats direct,
selection works, format errors are negligible, and human top-1 stays above 40%.

### Pipeline v3 paired results (2026-09-17)

All rows use the same 10,000-position source-offset sample.

| Checkpoint | Direct top1 | Thinking top1 | Recall | Selection |
| --- | ---: | ---: | ---: | ---: |
| control-01 | 46.23% | 39.43% | 70.40% | 47.65% |
| control-02 | 46.63% | 39.17% | 70.50% | 47.43% |
| DAgger-01 / `sft-01` | 47.01% | 40.50% | 69.81% | 49.75% |
| DAgger-02 / `sft-02` | 47.09% | 41.44% | 68.31% | 52.07% |

DAgger improves thinking and selection versus matched controls, but thinking is
not yet above direct, so the game ladder remains gated.

| DAgger-03 / `sft-03` | 47.99% | 42.40% | 67.64% | 53.69% |

DAgger-03 is the best thinking/selection checkpoint so far; its thinking-vs-direct
mean CP difference is -44.65. `sft-04` is now training from this checkpoint.

| DAgger-04 / `sft-04` | 48.51% | 44.01% | 67.87% | 55.79% |

DAgger-04 further improves all on-policy decision metrics; thinking remains below
direct (mean CP difference -32.11). `sft-04` human-top1 is 51.85%; a small raw
format probe showed roughly 9% effort-2 errors, so format is still monitored.

| DAgger-05 / `sft-05` | 48.96% | 44.04% | 67.84% | 55.92% |

DAgger-05 is non-regressive but thinking gains have plateaued (+0.03 pp); direct
and selection improved slightly. Do not continue SFT mechanically without a
material paired or format improvement.
