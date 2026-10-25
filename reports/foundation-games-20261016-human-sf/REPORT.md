# Elo z rzeczywistych gier — raport końcowy

**Ukończono 800/800 partii, po 200 na system.** Gry zakończyły się 16 października
2026 o 09:54 CEST. Wszystkie PGN niezależnie zweryfikowano po odzyskaniu połączenia
z Adrianem 17 października. Brak partii cenzurowanych, brak cutoffów liczonych jako remisy.

## Wyniki

| System | W / D / L | Rating w skali testu | Przybliżony 95% bootstrap CI | Konserwatywny 95% CI |
|---|---|---:|---|---|
| Foundation100M, główka ruchów | 65 / 37 / 98 | **1935** | 1845–2031 | 1663–2207 |
| Ten sam foundation, wybór przez główkę eval | 58 / 42 / 100 | **1890** | 1790–1991 | 1617–2162 |
| Human–SF, główka ruchów z rzeczywistymi podpowiedziami SF | 90 / 46 / 64 | **2231** | 2130–2327 | 1959–2506 |
| Sam Stockfish30k, kontrola | 197 / 3 / 0 | Poza zakresem drabinki | 3357–∞ | 2844–∞ |

**To NIE jest Elo FIDE ani Lichess.** Rating oznacza dopasowany odpowiednik etykiety
UCI_Elo przeciwnika przy tym konkretnym budżecie 20k nodes. Przedziały nie uwzględniają
nieznanego błędu kalibracji tych etykiet, zmiany puli przeciwników ani błędu modelu
logistycznego. Dodatkowa cyfra dokładności nie usuwa tych ograniczeń.

Kontrola SF30k uzyskała punktową ekstrapolację 3483, lecz najwyższy przeciwnik miał
etykietę 2900, a górny kraniec CI jest nieskończony. Nie traktować 3483 jako precyzyjnej
siły silnika ani nie porównywać tego z ludzkimi rankingami.

## Co rzeczywiście wynika z porównania

- System human–SF z podpowiedziami przewyższył foundation policy o **295 punktów**;
  przybliżony sparowany 95% bootstrap CI różnicy: **+165 do +421**.
- Względem foundation eval różnica wyniosła **+340**, CI **+200 do +477**.
- Eval selector minus policy: **−45**, CI **−186 do +96**. Nie wykazano przewagi
  gry przez główkę eval, mimo wyższego SF-top1 we wcześniejszym teście wyboru ruchów.

Porównania są po tych samych parach otwarć i wspólnej drabince, nie bezpośrednimi
meczami między systemami. Human–SF ma dodatkową informację i dodatkowy compute:
**prawdziwą podpowiedź Stockfisha30k przed każdym półruchem**, także historycznym
ruchem przeciwnika. To wynik wspomaganego systemu, nie samodzielnego modelu.
Nie można przypisać +295 samemu dotrenowaniu ani stwierdzić, że model osiągnął
2231 bez silnika. Sam podpowiadacz był zdecydowanie silniejszy w tej drabince.
Nie badano podobieństwa stylu gry do ludzkiego.

## Drabinka: W / D / L, 40 gier na komórkę

| UCI_Elo SF | Foundation policy | Foundation eval | Human–SF assisted | SF30k control |
|---:|---|---|---|---|
| 1320 | 32 / 3 / 5 | 25 / 8 / 7 | 36 / 2 / 2 | 40 / 0 / 0 |
| 1700 | 21 / 10 / 9 | 18 / 5 / 17 | 24 / 8 / 8 | 40 / 0 / 0 |
| 2100 | 7 / 9 / 24 | 9 / 11 / 20 | 20 / 9 / 11 | 40 / 0 / 0 |
| 2500 | 5 / 7 / 28 | 6 / 14 / 20 | 10 / 17 / 13 | 40 / 0 / 0 |
| 2900 | 0 / 8 / 32 | 0 / 4 / 36 | 0 / 10 / 30 | 37 / 3 / 0 |

## Zamrożona metodologia

100 różnych, nietransponujących pozycji po ośmiu półruchach, wybranych z historycznej
walidacji human przy seed20261015 bez filtrowania siłą/oceną. To nie niezależny
standardowy turniejowy opening book. Te same otwarcia dla każdego systemu;
obie barwy, 20 par na każdy z pięciu poziomów. Stała próba 200 gier na system,
bez zatrzymywania po osiągnięciu istotności.

Stockfish17.1, Threads1, Hash32MiB. Przeciwnicy: UCI_LimitStrength true i 20k
żądanych nodes/ruch; teacher/control: bez limitu siły i 30k nodes/ruch.
Neural inference FP32, conditioning2600/2600. Foundation używa effort0;
human–SF koduje sekwencję SF-token / faktyczny ruch bez effort0, zgodnie z treningiem.
Jego główka eval nie była uczona w tym fine-tuningu i nie jest tutaj używana.

Foundation value: wszystkie legalne dzieci, oczekiwana wartość sigmoid-transformowanych
reprezentantów binów CP, białe maksymalizują / czarne minimalizują; jawne dokładne
wyniki terminalne i policy tie-break. Ta transformacja nie jest kalibrowanym WDL.

Brak rezygnacji i adjudykacji na podstawie ocen. Reguły mat/pat/material/repetition/
move-count z claim_draw=True. Limit480 półruchów lub deadline oznaczałby partię
nieukończoną, nie remis. W tej serii nie było żadnej takiej partii.

Estymator ratingu: logistyczny quasi-MLE na wyniku W=1,D=0.5,L=0 i etykietach
przeciwników. Przybliżony CI: 10000 stratyfikowanych bootstrapów par otwarć,
seed20261015; zamienionych kolorów nie traktujemy jako niezależnych prób.
Sparowane różnice używają wspólnych indeksów resamplingu obu systemów.
Konserwatywny CI: Hoeffding dla ograniczonego wyniku pary, odwrócony przez tę samą
krzywą logistyczną. Oba zakładają niezależność par i model ratingowy; nie są
gwarancją przedziału dla nieznanego ratingu FIDE ani odpornością na błędy kalibracji.

## Weryfikacja i przebieg

Oryginalna seria ukończyła600 gier, po czym loader odrzucił historyczny alias
pathlib.PosixPath z checkpointu human–SF. Użytkownik zatwierdził dodatkowe45min
wyłącznie na naprawę i brakujące200 gier. Jawny bezpieczny alias naprawił zgodność
Pythona3.13; weights_only=True pozostało włączone. Smoke obu kolorów przeszedł.

Nowy root zachował identyczne600 rekordów parent runu, bez powtarzania partii.
Oryginalny snapshot, checkpointy i plan pozostały nietknięte. Gry human–SF trwały
łącznie1693s (28m13s); od utworzenia planu rozszerzenia do zakończenia1712s
(28m32s), około14m47s przed zatwierdzonym deadline’em10:09 CEST. Nie było treningu,
nowych zadań GPU po deadline ani ingerencji w CPU annotation/UWR.

Kontener zakończył się exit0, bezOOM, complete.reason=fixed_sample_complete.
Później połączenie SSH przerwało transfer PGN; częściowej kopii nie uznano za dowód
uszkodzenia gier. Po odzyskaniu dostępu pobrano całość i porównano wszystkie sześć
SHA256 z workerem. **Niezależnie odtworzono800 PGN**: legalność, końcowe FEN,
wynik zgodny z regułami, liczba półruchów i zgodność z JSON. Potwierdzono identyczność
pierwszych600 rekordów z parent runem i brak dodatkowych PGN po800.

## Artefakty i provenance

Adrian: `/home/ijakus/projects/krasnal-think/experiments/foundation-games-20261016-human-sf`.
Kontener `kt-foundation-human-sf-20261016`, ID
`b4f008b5af305dbddfbd9a42878eb4d65bcd4cadbb39ec4fab3e91f3e4999111`.

Foundation100M SHA256:
`b660b1f8eef7104123dd6551eff5c63ab3b665b72a8f1a8b2d17ec6fd71c7e81`.
Interleave SHA256:
`0605bf201b31397e50ed36705d4a5d3b8ae7f86607da1b45493f18b1c1544f33`.
Runner SHA256:
`798467624092237dd09c68a5fa925535e4c71ad7c2fbf29b752e30d637a5dd5b`.
Model source SHA256:
`be282f8cd104aacd07e297867ad11f7a7c276b39bcf3af44565acec1a60ec89b`.

Lokalne plan/result/status/complete JSON, gamesJSONL i pełny PGN: bieżący katalog.
Hash manifest: `SHA256.json`. Raport zamyka test; bez automatycznego zwiększania
próby ani nowego treningu.
