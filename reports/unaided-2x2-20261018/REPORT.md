# Czysty test2×2: oba modele i obie główki, bez podpowiedzi

**800/800 świeżych partii ukończonych i niezależnie zweryfikowanych.** Po200 gier
w każdej komórce,100 par z zamianą kolorów. Stockfish występował wyłącznie jako
przeciwnik. **Zero wywołań teachera, zero tokenówSF na wejściu, zero gier cenzurowanych.**

## Cztery wyniki

W komórce: punktowa estymata ratingu i przybliżony95% bootstrap CI.

| Model, bez podpowiedzi | Główka ruchów | All-legal child eval selector |
|---|---|---|
| Foundation100M | **1966** [1885,2046] | **1925** [1820,2031] |
| Human–SF fine-tune, plain-history ablation | **≈1037*** [936,1117] | **≈662*** [−∞,827] |

**To nie Elo FIDE ani Lichess.** Skala to dopasowany odpowiednik etykiety UCI_Elo
przeciwnika przy20k nodes/ruch. Przedziały nie obejmują błędu kalibracji etykiet ani
błędu przyjętego modelu logistycznego. Nie gwarantują ludzkiego ratingu.

\*Obie dolne estymaty są **ekstrapolacjami poniżej najniższego przeciwnika1320**,
przy zerze zwycięstw. Nie podajemy ich jako precyzyjnie zmierzonej siły1037 lub662.
Wąski bootstrap1037 nie rozwiązuje problemu ekstrapolacji i model misspecification.
Dla główki eval część resampli ma zero punktów, więc dolny kraniec pozostaje
nieograniczony zamiast być arbitralnie przycinany do zera.

### Ostrożniejsze95% przedziały: Hoeffding po parach otwarć

| Model | Główka ruchów | Główka eval |
|---|---|---|
| Foundation100M | [1693,2237] | [1653,2197] |
| Human–SF bezSF | [−∞,1437] | [−∞,1364] |

Te przedziały również są warunkowe względem niezależności par i logistycznego modelu
ratingowego. Brak skończonej dolnej granicy uczciwie pokazuje ograniczenia tej próby.

## Co można powiedzieć

1. **Foundation działa samodzielnie w tym formacie, na poziomie około1900–2000
   w umownej skali testu.** Główka eval minus policy:−40 punktów, sparowany przybliżony
   95% bootstrap CI **−176 do+96**. Nie wykazano przewagi jednej metody.
2. **Human–SF po prostym usunięciu podpowiedzi radzi sobie bardzo słabo:** ani jednego
   zwycięstwa w obu ramionach. Policy uzyskała15 remisów, eval tylko2. Ten checkpoint
   w badanym formacie nie wykazał zdolności do silnej samodzielnej gry.
3. To **nie dowodzi, że fine-tuning bezwarunkowo usunął wiedzę szachową**. Model
   trenowano na naprzemiennym wejściu SF/ruch, a tutaj usunięto wszystkie slotySF.
   Zmieniono rozkład wejścia. Wynik testuje również zależność od tej informacji
   i odporność na jej brak, nie izolowany przyczynowy wpływ samego treningu.
4. Fine-tune human–SF nie trenował główki eval, choć zmieniał wspólny trunk.
   Nie było podstaw do zakładania, że jej dawna kalibracja zostanie zachowana.
5. Historyczne2231 dotyczyło **modelu z prawdziwymi podpowiedziamiSF** i nie należy
   do tej tabeli. Wszystkie cztery obecne wyniki są nowe, bez przepisywania starych
   foundation1935/1890 do bieżącej próby.

## W/D/L i drabinka

| System | Zwycięstwa | Remisy | Porażki | Punkty /200 |
|---|---:|---:|---:|---:|
| Foundation policy |68|37|95|86.5|
| Foundation eval |61|43|96|82.5|
| Human–SF policy bezSF |0|15|185|7.5|
| Human–SF eval bezSF |0|2|198|1|

Każda komórka poniżej zawiera W/D/L z40 partii.

| UCI_Elo przeciwnika | Foundation policy | Foundation eval | Human–SF policy | Human–SF eval |
|---:|---|---|---|---|
|1320|35/1/4|29/8/3|0/6/34|0/1/39|
|1700|22/11/7|21/7/12|0/1/39|0/0/40|
|2100|11/7/22|7/15/18|0/2/38|0/1/39|
|2500|0/12/28|4/9/27|0/4/36|0/0/40|
|2900|0/6/34|0/4/36|0/2/38|0/0/40|

Remisy z silniejszymi poziomami nie czynią dolnych ratingów wiarygodnymi pomiarami
spoza drabinki. Różne style i skłonność do powtórzeń mogą naruszać pojedynczą krzywą
Elo; per-anchor W/D/L jest ważniejszym surowym dowodem niż dokładne cyfry estymatora.

## Wejście i brak pomocy silnika

Foundation: BOS,Elo2600,Elo2600,effort0,faktyczne ruchy.

Human–SF: BOS,Elo2600,Elo2600,faktyczne ruchy, **bezSF tokenów i bez effort0**.
Zachowano metadata jego encoder’a, ale usunięto sloty z zewnętrzną podpowiedzią.
Nie zastępowano ich gwiazdką ani nową generacją myśli.

W procesie powstał tylko jeden silnik: przeciwnik. Wywoływano go wyłącznie w jego
turze. Funkcja wyboru ruchu modelu otrzymywała model, planszę, tokeny rzeczywistych
ruchów i statyczne reprezentanty binów, bez engine object, engine info, ocen czy
podpowiedzi. Rzeczywiście zagrane ruchy przeciwnika wchodzą do historii jak w normalnej
partii; nie są to dodatkowe rekomendacje na turę modelu.

Policy: legal argmax główki ruchów. Eval: wspólny prefix cache, jeden token kandydata
dla każdego legalnego dziecka, oczekiwana sigmoid-transformowana ocena reprezentantów
29 binówCP; białe maksymalizują/czarne minimalizują. Dokładne wyniki terminalne,
policy tie-break. Nie jest to kalibrowanyWDL ani głęboki search.

## Metodologia statystyczna i budżet

100 zamrożonych, różnych pozycji po8 półruchach z historycznej human validation,
seed20261015, bez filtrowania siłą/oceną. Nie jest to niezależny standardowy engine
opening book. Te same otwarcia we wszystkich czterech ramionach, obie barwy,
20 par na każdy poziom1320/1700/2100/2500/2900. Kolejność ramion rotowana po otwarciach.

Stockfish17.1, UCI_LimitStrength true,20k nodes/ruch, Threads1,Hash32MiB.
Neural FP32;200 świeżych gier na wariant. Stała próba bez zatrzymywania po istotności.
Brak rezygnacji i adjudykacji oceną. Claim_draw=True. Max480 półruchów/deadline byłby
censored, nigdy automatycznym remisem; w praktyce wszystkie800 gier miało naturalne
zakończenie według reguł.

Rating: logistyczny quasi-MLE wyników W=1,D=.5,L=0 względem etykiet przeciwników.
Przybliżony95% CI:10000 stratyfikowanych bootstrapów par otwarć, bez rozdzielania
kolorów; wspólne indeksy dla sparowanych różnic. Konserwatywny CI: Hoeffding na
ograniczonym wyniku pary i odwrócenie krzywej logistycznej. Przedziały zakładają
niezależność par i model ratingowy, nie pokrywają nieznanej kalibracji skali.

Przykładowa sparowana różnica human–SF policy minus foundation policy to−929,
przybliżony CI−1069 do−810. To różnica wyników systemów o różnej zgodności wejścia
z treningiem, nie przyczynowa miara katastroficznego zapominania. Tak duża ekstrapolacja
poniżej drabinki dodatkowo ogranicza interpretację wielkości luki w punktachElo.

Run trwał **1442s, około24min2s**, zakończył się18 października2026 o14:26:29 CEST,
około96min przed nieodnawialnym2h deadline’em. Suma czasów partii: foundation
policy399s, foundation eval591s, human–SF policy197s, human–SF eval221s.
To czasy całych gier, nie samegoGPU. JedenRTX3090, brak treningu, brak konkurencyjnych
GPUjobs, CPU annotation/UWR nietknięte. Container exit0, bezOOM.

## Niezależna weryfikacja i provenance

Pobrano sześć małych artefaktów i porównano wszystkieSHA256 z Adrianem;
manifest w `SHA256.json`. Odtworzono **wszystkie800 PGN**: legalność, naturalny
wynik, końcowyFEN, liczba półruchów, zgodność zJSON. Potwierdzono po200 rekordów
na wariant i800 unikalnych kombinacji mode/opening/colour. Osobno sprawdzono dokładne
prefiksy otwarć, przypisanie ratingów i kolorów oraz:

- external_advice_calls=0 i sf_tokens_in_input=0 we wszystkich800 rekordach;
- model_calls+opponent_calls=plies−8 we wszystkich rekordach;
- brak dodatkowych PGN po800; brak cenzurowania i powtarzania historycznych gier.

Root Adrian: `/home/ijakus/projects/krasnal-think/experiments/unaided-2x2-20261018`.
Container `kt-unaided-2x2-20261018`, ID
`10ce5383382231cee707713919dec68c7ed9723f3ec3496e366925e268bd80b7`.
FoundationSHA256: `b660b1f8eef7104123dd6551eff5c63ab3b665b72a8f1a8b2d17ec6fd71c7e81`.
InterleaveSHA256: `0605bf201b31397e50ed36705d4a5d3b8ae7f86607da1b45493f18b1c1544f33`.
RunnerSHA256: `aaf2bcd4ab2ef47a78262df615545f6805e96300e2bbfd1651668a9fb4a6ee4c`.
Pozostałe źródła i modele: `plan.json` oraz `../unaided-2x2-20261018-protocol.md`.
Checkpointy i poprzednie eksperymenty zachowano. Bez automatycznego zwiększania próby
lub nowego treningu po tym raporcie.
