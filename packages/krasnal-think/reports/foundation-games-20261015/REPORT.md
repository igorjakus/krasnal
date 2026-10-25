# Gry foundation — raport częściowo ukończonej serii

Stan sprawdzony16 października2026 około09:20 CEST. **600/800 partii**;
foundation policy, foundation eval selector i kontrola SF ukończone w całości.
Human–SF: **0/200**, awaria bezpiecznego wczytywania checkpointu.
Żadnego ponownego treningu ani nowych gier nie uruchomiono po wygaśnięciu budżetu.

## Wyniki

| System | Partie | W/D/L | Performance rating* | Przybliżony95% bootstrap CI | Konserwatywny95% CI |
|---|---:|---|---:|---|---|
| Foundation100M, główka ruchów |200|65/37/98|1935|1845–2031|1663–2207|
| Ten sam foundation, główka eval, wszystkie legalne dzieci |200|58/42/100|1890|1790–1991|1617–2162|
| Sam Stockfish30k, kontrola |200|197/3/0|3483 (ekstrapolacja poza drabinką)|3357–∞|2844–∞|
| Human–SF z podpowiedziami |0|—|nie zmierzono|—|—|

*Skala odpowiednika UCI_Elo przeciwnika przy tym konkretnym budżecie20k nodes,
NIE Elo FIDE ani Lichess. Stockfishowe etykiety nie stanowią niezależnie skalibrowanego
ratingu w tym protokole; przedziały opisują niepewność losowania gier, nie błąd skali.
Wyniku kontrolnego3483 nie traktować jako wiarygodnej precyzyjnej siły: najwyższy
przeciwnik miał etykietę2900, a górny kraniec przedziału jest nieskończony.

Różnica eval minus policy wynosi **−45 Elo**. Sparowany bootstrap po tych samych
otwarciach: przybliżony95% CI **−186 do+96**. Nie ma dowodu przewagi żadnej metody;
wyższe SF-top1 eval selector’a z wcześniejszego testu nie przełożyło się tu na
wykazaną poprawę gry. To mecze przeciw wspólnym przeciwnikom, nie bezpośredni mecz
policy–value. Nie mierzono ludzkiego stylu gry.

## Drabinka: W/D/L, po40 partii na komórkę

| UCI_Elo przeciwnika | Foundation policy | Foundation eval | SF30k kontrola |
|---:|---|---|---|
|1320|32/3/5|25/8/7|40/0/0|
|1700|21/10/9|18/5/17|40/0/0|
|2100|7/9/24|9/11/20|40/0/0|
|2500|5/7/28|6/14/20|40/0/0|
|2900|0/8/32|0/4/36|37/3/0|

## Metodologia i walidacja

100 zamrożonych, różnych pozycji otwarcia po8 półruchach, seed20261015,
z historycznych partii walidacyjnych. Dwie zamienione barwy na otwarcie;
te same otwarcia dla wszystkich systemów.20 par na każdy z pięciu poziomów.
To nie niezależny standardowy turniejowy opening book. Przyjęty rozkład otwarć,
style przeciwników i dopasowanie logistycznej krzywej Elo ograniczają uogólnienie.

Stockfish17.1, przeciwnik20k nodes/move, teacher/control30k, Threads1/Hash32MiB.
Modele FP32, conditioning2600/2600; foundation z obowiązkowym effort0.
Eval selector: all-legal one-ply, expected sigmoid-transformedCP, oceny względem
białych, czarne minimalizują, jawne wyniki terminalne. Nie jest to kalibrowany WDL.

Próbka każdego z trzech ukończonych systemów jest stała,200 gier/100 par;
bez zatrzymywania po osiągnięciu istotności. Przybliżone przedziały:10000
stratyfikowanych bootstrapów par otwarć. Osobno konserwatywna granica Hoeffdinga
dla średniego wyniku pary, odwrócona przez przyjęty model logistyczny. Założenia:
niezależne pary i poprawność modelu odpowiedzi ratingowej; nie uwzględniają kalibracji
etykiet Stockfisha. Raportuj oba przedziały, nie wybieraj tylko węższego.

**Niezależnie zweryfikowano wszystkie600 PGN:** legalność ruchów, końcowe FEN,
rzeczywiste zakończenia zgodne z regułami, liczby półruchów i zgodność wyników z JSON.
Brak cenzurowanych gier; limitu półruchów ani deadline’u nie zakodowano jako remisu.
SHA256 każdej pobranej małej pozycji porównano z workerem; patrz manifest.

## Awaria human–SF i pozostały zakres

Dokładny checkpoint UWR został w całości skopiowany, a jego hash zweryfikowany:
`0605bf201b31397e50ed36705d4a5d3b8ae7f86607da1b45493f18b1c1544f33`.
Awaria nastąpiła przy `GPT.load` przed pierwszą partią tego modelu.

Potwierdzona przyczyna zgodności Pythona: checkpoint zawiera pickle-global
`pathlib.PosixPath`; runtime Python3.13 przedstawia tę samą klasę jako
`pathlib._local.PosixPath`. Dotychczasowy safe allowlist dodaje klasę pod jej
aktualną nazwą i nie rozpoznaje historycznego aliasu. Odczyt listy niezaufanych
globali wskazuje wyłącznie `pathlib.PosixPath`. Nie wyłączono `weights_only`.
Możliwa minimalna naprawa: jawne dopuszczenie historycznego aliasu dla standardowej
klasy, z zachowaniem bezpiecznego loadera, smoke i nowego przypiętego snapshotu.

Kontener `kt-foundation-games-20261015` zakończył się exit1 bezOOM.
Deadline zapisany w planie:16 października04:16:13 CEST; już minął.
**Nie resetowano deadline’u ani nie wznowiono gier.** Dokończenie200 partii human–SF
wymaga nowej zgody na ograniczone przedłużenie. Zachować600 wyników, istniejące
checkpointy, kod i protokół; nie powtarzać ukończonych ramion.

Human–SF w tym teście oznacza system z prawdziwą podpowiedzią SF przed KAŻDYM
półruchem, jak podczas treningu. Jego przyszły rating nie będzie siłą samodzielnego
modelu. Główka eval nie była uczona podczas fine-tuningu i nie jest używana w tym ramieniu.

## Czasy i artefakty

Czas sumy rozegranych gier: policy422s, eval590s, kontrola706s — około28m38s.
Transfer1.2GB oraz start/model loading/raportowanie są poza tymi sumami;
nie są to czasy treningu ani pomiar samegoGPU. Budżet globalny obejmował całość.

Root Adrian: `/home/ijakus/projects/krasnal-think/experiments/foundation-games-20261015`.
Container ID: `615b93a981a3676405c717cc1f249ab9842487905ffd9f8c2a2ca765884b6634`.
Foundation SHA256: `b660b1f8eef7104123dd6551eff5c63ab3b665b72a8f1a8b2d17ec6fd71c7e81`.
Pinned runner SHA256: `48c2e0ea226fb71fd49d7fb1fa9cf22392468d748ccbb8b308f202001c97dc39`.
Nie utworzono monitora: proces już zakończony awarią, ten raport jest jednorazowym
powiadomieniem o wyniku i brakującym zakresie. Nie ruszano generatoraCPU ani UWR.
