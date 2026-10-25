# Human–SF z różną jakością podpowiedzi — raport końcowy

**800/800 świeżych partii ukończonych.** Po100 na każdy z8 wariantów. CPU e2e audyt zweryfikował wszystkie partie oraz **73310 prefiksów podpowiedzi**; brak cenzurowania. Runtime44min45s, w nieodnawialnym limicie2h. To świadomie **gra wspomagana**, nie siła samodzielnego modelu.

## Główne wyniki

W komórkach: rating w umownej skali testu oraz przybliżony95% bootstrap CI.

| Podpowiadacz | Human–SF + podpowiedź | Sam podpowiadacz |
|---|---:|---:|
| Depth1 | **2021** [1900, 2140] | **2021** [1900, 2130] |
| Depth4 | **2100** [1981, 2210] | **2251** [2150, 2352] |
| Depth8 | **2130** [2001, 2251] | **2852** [2759, 2940] |
| 30k nodes (jak w treningu) | **2301** [2170, 2433] | **3431*** [3261, ∞] |

**Nie jest to Elo FIDE ani Lichess.** UCI_Elo przy20k nodes to tylko etykieta przeciwnika w tej procedurze. CI nie uwzględniają kalibracji skali, zmiany puli przeciwników ani błędu modelu logistycznego.

\* Kontrola30k osiągnęła98W/2D/0L. Punktowe3431 jest ekstrapolacją poza najwyższego przeciwnika2900; górnyCI∞. Nie traktować tego jako precyzyjnej siły3431.

## Co wynika z pilota

- Depth1: oba systemy uzyskały dokładnie37W/18D/45L i ten sam punktowy rating. Brak dowodu, że model poprawia słabą podpowiedź.
- Depth4: model2100 kontra teacher2251, nominalna różnica−150; efekt graniczny w niewielkim pilocie, bez korekty wielokrotnych porównań.
- Depth8: model2130 kontra teacher2852. Duża i wyraźna strata w wynikach całego systemu.
-30k: model2301, podczas gdy samo granie podpowiedzi niemal zdominowało drabinkę. Model nie wykorzystuje siły silnika w pełni; w tej konfiguracji pogarsza wynik względem użycia samego silnika.
- Mocniejsza podpowiedź poprawia wynik modelu:30k minus depth1 około+281, sparowany przybliżony95%CI[+110,+442]. Nie dowodzi to monotoniczności każdego kolejnego kroku ani przyczynowego mechanizmu wyborów.
- Model grał ruch podpowiadacza przy około49–52% własnych decyzji, podobnie dla wszystkich budżetów. To miara zgodności, nie trafności: brak kopiowania nie znaczy automatycznie korekty błędu.

Interpretacja: wyniki są zgodne z celem uczenia przewidywania ludzkich ruchów, a nie maksymalizacji siły gry. Nie dowodzą samodzielnie tej przyczyny. To jeden checkpoint, jeden seed/book, mała próba i przesunięcie jakości podpowiedzi względem treningu. Bez nowych treningów lub automatycznego skalowania po raporcie.

## Sparowane różnice model minus teacher

Te same50 otwarć, zamienione kolory i wspólne indeksy bootstrapu. To różnice performance ratingu i wyniku przeciw wspólnym przeciwnikom, nie bezpośredni mecz model–teacher.

| Budżet | Δrating | Przybliżony95% CI | Δśredniego wyniku | Przybliżony95% CI wyniku |
|---|---:|---|---:|---|
|Depth1|0|[−160,+160]|0pp|[−8,+8]pp|
|Depth4|−150|[−301,−10]|−7.5pp|[−15,−0.5]pp|
|Depth8|−721|[−869,−579]|−34.5pp|[−41.5,−28]pp|
|30k|≈−1130*|[−∞,−914]|−39pp|[−45.5,−32.5]pp|

\* Ratingowa luka przy30k obejmuje ekstrapolację kontroli poza drabinkę. Różnica surowych punktów jest bardziej bezpośrednią obserwacją. CIs nominalne, eksploracyjne, bez korekty wielokrotnych porównań.

Dodatkowo human–SF: depth4 minus30k=−201,CI[−353,−40]; depth8 minus30k=−171,CI[−333,−10]. Nie należy z tego wybierać konfiguracji i ogłaszać jej zwalidowanym optimum bez niezależnego testu.

## Konserwatywne95% CI i W/D/L

| System | W/D/L | Rating | Konserwatywny95% Hoeffding CI |
|---|---|---:|---|
| human_sf_d1 | 37/18/45 | 2021 | [1635, 2407] |
| teacher_d1 | 37/18/45 | 2021 | [1635, 2407] |
| human_sf_d4 | 41/18/41 | 2100 | [1715, 2488] |
| teacher_d4 | 50/15/35 | 2251 | [1866, 2644] |
| human_sf_d8 | 42/19/39 | 2130 | [1745, 2519] |
| teacher_d8 | 78/16/6 | 2852 | [2439, ∞] |
| human_sf_n30k | 52/16/32 | 2301 | [1916, 2698] |
| teacher_n30k | 98/2/0 | 3431 | [2710, ∞] |

Hoefdding CI odwraca ograniczenie średniego wyniku pary przez model logistyczny. Zakłada niezależność par i ten model ratingowy; nie naprawia nieznanej kalibracji UCI_Elo. Bootstrap jest przybliżony. Przy100 grach/50 parach nie należy udawać wąskiej, absolutnej niepewności siły.

## Drabinka: W/D/L,20 gier na komórkę

| Przeciwnik | humanSF d1 | teacher d1 | humanSF d4 | teacher d4 | humanSF d8 | teacher d8 | humanSF30k | teacher30k |
|---:|---|---|---|---|---|---|---|---|
|1320|18/1/1|17/3/0|18/1/1|17/2/1|18/0/2|20/0/0|19/0/1|20/0/0|
|1700|14/2/4|12/2/6|14/4/2|20/0/0|13/2/5|19/1/0|16/2/2|20/0/0|
|2100|3/7/10|6/6/8|4/5/11|10/6/4|8/4/8|19/1/0|10/7/3|20/0/0|
|2500|2/6/12|2/7/11|5/4/11|3/4/13|3/8/9|17/2/1|7/4/9|19/1/0|
|2900|0/2/18|0/0/20|0/4/16|0/3/17|0/5/15|3/12/5|0/3/17|19/1/0|

## Rzeczywisty budżet i koszt

| System | Średni osiągniętydepth | Średnie nodes/prefiks | Czas advisera /100gier | Czas modelu /100gier | Follow-rate |
|---|---:|---:|---:|---:|---:|
|human_sf_d1|1.00|41|15.4s|45.9s|51.01%|
|teacher_d1|1.00|38|20.6s|0.0s|—|
|human_sf_d4|4.00|285|24.7s|51.2s|49.01%|
|teacher_d4|4.00|253|30.2s|0.0s|—|
|human_sf_d8|8.00|1252|54.9s|53.0s|49.59%|
|teacher_d8|8.00|1176|64.2s|0.0s|—|
|human_sf_n30k|23.88|29324|561.1s|39.0s|52.43%|
|teacher_n30k|26.30|29109|459.3s|0.0s|—|

30k nodes **nie oznacza ustalonej głębokości**. Średnia23.88/26.30 obejmuje całe trajektorie, także końcówki i pozycje wymuszone, a obie grupy systemów odwiedzają inne pozycje. Nie porównywać tych średnich jak wspólnej próbki pozycji. Mate/forced-move search może kończyć się przed node budget, dlatego średnie są nieco poniżej30k. Czas advisera obejmuje wszystkie prefiksy, również otwarcia i ruchy przeciwnika. Czas modelu to zmierzona funkcja wyboru, nie czyste utilizationGPU; pozostały wall-clock zawiera przeciwnika, I/O i raportowanie.

## Zamrożona metodologia i ograniczenia

Model: human–SF fine-tune od foundation100M, główka ruchów, legal argmax, FP32, Elo2600/2600. Format zgodny z treningiem: BOS,EloW,EloB,SF-token,faktyczny ruch,…,bieżącySF-token; bez effort0. Podpowiadacz wywoływany przed KAŻDYM rzeczywistym półruchem, także8ply otwarcia i ruchami przeciwnika. Model otrzymuje tylko tożsamość ruchu, bez score/PV ani przyszłej historii.

Dwa osobne procesy: adviser unrestrictedSkill20/Threads1/Hash16 oraz opponent UCI_LimitStrength true/Threads1/Hash32. Ten sam SF17.1 binary. Adviser ma czysty limit depth1/4/8 albo osobny30k node limit z treningu; brak mieszania depth z narzuconym node capem. Teacher-only oblicza adviser na wszystkich tych samych rodzajach prefiksów i gra jego ruch we własnej turze. Nowy game-id/reset tabel silnika na każdą partię. Użytkownik świadomie zlecił grę wspomaganą.

50 wspólnych8-ply otwarć wybranych seed20261023 z przypiętych100 historycznych human validation;10 par/anchor. Nie jest to niezależny standardowy engine opening book. Rotacja8ramion po otwarciach. Stała próba100gier/mode, bez optional stopping po istotności. Bootstrap10000 stratified resamples po plan.opening_ids z zachowaniem kolorów, seed20261023 dla różnic. Reportowane indywidualneCI używają seed20261015 w dotychczasowym helperze; różne seedy nie zmieniają definicji próby.

Brak rezygnacji/adjudykacji oceną. Claim_draw=True.480ply lub deadline byłyby censored, nie remisem; w praktyce0censored.1RTX3090,4CPU/12GiB,2h immutable deadline. Ukończono po2684.76s (44min45s), bezOOM, container exit0. Nie zmieniano innych zadań ani UWR.

## Audyt i artefakty

AudytCPU bezGPU odtworzył wszystkie800 PGN i73310 rekordów advice: legalne ruchy i hinty, dokładny pre-moveFEN/półruch, limitydepth, odtworzenie wyniku/końcowegoFEN, liczniki ról/follow/nodes/depth, poprawne otwarcia i kolory. Dla teacher-only potwierdzono rzeczywiste granie hintu. Każdy wariant ma100 partii; brak dodatkowychPGN/trace po końcu. Nie używano unit tests.

Pobrane małe artefakty zweryfikowano SHA256 względem workera. advice.jsonl pozostaje na Adrianie z hashem w audycie i manifeście. Proces po ukończeniu nie był restartowany. Pełne wyniki i czasy: result.json; audyt: audit.json; hashe: SHA256.json.

Adrian root: /home/ijakus/projects/krasnal-think/experiments/human-sf-depths-20261023.
Container kt-human-sf-depths-20261023, ID c3bfd2242abf45a66dfa3d6e45f312d2468e7e27786c3d01348ac3290a2f5f99.
ModelSHA256:0605bf201b31397e50ed36705d4a5d3b8ae7f86607da1b45493f18b1c1544f33.
RunnerSHA256:519740fe89495729fdd398965f35a9a6c348e75685f037ba49720f4a2fa5f597.
AuditorSHA256:677a1d657726421b843452093c7044a46d8307d690ee873296274c4cb4023976.
Pozostałe źródła i foundation hash w plan.json oraz ../human-sf-depths-20261023-protocol.md.

