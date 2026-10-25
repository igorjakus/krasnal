# Human–SF: odporność na jakość podpowiedzi —23 października2026

Cel: rozegrać partie human–SF z innymi ograniczeniami głębokości silnika podpowiadającego.
**Wyniki dotyczą systemu wspomaganego**, nie samodzielnej siły modelu.
Bez treningu. Pierwszy test dotyczy wyłącznie główki ruchów, legal argmax, nie eval head.

## Cztery budżety, po dwie kolumny

| Budżet SF podpowiadającego | Human–SF + podpowiedź | Sam podpowiadacz |
|---|---|---|
| Depth1 |100gier|100gier|
| Depth4 |100gier|100gier|
| Depth8 |100gier|100gier|
|30k nodes, jak w treningu|100gier|100gier|

SF17.1 ten sam binary we wszystkich ramionach. Depth1/4/8 to czyste limity depth,
bez dodatkowego capu nodes ani sztucznej redukcji UCI_Elo.30k to **osobny limit
węzłów, nie ustalony depth**. Dane treningowe używały30k nodes według manifestu
`reports/sf-interleave-100m-full-20261009/dataset.manifest.json`, binarySHA identyczny.
Rejestrujemy rzeczywisty osiągniętydepth/nodes/czas, nie zakładamy monotonicznej siły
na podstawie samej etykiety. Teacher unrestricted, Skill20, Threads1,Hash16MiB.

Podpowiadacz i przeciwnik są **osobnymi procesami**. Przeciwnik pozostaje identyczny:
UCI_Elo1320/1700/2100/2500/2900,20k nodes/ruch,Threads1,Hash32MiB.
Model widzi tylko token ruchu podpowiedzi, bez ocen, PV czy engine-info.

## Format, kontrole, próba

Checkpoint human–SF z7M-game fine-tune od foundation100M. Wejście zgodne z SWSB:
BOS,Elo2600,Elo2600,SF-token,faktyczny ruch,SF-token,faktyczny ruch,…, bieżącySF-token.
Bez effort0. Podpowiedź liczona przy KAŻDYM rzeczywistym prefiksie, także dla ośmiu
półruchów otwarcia i ruchów przeciwnika, zawsze przed faktycznym ruchem. Nie podajemy
przyszłej historii. To zachowuje strukturę wejścia, zmienia jakość warunkuSF.

Kontrola teacher-only wykonuje ten sam budżet podpowiadacza na każdym prefiksie,
również w turach przeciwnika; we własnej turze gra jego ruch bezpośrednio.
Nie dostaje mniejszej liczby wyszukiwań niż model. Nowy game-id/reset stanu silnika
na każdą partię, by nie przenosić tabel z mocniejszych budżetów do słabszych ramion.

800 świeżych gier,100/ramię.50 tych samych otwarć po8 półruchach dla wszystkich
8ramion, obiebarwy. Seed20261023 wybiera10 z20 historycznych otwarć na poziom,
z przypiętego planu foundation-games-20261015. Daje20gier/poziom/ramię.
Kolejność8ramion rotowana po otwarciach. Brak wyboru otwarć lub stopu na podstawie
wyników. To mały kontrolowany pilot, nie gwarancja wąskiegoCI.

Brak rezygnacji/adjudykacji oceną. Claim_draw=True; max480plies/deadline to
censored, NIE remis. Bootstrap10000resampli po50parach, stratyfikowany według
przeciwnika, wspólne indeksy przy różnicach, dodatkowy konserwatywny95%HoeffdingCI.
Rating warunkowy względem SF-UCI20k, nieFIDE/Lichess. Niepewność kalibracji i
logistic-model misspecification nie jest objętaCI. Przy floor/ceiling nie wymyślać
precyzyjnegoElo. Raportować W/D/L, model-minus-teacher, follow-rate, reacheddepth,
nodes, teacher/model czas. Depth shift może być OOD jakościowo mimo zgodnego formatu.

## Bezpieczeństwo i provenance

Skill `compute-hosts`: sprawdzono docker/GPU, brak aktywnych zadań przed dispatch;
nowy root/pinned code, jedenRTX3090, limit4CPU/12GiB. Nie ruszano UWR/innych prac.
**Nieodnawialny2h deadline** od utworzenia planu, także przy wznowieniu.
Bez automatycznego skalowania ani trenowania po pilocie.

Root Adrian: `experiments/human-sf-depths-20261023`, kod `root/code`.
Kontener `kt-human-sf-depths-20261023`, ID
`c3bfd2242abf45a66dfa3d6e45f312d2468e7e27786c3d01348ac3290a2f5f99`.
Plan SHA256 (Mac=Adrian): `20cbe796c50bb03b85396901365c2cc06e44bda0f5720b569b9b233598da3786`.
Nieodnawialny deadline: `1792749112.641077`.
Jednorazowy odbiór raportu: `wyniki-human-sf-r-ne-g-boko-ci`, kontrola co godzinę,
bez powiadomień o normalnym postępie; usunięcie automatyzacji po raporcie.
Planowane logi: dockerlogs,plan/status/result/completeJSON,gamesJSONL/PGN.
`advice.jsonl` na workerze przechowuje dla każdego prefiksuFEN,hint,depth,nodes;
audytować jego zgodność zPGN, brak przyszłych ruchów i legalność hintów. Nie ma
potrzeby ściągać całego większego trace na Maca; można pobrać mały wynik audytu iSHA.

ModelSHA: `0605bf201b31397e50ed36705d4a5d3b8ae7f86607da1b45493f18b1c1544f33`.
FoundationSHA: `b660b1f8eef7104123dd6551eff5c63ab3b665b72a8f1a8b2d17ec6fd71c7e81`.
RunnerSHA: `519740fe89495729fdd398965f35a9a6c348e75685f037ba49720f4a2fa5f597`.
HelperSHA: `798467624092237dd09c68a5fa925535e4c71ad7c2fbf29b752e30d637a5dd5b`.
ModelsourceSHA: `be282f8cd104aacd07e297867ad11f7a7c276b39bcf3af44565acec1a60ec89b`.
SelectorSHA: `1226b87caaa347f2c7f89d7f96e49cdbd40350fefdcf14519bc6da5ad3ce38db`.
TokenSHA: `9410613365aeacc027293509f4f881319c05b40f2525a965b8f9f21f31ad84c6`.
EngineSHA: `38faa5883b03652f847a87ed168b1bfee81b361db9584dae59a51cb91e69d9d6`.
Image: `sha256:ec6ccab1583618021d1fc2a582ce9e57c37bb3d5c33ae21d43facdc9a5dc4872`.

Smoke passed16krótkichpartii:4budżety×2systemy×2kolory, legalność, prefiksy,
actual depth limits. Smoke’y censored są oddzielone od wyników. Nie pisano unit tests.
Po starcie audytor `scripts/audit_human_sf_depth_games.py` niezależnie odtworzył
pierwsze16 PEŁNYCH partii i1809 prefiksów podpowiedzi: PGN, legalne hinty przy
dokładnym pre-move FEN, limity depth, teacher-control gra hint, wyniki/FEN/role/follow/
node/depth counters. Brak cenzurowanych partii w tej próbce. Auditor SHA256:
`677a1d657726421b843452093c7044a46d8307d690ee873296274c4cb4023976`.
Finalnyraport i małe artefakty naMac zmanifestSHA; zachować wcześniejsze eksperymenty.
