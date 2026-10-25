# Dokończenie human–SF — zatwierdzone45min,16 października2026

Użytkownik jawnie zaakceptował dodatkowe45min i dokończenie brakujących200 partii.
Nie zmieniamy600 ukończonych wyników ani oryginalnego deadline’u/protokołu.
Nowy niezależny root Adrian: `experiments/foundation-games-20261016-human-sf`.
Kontener: `kt-foundation-human-sf-20261016`.
ID: `b4f008b5af305dbddfbd9a42878eb4d65bcd4cadbb39ec4fab3e91f3e4999111`.
Nowy plan SHA256 (Mac=Adrian): `a363e4814a1f86b072cf7c0db60c0090098976b2a8657078e96a697b46632c97`.
Jednorazowy odbiór raportu: automatyzacja `jednorazowy-raport-elo-human-sf`,
kontrola co godzinę bez zwykłych powiadomień; po raporcie usuwa samą siebie.
Nieodnawialny deadline rozszerzenia: `1792138140`,16 października10:09:00 CEST;
obejmuje naprawę/przygotowanie/smoke/games. Bez automatycznego zwiększania próby.

Naprawa: `torch.serialization.safe_globals` dopuszcza jawny alias
`(PosixPath, "pathlib.PosixPath")` oprócz aktualnej klasy. Nadal `weights_only=True`.
Zmiana tylko na Macu i w nowym snapshotcie; stary pinned code nietknięty.

Runner dodaje wyłącznie jawne wznowienie z parent root: zachowuje jego100 otwarć,
sampling, ratingi, limity, checkpoint hashes i kopiuje600 atomowo ukończonych
wyników do nowego root jako istniejące dane. Nie rozgrywa ich ponownie.
Oryginalny plan jest powiązany przez SHA256; deadline rozszerzenia zapisuje się raz.

Źródła przed dispatch, identyczne lokalnie/worker:

- `scripts/foundation_games.py`: `798467624092237dd09c68a5fa925535e4c71ad7c2fbf29b752e30d637a5dd5b`.
- `src/krasnal_think/model.py`: `be282f8cd104aacd07e297867ad11f7a7c276b39bcf3af44565acec1a60ec89b`.
- Selector helper i tokens bez zmian wobec protokołu15 października.
- Foundation: `b660b1f8eef7104123dd6551eff5c63ab3b665b72a8f1a8b2d17ec6fd71c7e81`.
- Interleave: `0605bf201b31397e50ed36705d4a5d3b8ae7f86607da1b45493f18b1c1544f33`.

Smoke GPU przeszedł: pełne wczytanie i ścisły state_dict, cache/probability agreement,
dwie krótkie partie human–SF w obu kolorach z prawdziwym teacherem30k, legalność.
Te krótkie próby były oznaczone censored/smoke i nie wchodzą do wyników.
Nie dodano unit tests. Nie ruszano CPU annotation ani UWR. Przed startem GPU wolne.

Oczekiwane artefakty: plan/status/result/complete JSON oraz games JSONL/PGN.
Przy zakończeniu pobrać małe artefakty, porównać SHA256, niezależnie odtworzyć PGN,
dodać human–SF do raportu wraz z oboma95% CI, ograniczeniami i kontroląSF30k.
Nie nazywać wyniku wspomaganego siłą samodzielnego modelu ani Elo FIDE/Lichess.
