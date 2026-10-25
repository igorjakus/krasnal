# Uruchomienie testu gier

Adrian, 15 października 2026, kontener `kt-foundation-games-20261015`,
ID `615b93a981a3676405c717cc1f249ab9842487905ffd9f8c2a2ca765884b6634`.
Root: `/home/ijakus/projects/krasnal-think/experiments/foundation-games-20261015`.

Zaplanowano 800 partii: foundation policy, foundation eval selector, SF30k control,
human–SF policy z prawdziwymi podpowiedziami. Po200 na wariant. To nie jest trening.
Checkpoint interleave jest kopiowany z UWR przez Mac; wariant uruchomi się dopiero po
weryfikacji dokładnej liczby bajtów i SHA256. Model human–SF nie jest samodzielnym silnikiem
w tym protokole; testuje się system z podpowiedziami, zgodny z formatem treningu.

Weryfikacja uruchomienia: kontener działa, jeden proces GPU, generator CPU nienaruszony.
Pierwsze40 kompletnych PGN przeliczono niezależnie: legalne ruchy, końcowe pozycje,
faktyczne zakończenia i wyniki zgodne z JSON. Przy kontroli zapisano62 partie; seria trwa.
Nie ogłaszamy wyniku Elo na podstawie tej wczesnej próbki.

Plan ma nieodnawialny deadline `1792116973.2721786` (16 października04:16:13 CEST).
Szacowany czas całej serii1–2h jest orientacyjny; transfer i długie partie mogą go wydłużyć.
Maksymalny budżet6h. Nie utworzono automatyzacji ani dodatkowych zadań GPU.

Małe kopie lokalne:

- `foundation-games-20261015-plan.json`: SHA256 `32f7ece4df2e88303e0f190f8c01f1478040329f22f909f78d7006418e6fd397`.
- `foundation-games-20261015-interim.json` (40gier, NIE wynik końcowy): SHA256 `08e21e9d05fbc7b8dcbeb083ebc3f2a23fe554a46ec87c74720ed6f06538ff19`.

Finalne `result.json`, `games.jsonl`, `games.pgn`, `complete.json` pozostają na workerze.
Do końcowego raportu pobrać małe artefakty i zweryfikować SHA256, zachowując ten snapshot
interim jako historyczny. Przedziały bootstrap są przybliżone; raport zawiera również
konserwatywny95% przedział po parach otwarć. Przy cenzurowanych grach priorytetem są
granice uwzględniające najgorszy/najlepszy wynik, nie estymator complete-case.
Skala: odpowiednik ustawienia UCI_Elo przy20k nodes, nie Elo FIDE/Lichess.
