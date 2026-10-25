# Format thinking

Nowy pipeline używa profilu **compact-v1** opisanego w
[docs/curriculum.md](docs/curriculum.md): direct + effort 2, maksymalnie 3 PV ×
3 plies, cztery klasy delty i wykładniczo malejące wagi ruchów PV. Profil jest
zapisywany w checkpointach; poniżej pozostaje opis formatu legacy dla starych runów.

Historia partii jest zapisana jako ruchy UCI. Bez thinking model od razu zwraca
kolejny ruch:

```text
<bos> <elo:8> <elo:8> e2e4 e7e5 g1f3 b8c6 f1b5
```

Jeśli model dostaje budżet, token `<effort:n>` występuje bezpośrednio przed
thinking dla danego ruchu. Budżet jest maksymalną liczbą tokenów wewnątrz
`<think>`, a nie ustaloną liczbą PV ani ich długością.

```text
<bos> <elo:8> <elo:8> e2e4 e7e5 g1f3 b8c6
<effort:2>
<think>
  <pv> f1c4 g8f6 d2d3 <delta:3> </pv>
  <pv> f1b5 a7a6 b5a4 <delta:0> </pv>
</think>
f1b5
```

Krótszy poprawny thinking z tym samym budżetem:

```text
... d2d4 d7d5
<effort:2>
<think>
  <pv> c2c4 e7e6 <delta:0> </pv>
</think>
c2c4
```

Po ruchu blok thinking nie zostaje w historii następnej decyzji:

```text
... d2d4 d7d5 c2c4 e7e6
<effort:1>
<think> <pv> b1c3 g8f6 <delta:0> </pv> </think>
b1c3
```

Zasady:

- bez thinking nie emitujemy `<effort:0>` ani pustego `<think>`;
- w SFT około połowa przykładów uczy bezpośredniego ruchu, a pozostała połowa
  jest równo dzielona między budżety 1–4;
- `<effort:1>` ... `<effort:4>` oznaczają wyłącznie maksymalny budżet;
- każdy `<pv>` zawiera co najmniej jeden ruch, token oceny i `</pv>`;
- kolejność kandydatów jest mieszana, żeby pierwsze PV nie zdradzało odpowiedzi;
- każde PV kończy token straty względem najlepszego kandydata (`<delta:...>`),
  a najlepszy kandydat ma deltę 0;
- po `</think>` występuje dokładnie jeden ruch końcowy;
- ruchy wewnątrz PV i ruch końcowy muszą być legalne;
- thinking jest usuwany z kontekstu po wykonaniu ruchu.

## DAgger

On-policy DAgger obejmuje wyłącznie przykłady z effort 1--4 i zawsze wymusza
różne pierwsze ruchy PV, o ile pozycja ma wystarczająco wiele legalnych ruchów.
Stockfish ponownie ocenia kandydatów, wylicza delty i wskazuje ruch końcowy.
Globalny najlepszy ruch Stockfisha jest zachowany jako target; dane DAgger nigdy
nie zastępują nim targetu direct, ponieważ direct pochodzi wyłącznie ze strumienia
teacher. Kolejność PV jest losowana podczas kodowania każdej epoki.
DAgger examples always use effort 1--4. Their PV roots must be roots actually
generated on-policy by the model; the global Stockfish best move must not be
injected when absent. Stockfish re-evaluates those roots, deltas are relative to
the best generated root, PV order is randomized during encoding, and the final
move target is the best generated root. Effort 0 comes only from teacher data,
where its target remains the global Stockfish best move.
