# Czysty test2×2 —18 października2026

Cel użytkownika: cztery wartości w tabeli2×2, oba checkpointy grane normalnie
główką ruchów i wyborem ruchu przez główkę eval, **bez zewnętrznych podpowiedziSF**.
Nie utożsamiać starego wspomaganego wyniku2231 z tym testem.

| Model | Legal policy argmax | All-legal child eval selector |
|---|---|---|
| Foundation100M |200 nowych gier|200 nowych gier|
| Human–SF fine-tune, bez SF input |200 nowych gier|200 nowych gier|

Jedyny silnik w procesie jest **przeciwnikiem**. Nie tworzymy teachera. Silnik
jest wywoływany tylko w gałęzi ruchu przeciwnika. Neural chooser otrzymuje model,
board, tokeny faktycznej historii i statyczne reprezentanty binów — żadnego silnika,
ocen, podpowiedzi ani nowych analiz. Faktycznie zagrany ruch przeciwnika oczywiście
staje się elementem historii, jak w normalnej grze.

## Wejścia i ograniczenia

- Foundation: BOS,Elo2600,Elo2600,effort0,faktyczne ruchy.
- Human–SF: BOS,Elo2600,Elo2600,faktyczne ruchy; **bez tokenówSF i bez effort0**.
  Metadata odpowiada jego encoderowi human-only; pominięto wszystkie slotySF.
  To jawna ablacja wejścia poza rozkładem jego treningu SWSB, nie dowód wyizolowanego
  wpływu fine-tuningu. Nie zastępujemy SF prawdziwą podpowiedzią, pustą gwiazdką ani
  dodatkową generacją bez zgody.
- Fine-tune human–SF nie trenował eval head. Zachowana29-klasowa główka i zmieniony
  wspólny trunk mogą być gorzej skalibrowane; mimo tego mierzymy ją zgodnie z prośbą.
- Oba selektory eval: wszystkie legalne dzieci, expected sigmoid-transformedCP,
  stałe reprezentanty binów, białe maksymalizują/czarne minimalizują, dokładne wyniki
  terminalne i policy tie-break. Nie jest to skalibrowaneWDL ani głębokie search.

## Zamrożony eksperyment

800 świeżych partii, bez przepisywania historycznych wyników do nowej próby.
Te same100 otwarć z planu foundation-games-20261015;20par na poziom1320/1700/2100/
2500/2900, obie barwy. Kolejność czterech systemów rotowana po otwarciach.
Opponent SF17.1,20k nodes/move, Threads1,Hash32MiB, UCI_LimitStrength true.
FP32, legal move mask, conditioning2600/2600. No training. JedenRTX3090.
Nieodnawialny2h deadline zapisuje się raz przy utworzeniu planu; resumption go
nie przedłuża. Zwykłe zakończenia i claim_draw=True;480ply/deadline jest censored,
nie remisem. Bez rezygnacji i adjudykacji na podstawie ocen.

Rating to ekwiwalent UCI_Elo przy tych warunkach, nieFIDE/Lichess. Per-anchorWDL,
przybliżony95% bootstrap10000 resampli po parach otwarć i konserwatywny95%Hoeffding
z odwróceniem krzywej logistycznej. Niepewność kalibracji skali nie jest objętaCI;
niepełna próba i cenzurowanie wymagają jawnych ostrzeżeń/granic. Brak optional stopping
po istotności. Po zakończeniu niezależna weryfikacjaPGN i SHA256 kopii.

## Provenance

Adrian root: `experiments/unaided-2x2-20261018`; osobny pinned snapshot `code/`.
Container `kt-unaided-2x2-20261018`, ID
`10ce5383382231cee707713919dec68c7ed9723f3ec3496e366925e268bd80b7`.
Plan SHA256 (Mac=Adrian): `6a39ef733b7c3002521a5c7210741d16b2d257d1b969a8529d449d6be2af4bc7`.
Nieodnawialny deadline planu: `1792332147.1137862`.
Jednorazowy odbiór raportu: `raport-2-2-bez-podpowiedzi-sf`; kontrola co godzinę,
cisza przy zwykłym postępie, po końcowym raporcie usuwa automatyzację.
Nie nadpisujemy starszych kodów, checkpointów, danych ani protokołów. CPU annotation
pozostaje aktywne, UWR nietknięte. GPU przed dispatch było wolne.

Foundation SHA256: `b660b1f8eef7104123dd6551eff5c63ab3b665b72a8f1a8b2d17ec6fd71c7e81`.
Interleave SHA256: `0605bf201b31397e50ed36705d4a5d3b8ae7f86607da1b45493f18b1c1544f33`.
Runner `scripts/unaided_foundation_games.py`:
`aaf2bcd4ab2ef47a78262df615545f6805e96300e2bbfd1651668a9fb4a6ee4c`.
Shared report/chooser `scripts/foundation_games.py`:
`798467624092237dd09c68a5fa925535e4c71ad7c2fbf29b752e30d637a5dd5b`.
Model source:
`be282f8cd104aacd07e297867ad11f7a7c276b39bcf3af44565acec1a60ec89b`.
Selector helper:
`1226b87caaa347f2c7f89d7f96e49cdbd40350fefdcf14519bc6da5ad3ce38db`.
Tokens:
`9410613365aeacc027293509f4f881319c05b40f2525a965b8f9f21f31ad84c6`.
Image: `sha256:ec6ccab1583618021d1fc2a582ce9e57c37bb3d5c33ae21d43facdc9a5dc4872`.
Engine: `38faa5883b03652f847a87ed168b1bfee81b361db9584dae59a51cb91e69d9d6`.

Smoke obejmuje oba checkpointy, obie główki i oba kolory, legalność, właściwy kierunek
White/Black oraz cached/uncached agreement. Każdy wiersz wyników zapisuje model_calls,
opponent_calls, external_advice_calls=0 i sf_tokens_in_input=0. Runtime asercje sprawdzają
brakSF tokenów i dokładną długość historii. Nie dodano unit tests.
Smoke passed dla wszystkich8 kombinacji checkpoint/główka/kolor. Maksymalne
cached/uncached probability delta: foundation1.79e-7,interleave1.20e-7.
Po starcie niezależnie odtworzono pierwsze8 PEŁNYCH partii (wszystkie kombinacje):
PGN/legalność/outcome/FEN zgodne z JSON; model_calls+opponent_calls=plies−8;
external_advice_calls=0 i sf_tokens_in_input=0. Przy ostatniej kontroli zakończono
co najmniej200/800 gier; nie są to jeszcze końcowe wyniki.
