# Ewaluacja float32 — kontrola precyzji i effort:0

Oba przebiegi treatment zakończyły się kodem 0 na UWR, CPU `float32`, na tych
samych 9666 partiach testowych (652712 ruchów ludzkich). SF było teacher-forced.
Checkpoint treatment SHA-256:
`0605bf201b31397e50ed36705d4a5d3b8ae7f86607da1b45493f18b1c1544f33`.

| Wariant | `effort:0` | SF teacher-forced | Top-1 | NLL |
|---|---:|---:|---:|---:|
| Pretrain, poprawny audyt float32 | tak | nie | 54.1772% | 1.408680 |
| Pretrain, diagnostyczny audyt float32 | nie | nie | 51.5304% | 1.577603 |
| SF fine-tune, pierwotny format, float32 | nie | tak | 54.9561% | 1.380621 |
| SF fine-tune, dodany `effort:0`, float32 | tak | tak | 53.9503% | 1.432246 |
| SF fine-tune, pierwotny pomiar GPU bf16 | nie | tak | 54.9458% | 1.380821 |

**Wniosek o precyzji:** CPU float32 względem pierwotnego GPU bf16 zmienił treatment
o zaledwie **+0.0103 pp top-1** i **−0.000200 NLL**. Różnica precyzji nie wyjaśnia
wcześniejszego wyniku.

**Ważne zastrzeżenie:** kod treningu/evaluatora interleave pomijał `effort:0`,
którego używał pretrain. Wstawienie go tylko podczas ewaluacji treatment obniżyło
top-1 o ok. 1 pp; treatment nie był trenowany z tym tokenem, więc ten wariant
jest diagnostyką wrażliwości na format, nie rozstrzygającym porównaniem.

Wyniki dwóch porównań wejściowych są przeciwne: bez `effort:0` treatment ma
**+3.426 pp** względem baseline bez tokenu; z `effort:0` ma **−0.227 pp** względem
baseline z tokenem. Nie należy więc przedstawiać żadnej z tych różnic jako
czystego efektu SF. Potrzebny byłby trening kontrolny i treatment z identycznym
formatem wejścia (`effort:0`) oraz identycznym budżetem.

Pliki wynikowe SHA-256:

- `treatment-cpu-float32.json`: `eda7b0f67e6d7ce2d0709dc9243a99298a1df506e960b410602cc5cbf7ff501d`
- `treatment-effort0-cpu-float32.json`: `45ece5623dad8fb612a899bfff0e3d9f7d8173f69ac6db4808915a9206aff1a3`
- Dodatkowy evaluator z `effort:0`: `93933bf50ecf155f260f52132b24b1d41c7af759180acdf1338afc9c4138c27d`

Joby Slurma 19173 i 19174: `COMPLETED`, exit `0:0`, po 8 CPU na `opus0`;
czasy 26:24 i 26:14. Bez treningu i bez użycia GPU.
