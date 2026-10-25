# Mix i pilot RL — raport końcowy, 11 października 2026

Etap B ukończony, kontener zakończył się kodem0 o21:06:56 CEST. 1000updates/batch64, nominalnie64000przykładów; aktualnego rozkładu trained exposures per component nie zapisano (sampling przed bucketingiem nie jest dokładną liczbą trained exposures). Przygotowanie+trening+fixed eval około7min29s; trening około75s. Budżet całej kontynuacji kończy się11października02:59:30 CEST i nie jest resetowany dla RL.

| Wariant | Human% | SFdirect% | SFthinking% | Recall% | Selection% | Conditional% | Format errors | Direct CP-gap | Thinking CP-gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DAgger3 | 50.58 | 51.1 | 46.1 | 71.6 | 56.6 | 64.11 | 0/100 | 174.72 | 159.83 |
| Teacher SFT trace | 50.60 | 50.5 | 41.7 | 71.8 | 51.1 | 57.94 | 0/100 | 166.68 | 194.13 |
| Teacher SFT final-only | 50.58 | 50.3 | 40.7 | 68.7 | 56.3 | 59.24 | 1/100 | 166.69 | 184.88 |
| Mix50/40/10 | 50.71 | 50.1 | 47.0 | 72.8 | 56.9 | 64.15 | 0/100 | 186.78 | 144.23 |
| Mix + pilot RL | 50.69 | 50.5 | 47.3 | 73.1 | 56.0 | 63.75 | 0/100 | 185.97 | 152.94 |

Mix daje mały wzrost thinking top1/recall/human względem startu, ale spadek direct top1. Thinking CP-gap jest niższy, direct wyższy. To jedno ziarno i1000fixedpozycji, bez paired uncertainty analysis, nie dowód przewagi ani siły gry. Start RL wybrano operacyjnie z mixu, nie jako statystycznie udowodniony zwycięski checkpoint. Selection i conditional selection mają różne definicje (patrz raportA). Nie wykonano gier ani wiarygodnej metryki delt.

Metodologia B identyczna z planem:50%onpolicy final-choice (4096deduplicated perround1–4),40%teacher trace8192nextshard,10%human train.bin direct. Każde źródło filtruje fixedgold histories; teacher trace pozostaje normalnie nadzorowane, own/human finalonly. Optimizer kontynuowany od wspólnego DAgger3, seed42, LR5e-6→5e-7,warmup50,1000updates. Fixedeval1000gold/SF30k/human30batches seed1729/format100. Foundation100M, nie retired34M. Checkpoint mixSHA256 `3ce3fd16bcb5b0659e56dd597d7a95be0dc03f14774f32e1a09ba3757fab50e6`.

## RL protokół i weryfikacja

Osobny snapshot rl-code, bez modyfikowania kodu aktywnegoB. REINFORCE, cztery próbki na pozycję, do256unikalnychtrainFEN znextshard, goldhistorywykluczone. Compacteffort2,max3PV,max3plies,temperature1/top_p1. Model w evalmode dla zgodności sampling/scoring (gradienty nadal aktywne). Cały sampled trace+finalmove ma legality-conditioned likelihood; forced/singleton grammar nie jest stochastyczną akcją i nie wchodzi do normalizacji. Średni logprob po stochastycznych decyzjach, group-centered reward/std floor0.05, gradientclip1, AdamWlr1e-6/wd0. KL0.02 to exact categorical KL na tych samych legalnych allowedsets do zamrożonego startowego mixu. Jest to length-normalized surrogate PG; różne długości trace zmieniają wagę względem klasycznej sumy logprob.

Reward: minus clipped regret WDL expectedoutcome między SFbest iSFroot-constrained finalmove, oba przy oryginalnej pozycji/side-to-move, SF30knodes,Threads1,Hash64. Nie selfplay ani reward za zgodność trace zteacherem. Czas do29min30s na wejście nowych grup, pozostawiono30s zlimitu30min; save/eval wglobal6h. Ciężka grupa może przekroczyć margines — czas rzeczywisty trzeba odczytać zcomplete.json, nie założyć.

GPU smoke sprawdził alignment sampled token/feed, legalność finalmove, mate white/black perspective, WDLzero0.5, normalization finite, KL≈0 dlaidentycznych modeli, niezerowy trace gradient≈96711.5. Pierwsza pozycja miała cztery równe reward0, więc nie dawała PGsignal; to nie dowód skuteczności. RL loguje rewardy i gradienty każdej aktualizacji, co pozwoli policzyć grupy zniezerową wariancją. Smoke nie jest wynikowym pilotem. Żadnychunit tests nie dodano.

## SHA256 małych artefaktów

Porównano Mac/Adrian, zgodne. Lokalne JSON/logi: reports/thinking-diagnostic-bc-20261010-artifacts; dużecheckpointy pozostają naAdrianie.

| Plik | SHA256 |
|---|---|
| plan.json | 889c45a2f45134f3f3f1f3c93e4d69cb82fbdc4ead258dc5e9a19ce5c4cdd76f |
| mix-plan.json | 6dedbe76d96e201830b45b0fd055fecca5154220132df845630375e37405c343 |
| stage-b-complete.json | 78904cc87136cad831aaccd3e9232d51717ea74acc95745839e48724b5a37e0e |
| mixed/eval/result.json | e00d36f02efedc6e653161dd6abd1f62f00c8eb16a9f210908a55fedc440dc40 |

SnapshotC exactscriptSHA256 przed dispatch: diagnostic_rl.py `8dac7d7e6173b6416750a72cf87d51e120c6ee46e25b0202cba9b873f7aa8e75`; run_diagnostic_rl.py `6c0d65b0dffab8a7205328fb7eceee9a0cb78b8a33cf26f0a265c77bd04ee7a9`; run_thinking_ablation.py `8cbfafce23915b059e990e0c9b3d9e4786f97ac9a256b1b27ba257be0d37c28c`. Pełnýmanifest wrl-source-manifest.json przy uruchomieniu. Wyniki RL zostaną dopisane po fixedeval, nie deklarujemy ich z góry.

C wystartował23:05:54CEST, kontener `kt-thinking-diagnostic-rl-20261010`, ID `b25c00ee895f27b1a066cb6dd2e6935c378a5e77348086517182d903ba435692`. Pierwsze12update wykonane; update11/12 miały nierówne nagrody i niezerowy gradient. To potwierdza działanie techniczne, nie poprawę. Driver automatycznie wykonuje fixedeval po zapisaniu checkpointu; monitor zaktualizowany, by nie dispatchować duplikatu.

## Zakończenie i interpretacja

Wykonano całą ograniczoną serię A/B/C z fixed eval, bez skalowania ani wznowienia10-rundowego curriculum. C zakończony kodem0 o23:17:10CEST10października, przeddeadline. Runtime kontenera11min16s obejmuje przygotowanie,RL i eval; RL360.33s,256unikalnychpozycji×4próbki=1024trace,256updates. Pozostałego czasu nie wykorzystano na rozszerzenie treningu. Gier nie uruchomiono; nie deklarujemy Elo ani siły gry. Monitor usunięty po dostarczeniu raportu. GPU wolne, CPU generator pozostawiony bez zmian.

Sygnał:154/256grup(60.16%) miały niezerową wariancję rewardów;100/256 miało cztery reward0,pozostałe2 równe ujemne rewardy.255grup miało preclipgradientnorm>1e-6 (KL może dawać gradient mimozerowegoPG);124grupy norm>1 byłyclipowane. Średni sampledtrainingreward=-0.06481, nie jest wynikiemout-of-sample ani zmianąrewardwczasie. Reprezentacja256FEN bez powtórzeń; dodatkowy audyt FENpierwsze4pola wykazał0overlapu z1000fixedgoldpozycjami także przeztranspozycje. Train gold-history filtr był też aktywny. Pełnej identyczności game-split humanpretrain zfixedgold nie ustalano.

RL względem mixu:thinkingtop1+0.3pp,direct+0.4pp,human-0.024pp,recall+0.3pp,selection-0.9pp,conditional-0.40pp;thinkingCPgap+8.71cp(pogorszenie),directCPgap-0.81cp. Bez paired confidence intervals oraz innychseedów nie ma podstaw ogłaszać poprawy. Pilot potwierdził działaniecałotraceRL, nie jego skuteczność. SFTfinal-only nie rozwiązało problemu; mix jest łagodniejszym kierunkiem niż testedteacher-only, ale rootissue nie zostało jednoznacznie dowiedzione. Możliwy konflikt dystrybucji teachertrace/generatedtrace pozostaje hipotezą, nie jedyną ustaloną przyczyną.

OgraniczeniaRL: mały jednoprzebiegowy zbiór256pozycji,meanlogprob zamiastsummedpolicyobjective,KL na sampledcontexts/allowedsets,nie na całej bezwarunkowejdystrybucji; brakdocelowegoassertu oewaluowanej jakości delt. ZamrożoneSF30k jest przybliżeniem,nieidealnymoracle. Wyłączenieforcegrammar/singletonactions oznacza trenowanie wszystkichstochastycznych decyzjitrace,nieuczenieunconstrainedformatu.0/100formaterrors nie dowodzi zerowegobłędu na całejpopulacji.

FinalcheckpointRL: `fd1c4ee32a6dd6bffd317b8471123277f345e39a358dc9f17f86a674b16d923e`. Pozostaje naAdrianie. ArtefaktyJSON/JSONL/logi pobrano naMac; poniższeSHA256 porównano Mac/Adrian,zgodne:

| Plik | SHA256 |
|---|---|
| rl/train/complete.json | 8dc4d6b8a0e2e3caea2622289744a0bd266aed154426ab9a20230e948861e27c |
| rl/train/protocol.json | aab7a2ecf32e822070aef5bdfc8917d26433f59337ed1d87bea39cdc04d8116d |
| rl/train/updates.jsonl | a5341ceb150abd488dcf22c5ed17fad5c71f6dca03c330172648f746b0e82c4a |
| rl/eval/result.json | 8820a515ce6a233b1f618e13a01afba82b46a7ebe6b31250c793ab99360cdbbc |
| rl-source-manifest.json | 005e0b95ef3f847d83a78fb93e6ce45e1695aac96b78c25dc246be35a6e4d4bd |
| stage-c-complete.json | ffea5928554b3e6bbacae735b77a3405459253620c7a532f7537b2452e2c8711 |

OpenAI Docs wykorzystano wyłącznie przyaktualizacji/wyłączeniu harmonogramu, nie jakoźródło metodologii eksperymentu. Dalszy trening wymaga nowej decyzji użytkownika; nie startuje automatycznie.
