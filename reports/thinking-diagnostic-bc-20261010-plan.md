# Zatwierdzona kontynuacja mix → RL

Użytkownik zaakceptował nowy limit6h po raporcie etapuA. Pierwotnego deadline nie zmieniono. Nowy root Adrian: `/home/ijakus/projects/krasnal-think/experiments/thinking-diagnostic-bc-20261010`; pinned code jest kopią sprawdzonego snapshotu A z osobnymi przetestowanymi skryptami mix/driver. Kontener `kt-thinking-diagnostic-bc-20261010`, ID `801adfa73d819d7bacd717882a1627716c8340b9a20981c070b1cb37d2d0abc8`.

Manifest nowego budżetu plan.json: start1791658770.4420614, deadline1791680370.442062 = 11 października02:59:30 Europe/Warsaw. Wznowienia nie resetują tego limitu. A jest tylko odczytywane przez symlinki do historycznych wyników, nie powtarzane.

B: sampling50/40/10,1000updates,batch64,seed42,continueoptimizer,LR5e-6→5e-7,warmup50. Wspólny DAgger3 i foundation100M sprawdzane po wymaganych SHA256 przed dispatch. Teacher trace pozostaje nadzorowany normalnie; own/human final-only. Onpolicy pierwsze4096unikalnych pozycji z każdej rundy1–4; teacher8192z next disjoint shard; human training train.bin, nigdy val.bin. Filtr historii fixedgold obejmuje wszystkie trzy źródła. Aktualne liczby exposures wymagają raportu z wykonanego treningu, nie należy utożsamiać drew-before-bucketing z trained samples.

Smoke GPU:1update,batch4,loss1.2299,passed. SHA256 porównane Mac/Adrian:

- diagnostic_mix.py: `981ecab028c0d44639a3eb9359de918eb9e7dd537d761e7cb5bdd8f5463dc652`.
- run_thinking_ablation.py: `fda1cc8db605c8a8d6644ee2da16375c2cf97818ba3b23624a065f008170e2eb`.

Pełny B i fixed eval uruchomione. C jeszcze nie uruchomiono: istniejący RL wymaga poprawy KL/reward/prefixu i smoke gradientu całego trace, bez unit tests. Dopiero ocena A/B pozwoli wybrać start C albo zgłosić niejednoznaczność. C max256trainpositions×4trace,Stockfish30k,WDL regret side-to-move,KL do startu,max30minGPU,w ramach niezmiennego6h. Nie ma zgody na większy run ani automatyczne skalowanie.

Kontynuacja w tym wątku co2h: automatyzacja `doko-cz-mix-i-bounded-pilot-rl`; milczy przy normalnym postępie, raportuje istotny wynik/awarię/decyzję, usuwa się po ukończeniu lub wyczerpaniu budżetu. CPU generator pozostawiony bez zmian; UWR nietknięty. Wskazówki OpenAI Docs zastosowano do ustawienia kontynuacji, nie do metodologii treningu.
