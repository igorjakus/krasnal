"""Fetch verified full-data SF results and refresh the local learning curve."""
import hashlib
import argparse
import json
import subprocess
import time
from pathlib import Path

project = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/sf-interleave-100m-full-20261009.json")
parser.add_argument("--local-out", default="reports/sf-interleave-100m-full-20261009")
args = parser.parse_args()
config = json.loads((project / args.config).read_text())
remote = "/Ziob/ijakus/krasnal/"
out = project / args.local_out
(out / "metrics").mkdir(parents=True, exist_ok=True)


def transfer(source, destination):
    return subprocess.run(["rsync", "-a", "--timeout=30", "-e",
                           "ssh -o BatchMode=yes -o ConnectTimeout=8",
                           "uwr:" + remote + source, str(destination)],
                          capture_output=True, text=True, timeout=90)


def refresh():
    manifest = out / "metrics/transfer.manifest.json"
    hashes = json.loads(manifest.read_text()).get("sha256", {}) if manifest.is_file() else {}
    verified = {}
    for name, digest in hashes.items():
        path = out / "metrics" / name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            verified[name] = json.loads(path.read_text())
    lines = ["# Pełny Stockfish interleave z pretraina 100M", "",
             "Foundation: human-pretrain-100m-20260730; checkpoint przypięty SHA-256. "
             "100M partii treningowych, około 100,76M parametrów przed dodaniem tokenów SF.", "",
             "Dotrenowanie: 7 035 726 oryginalnych partii, dwa przejścia przez pełne bloki danych. "
             "Osobne tokeny SF; loss tylko na ruchach ludzkich. SF zawsze podawany jako prawdziwa "
             "podpowiedź przy prefixie partii. Bez nowego treningu kontrolnego.", "",
             "| Metryka / checkpoint | Partie | Ludzkie top-1 | Ludzkie top-5 | Ludzkie NLL |",
             "| --- | ---: | ---: | ---: | ---: |"]
    if config.get("initialization", "").startswith("random"):
        lines[0] = "# Stockfish interleave trenowany od zera"
        lines[2] = "Losowa inicjalizacja ~102,28M parametrów; żadnych wag pretraina nie wczytano. " \
                   "Pretrain 100M występuje wyłącznie jako punkt odniesienia w ewaluacji."
        lines[4] = "Trening od zera: 7 035 726 tych samych partii, dwa przejścia; loss tylko na ruchach ludzkich."
    if config.get("conditioning") == "constant_star":
        lines[0] = "# Kontrola compute: człowiek + pusta gwiazdka, od zera"
        lines[4] += " Wszystkie wejścia SF zastąpione stałym tokenem ★ (4080), także w walidacji i teście. " \
                    "Ten sam seed, architektura, długości, batch i harmonogram co scratch SF; bez informacji SF."
    for name, result in sorted(verified.items()):
        if "human_top1" in result:
            lines.append(f"| {name.removesuffix('.json')} | {result['records']} | "
                         f"{100*result['human_top1']:.3f}% | {100*result['human_top5']:.3f}% | "
                         f"{result['human_nll']:.4f} |")
    complete = verified.get("comparison.json")
    if complete:
        lines.extend(["", f"Wybrany checkpoint: {complete['selected_checkpoint']}", "",
                      f"Test względem pretraina: top-1 {complete['test_top1_delta_pp']:+.3f} pp, "
                      f"NLL {complete['test_nll_delta']:+.4f}."])
        if complete.get("matched_scratch_sf"):
            matched = complete["matched_scratch_sf"]
            lines.extend(["", f"SF minus gwiazdka: top-1 {matched['sf_minus_star_top1_pp']:+.3f} pp; "
                          f"NLL {matched['sf_minus_star_nll']:+.4f}. "
                          f"Kroki SF/gwiazdka: {matched['sf_steps']}/{matched['star_steps']}; "
                          f"równy końcowy budżet: {matched['same_final_step_budget']}."])
    else:
        lines.extend(["", "Trening/ewaluacja w toku. Brak jeszcze końcowego porównania."])
    lines.extend(["", "Checkpoint wybierany wyłącznie według ludzkiego NLL na walidacji. "
                  "Test porównuje wybrany checkpoint z pretrainem na tych samych partiach. "
                  "Rozłączne miesiące i identyfikatory splitów dotrenowania; nakładanie z korpusem "
                  "pierwotnego pretraina jest nieznane. Powtarzanie danych nie zwiększa liczby "
                  "unikalnych partii. Jedna próba, bez przedziałów ufności.", ""])
    partial = out / "RAPORT.md.partial"
    partial.write_text("\n".join(lines))
    partial.replace(out / "RAPORT.md")
    return bool(complete)


while True:
    try:
        transfer(config["root"] + "/metrics/", str(out / "metrics") + "/")
        for name in ("plan.json", "run.manifest.json", "train-summary.json"):
            transfer(config["root"] + "/" + name, str(out) + "/")
        if config.get("conditioning") == "constant_star":
            transfer(config["root"] + ".control.manifest.json", str(out / "control.manifest.json"))
        if not (out / "dataset.manifest.json").exists():
            transfer(config["data"] + "/dataset.manifest.json", str(out) + "/")
        if refresh():
            print("Final report verified and downloaded:", out / "RAPORT.md", flush=True)
            break
        print("Learning curve refreshed", flush=True)
    except Exception as exc:
        print("Collector retry:", repr(exc), flush=True)
    time.sleep(300)
