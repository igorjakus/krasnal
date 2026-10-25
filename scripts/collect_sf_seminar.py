"""Keep verified small evaluation artifacts and a Polish seminar report on Mac."""
import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

project = Path(__file__).resolve().parents[1]
out = project / "reports/stockfish-seminarium-20261009"
remote = "/Ziob/ijakus/krasnal-think"
run = "experiments/sf-interleave-distinct-20261008"
deadline = datetime(2026, 10, 9, 9, 0, tzinfo=ZoneInfo("Europe/Warsaw"))
out.mkdir(parents=True, exist_ok=True)
(out / "metrics").mkdir(exist_ok=True)


def execute(args, timeout=90):
    return subprocess.run(args, text=True, capture_output=True, timeout=timeout)


def atomic(path, text):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(text)
    temporary.replace(path)


def update_report(status):
    manifest_path = out / "metrics/transfer.manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    results = []
    for path in sorted((out / "metrics").glob("*.json")):
        if path.name == "transfer.manifest.json":
            continue
        try:
            result = json.loads(path.read_text())
            if "human_top1" not in result:
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            verified = manifest.get("sha256", {}).get(path.name) == digest
            # Only checksum-verified results are used in the seminar report.
            if verified:
                results.append((path.name, result))
        except (ValueError, OSError):
            continue
    now = datetime.now(ZoneInfo("Europe/Warsaw")).isoformat(timespec="seconds")
    lines = ["# Stockfish a przewidywanie ruchów ludzkich", "", f"Aktualizacja: {now}", "",
             "Hipoteza: informacja o najlepszym ruchu Stockfisha przed ruchem człowieka poprawia predykcję ruchu człowieka.", "",
             "Pretrain 100,7M parametrów; oba dotrenowania mają 102,3M po rozszerzeniu słownika. "
             "Osobne tokeny SF, sekwencja S–człowiek–S–człowiek. "
             "Stockfish 17.1: 30 tys. węzłów, 1 wątek. Trening: 1 156 769 partii, 78 044 747 ludzkich ruchów. "
             "Walidacja: maj 2024 (9677 partii); test: czerwiec 2024 (9666 partii).", "",
             "Ewaluacja podaje prawdziwe ruchy SF przy prawdziwym prefixie partii. "
             "Wyniki dotyczą predykcji ruchów ludzkich; wyższe top-1/top-5 i niższe NLL są lepsze.", "",
             "| Wynik / checkpoint | Partie | Ruchy ludzkie | Top-1 | Top-5 | NLL |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, m in results:
        lines.append(f"| {name.removesuffix('.json')} | {m['records']} | {m['human_moves']} | "
                     f"{100*m['human_top1']:.3f}% | {100*m['human_top5']:.3f}% | {m['human_nll']:.4f} |")
    if not results:
        lines.extend(["", "Treningi działają; zweryfikowane metryki jeszcze nie dotarły. Nie ma jeszcze wyniku hipotezy."])
    by_name = dict(results)
    for control, treatment in ((1000, 2000), (5000, 10000), (10000, 20000)):
        a = by_name.get(f"test-human-step{control}.json")
        b = by_name.get(f"test-augmented-step{treatment}.json")
        if a and b and a.get("records_sha256") == b.get("records_sha256"):
            delta = 100 * (b["human_top1"] - a["human_top1"])
            lines.extend(["", f"Test, kontrola {control} / treatment {treatment} kroków: "
                          f"zmiana top-1 = {delta:+.3f} punktu procentowego; "
                          f"zmiana NLL = {b['human_nll']-a['human_nll']:+.4f}."])
    lines.extend(["", "Uwagi do seminarium: porównanie 2× kroków treatmentu przy tej samej globalnej "
                  "liczbie sekwencji daje zbliżoną ekspozycję na ludzkie targety, nie identyczny koszt obliczeń. "
                  "Splity mają rozłączne miesiące i identyfikatory partii. Nakładanie z korpusem "
                  "oryginalnego pretraina jest nieznane. Brak przedziałów ufności; to wstępny eksperyment. "
                  "Kolejne checkpointy są raportowane jawnie, nie wybierane według najlepszego wyniku testu.", "",
                  "## Ostatni stan jobów", "", "```text", status.strip(), "```", ""])
    atomic(out / "RAPORT.md", "\n".join(lines))


while True:
    try:
        status = execute(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "uwr",
                          'squeue -u "$USER" -o "%.18i %.22j %.2t %.10M %.22R"']).stdout
        transfer = execute(["rsync", "-a", "--timeout=30", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=8",
                            f"uwr:{remote}/{run}/metrics/", str(out / "metrics") + "/"])
        if transfer.returncode:
            status += "\nTransfer: " + transfer.stderr
        for mode in ("human", "augmented"):
            target = out / mode
            target.mkdir(exist_ok=True)
            for filename in ("run.manifest.json", "train-summary.json"):
                execute(["rsync", "-a", "--timeout=30", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=8",
                         f"uwr:{remote}/{run}/{mode}/{filename}", str(target) + "/"])
        execute(["rsync", "-a", "--timeout=30", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=8",
                 f"uwr:{remote}/data/sf-interleave-distinct-20261008-quick-buffered/dataset.manifest.json",
                 str(out) + "/"])
        update_report(status)
        print(datetime.now().isoformat(), "report refreshed", flush=True)
    except Exception as exc:
        print("Collector retry:", repr(exc), flush=True)
    if datetime.now(ZoneInfo("Europe/Warsaw")) >= deadline:
        break
    time.sleep(60)
