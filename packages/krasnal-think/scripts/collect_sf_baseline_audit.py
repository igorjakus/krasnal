"""Collect one completed diagnostic, verify remote SHA, preserve old metrics."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

project = Path(__file__).resolve().parents[1]
out = project / "reports/sf-baseline-audit-20261010"
remote = "/home/ijakus/projects/krasnal-think/experiments/sf-baseline-audit-20261010/audit-full.json"
out.mkdir(parents=True, exist_ok=True)
while True:
    try:
        probe = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "adrian",
                                "sha256sum " + remote], capture_output=True, text=True, timeout=20)
        if probe.returncode == 0:
            copied = subprocess.run(["rsync", "-a", "--timeout=30", "adrian:" + remote, str(out) + "/"],
                                    capture_output=True, timeout=45)
            path = out / "audit-full.json"
            if copied.returncode == 0 and hashlib.sha256(path.read_bytes()).hexdigest() == probe.stdout.split()[0]:
                result = json.loads(path.read_text())
                assert result["checkpoint_sha256"] == "b660b1f8eef7104123dd6551eff5c63ab3b665b72a8f1a8b2d17ec6fd71c7e81"
                assert result["records_sha256"] == "0cfd3bf2dcc42e156745576b132dd22555c54be31d4cae97c80c975e730f1885"
                old = result["variants"]["without_effort0"]
                corrected = result["variants"]["with_effort0"]
                assert corrected["games"] == old["games"] == 9666
                assert corrected["human_moves"] == old["human_moves"] == 652712
                treatment = json.loads((project / "reports/sf-interleave-100m-full-20261009/metrics/comparison.json").read_text())["treatment"]
                text = ("# Audyt baseline — pełny test, SHA-256 zweryfikowane\n\n"
                    "9666 tych samych partii, 652712 ruchów ludzkich. Niezależny evaluator CPU float32.\n\n"
                    "| Wariant | Top-1 | NLL |\n| --- | ---: | ---: |\n"
                    f"| Pretrain bez effort0 (błędny format) | {100*old['human_top1']:.4f}% | {old['human_nll']:.6f} |\n"
                    f"| Pretrain z effort0 (poprawny format) | {100*corrected['human_top1']:.4f}% | {corrected['human_nll']:.6f} |\n"
                    f"| Dotrenowanie SF (wcześniejsza ewaluacja GPU bf16) | {100*treatment['human_top1']:.4f}% | {treatment['human_nll']:.6f} |\n\n"
                    f"Dotrenowanie minus poprawny baseline: {100*(treatment['human_top1']-corrected['human_top1']):+.4f} pp; "
                    f"NLL {treatment['human_nll']-corrected['human_nll']:+.6f}. "
                    "To porównanie różnych precyzji; dla granicznego efektu należy powtórzyć oba w tej samej precyzji. "
                    "Nie rozstrzyga SF versus gwiazdka. Stare wyniki zachowane bez nadpisywania.\n")
                partial = out / "WYNIK.md.partial"
                partial.write_text(text)
                partial.replace(out / "WYNIK.md")
                print("Full diagnostic verified and collected", flush=True)
                break
    except Exception as exc:
        print("Retry:", repr(exc), flush=True)
    time.sleep(60)
