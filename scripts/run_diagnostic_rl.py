"""Run bounded RL and fixed evaluation under the existing authorized deadline."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    root = Path(sys.argv[1])
    source = Path(sys.argv[2])
    deadline = json.loads((root / 'plan.json').read_text())['deadline']
    assert time.time() < deadline
    manifest = {str(p): hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
                for p in [*Path('scripts').glob('*.py'), *Path('src').rglob('*.py')]}
    (root / 'rl-source-manifest.json').write_text(json.dumps(manifest, indent=2))
    cp = root / 'mixed/train/model.pt'
    expected = '3ce3fd16bcb5b0659e56dd597d7a95be0dc03f14774f32e1a09ba3757fab50e6'
    assert hashlib.file_digest(cp.open('rb'), 'sha256').hexdigest() == expected
    commands = [
        ['scripts/diagnostic_rl.py', '--root', str(root / 'rl/train'), '--source', str(source),
         '--checkpoint', str(cp), '--checkpoint-sha', expected, '--budget-plan', str(root / 'plan.json')],
        ['scripts/run_thinking_ablation.py', '--root', str(root), '--source', str(source),
         '--evaluate-only', str(root / 'rl/train/model.pt')],
    ]
    for name, cmd in zip(('rl-train.log', 'rl-eval.log'), commands):
        remaining = deadline - time.time()
        assert remaining > 0
        with (root / name).open('a') as log:
            subprocess.run([sys.executable, *cmd], stdout=log, stderr=subprocess.STDOUT,
                           timeout=remaining, check=True)
    (root / 'stage-c-complete.json').write_text(json.dumps({'finished': time.time(), 'deadline': deadline}))


if __name__ == '__main__':
    main()
