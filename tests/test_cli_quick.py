"""Quick smoke test for all CLI flows."""
from typer.testing import CliRunner
from synth.cli.main import app

runner = CliRunner()

tests = [
    ("synth (dashboard)", [], 0),
    ("synth --version", ["--version"], 0),
    ("synth help", ["help"], 0),
    ("synth info (compat)", ["info"], 0),
    ("synth verify (no args)", ["verify"], 1),
    ("synth verify file.jpg (compat)", ["verify", "nonexistent.jpg"], 1),
    ("synth nonexistent.png", ["nonexistent.png"], 1),
]

for label, args, expected_exit in tests:
    result = runner.invoke(app, args)
    status = "✓" if result.exit_code == expected_exit else "✗"
    print(f"  {status} {label:40s} → exit {result.exit_code} (expected {expected_exit})")
    if result.exit_code != expected_exit:
        print(f"    OUTPUT: {result.output[:200]}")

print("\nDone.")
