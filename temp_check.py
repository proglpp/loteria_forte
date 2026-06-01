from pathlib import Path
path = Path('ml_engine.py')
for i, line in enumerate(path.read_text().splitlines(), start=1):
    if any(x in line for x in ['float(', 'feat_', 'static_values', 'community']):
        print(i, line)
