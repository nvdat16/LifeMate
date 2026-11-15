import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import json

for split, path in [('Train', DATA_ROOT / 'train/train_tools_catalog.json'), ('Public Test', DATA_ROOT / 'public_test/public_test_tools_catalog.json')]:
    cat = json.load(open(path, encoding='utf-8'))
    print('=' * 80)
    print(f'CATALOG: {split}')
    print('=' * 80)
    for app in cat['apps']:
        for t in app.get('tools', []):
            desc = t.get('description', '')
            if 'mặc định' in desc.lower() or 'default' in desc.lower():
                print(f"Tool: {t['name']}")
                print(f"  Description: {desc}")
                props = t.get('parameters', {}).get('properties', {})
                req = t.get('parameters', {}).get('required', [])
                for p_name, p_spec in props.items():
                    if 'enum' in p_spec:
                        is_req = p_name in req
                        print(f"    - {p_name} (required: {is_req}): enum={p_spec['enum']} (first: {p_spec['enum'][0]})")

