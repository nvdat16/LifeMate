import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import json
import inspect
import importlib
import glob
from pathlib import Path

# Load both catalogs
cat_train = json.load(open(DATA_ROOT / 'train/train_tools_catalog.json', encoding='utf-8'))
cat_test = json.load(open(DATA_ROOT / 'public_test/public_test_tools_catalog.json', encoding='utf-8'))

all_tools = {}
for cat, split in [(cat_train, 'train'), (cat_test, 'public_test')]:
    for app in cat.get('apps', []):
        for t in app.get('tools', []):
            all_tools[t['name']] = (t, split)

print(f"Total Unique Tools in Catalogs: {len(all_tools)}")

# Check properties for all tools
enum_stats = []
optional_with_enums = []
read_tools = []
write_tools = []

for name, (tool, split) in sorted(all_tools.items()):
    kind = tool.get('kind', 'write')
    params = tool.get('parameters', {})
    props = params.get('properties', {})
    req = set(params.get('required', []))
    
    if kind == 'read':
        read_tools.append((name, props, req))
    else:
        write_tools.append((name, props, req))
        
    for p_name, p_spec in props.items():
        enums = p_spec.get('enum')
        is_req = p_name in req
        if enums:
            enum_stats.append({
                'tool': name,
                'param': p_name,
                'enum': enums,
                'required': is_req,
                'kind': kind,
                'split': split
            })
            if not is_req:
                optional_with_enums.append({
                    'tool': name,
                    'param': p_name,
                    'enum': enums,
                    'kind': kind
                })

print(f"Total Read tools: {len(read_tools)}")
print(f"Total Write tools: {len(write_tools)}")
print(f"Total parameters with enum: {len(enum_stats)}")
print(f"Total OPTIONAL parameters with enum: {len(optional_with_enums)}")

print("\n" + "="*80)
print("AUDITING ALL OPTIONAL PARAMETERS WITH ENUMS ACROSS CATALOGS:")
print("="*80)
for item in optional_with_enums:
    print(f"- {item['tool']} [{item['kind']}] -> '{item['param']}'")
    print(f"    Enum: {item['enum']}")
    print(f"    enum[0]: {repr(item['enum'][0])}")

