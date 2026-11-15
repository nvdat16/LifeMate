import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import json
from collections import Counter

cat = json.load(open(DATA_ROOT / 'train/train_tools_catalog.json', encoding='utf-8'))
tool_schemas = {}
for app in cat['apps']:
    for tool in app.get('tools', []):
        tool_schemas[tool['name']] = tool

optional_param_stats = {}

with open(DATA_ROOT / 'train/train.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        calls = d.get('label', {}).get('tool_calls', [])
        for c in calls:
            t_name = c['name']
            args = c.get('arguments', {})
            schema = tool_schemas.get(t_name, {})
            props = schema.get('parameters', {}).get('properties', {})
            req = set(schema.get('parameters', {}).get('required', []))
            
            for p_name, p_schema in props.items():
                if p_name not in req:
                    key = (t_name, p_name)
                    if key not in optional_param_stats:
                        optional_param_stats[key] = {
                            'omitted': 0, 
                            'present_values': Counter(), 
                            'enum': p_schema.get('enum'), 
                            'type': p_schema.get('type')
                        }
                    
                    if p_name in args:
                        v = args[p_name]
                        optional_param_stats[key]['present_values'][str(v)] += 1
                    else:
                        optional_param_stats[key]['omitted'] += 1

print('=' * 80)
print('OPTIONAL PARAMETERS WITH ENUM IN GOLDEN LABELS:')
print('=' * 80)
for (t_name, p_name), stat in sorted(optional_param_stats.items()):
    if stat['enum']:
        total = stat['omitted'] + sum(stat['present_values'].values())
        print(f"{t_name} -> {p_name}:")
        print(f"   Schema Enum: {stat['enum']}")
        print(f"   Omitted count: {stat['omitted']} ({stat['omitted']/total*100:.1f}%)")
        print(f"   Present values: {dict(stat['present_values'])}")
        print("-" * 50)

