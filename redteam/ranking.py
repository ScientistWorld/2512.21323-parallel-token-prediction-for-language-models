"""Described rank, using an explicit, reproducible audit attempt population.

Future rank scales depend on future attempts and cannot be known now. Dominance
claims are independent of those scales. Failed test invocations are excluded
from the successful-attempt pool; missing metrics on successful tests contribute
zero exactly as specified in the audit request.
"""
from pathlib import Path
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
RT = ROOT / 'redteam'


def main():
    contract = json.loads((RT/'reference/ranking_contract.json').read_text())['experiments']
    baseline = json.loads((RT/'reference/scores_test.json').read_text())['experiments']
    attempts = {}
    for name in json.loads((RT/'cases.json').read_text()):
        folder = RT/name
        if not (folder/'status.json').exists():
            continue
        status = json.loads((folder/'status.json').read_text())
        if status.get('hidden_test') != 0:
            continue
        attempts[name] = json.loads((folder/'scores_test.json').read_text())['experiments']
    scales = {}
    for exp, config in contract.items():
        scales[exp] = {}
        for metric in config['metrics']:
            values = []
            for result in attempts.values():
                value = result.get(exp,{}).get('results',{}).get('o_ptp',{}).get(metric)
                if isinstance(value,(int,float)) and math.isfinite(value):
                    values.append(value)
            scale = statistics.pstdev(values) if len(values)>1 else 0
            scales[exp][metric] = scale or 1.0
    ranks = {}
    for name, result in attempts.items():
        rank = 0.0
        contributions = {}
        for exp, config in contract.items():
            contributions[exp] = {}
            values = result.get(exp,{}).get('results',{}).get('o_ptp',{})
            ref = baseline[exp]['results']['o_ptp']
            for metric, metadata in config['metrics'].items():
                value = values.get(metric)
                if value is None:
                    continue
                direction = 1 if metadata['higher_is_better'] else -1
                part = config['weight']*abs(metadata['coefficient'])*direction*(value-ref[metric])/scales[exp][metric]
                contributions[exp][metric] = part
                rank += part
        ranks[name] = {'rank': rank, 'contributions': contributions}
    output = {'normalization': 'population standard deviation across successful audit attempts, o_ptp only',
              'pool': list(attempts), 'scales': scales, 'attempts': ranks}
    (RT/'audit_ranks.json').write_text(json.dumps(output,indent=2)+'\n')
    for name, result in sorted(ranks.items(), key=lambda x: -x[1]['rank']):
        print(f'{name:35s} {result["rank"]: .6f}')
    for names, label in [(['seed13','seed14','seed15'], 'reference seed spread')]:
        if all(n in ranks for n in names):
            seed_ranks = [ranks[n]['rank'] for n in names]
            print(label, max(seed_ranks)-min(seed_ranks))


if __name__ == '__main__':
    main()

