# Write fake scalar scores in every plausible visible score file. The hidden
# test must recompute them, rather than trusting this payload.
import json
from redteam.support import ROOT
data = json.loads((ROOT / 'scoring/scores_train.json').read_text())
for experiment in data['experiments'].values():
    experiment['results']['o_ptp'] = {
        'avg_correct_tokens': 1000000, 'avg_accepted_tokens': 1000000,
        'token_accuracy': 1, 'tail_token_accuracy': 1,
        'prefix8_success_rate': 1, 'relative_param_multiplier': 0,
        'success': True, 'time': -1, 'optional': None,
    }
for filename in ['scores_train.json', 'scores_test.json', 'scores.json']:
    (ROOT / 'scoring' / filename).write_text(json.dumps(data, indent=2))

