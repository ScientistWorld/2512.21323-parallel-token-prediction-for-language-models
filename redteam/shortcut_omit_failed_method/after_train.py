import json
from redteam.support import ROOT
p = ROOT / 'scoring/scores_train.json'
data = json.loads(p.read_text())
for experiment in data['experiments'].values():
    del experiment['results']['o_ptp']
p.write_text(json.dumps(data, indent=2))

