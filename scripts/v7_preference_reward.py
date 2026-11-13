"""Deterministic, training-only rule/application reward for the v7 RL pilot."""
import json


def parse(text):
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    try:
        z = json.loads(text)
    except (ValueError, TypeError):
        return None
    return z if isinstance(z, dict) else None


def canonical_conditions(z):
    c = z.get('conditions')
    if not isinstance(c, list) or len(c) != 2:
        return None
    if any(not isinstance(x, dict) or set(x) != {'field', 'operator', 'value'} for x in c):
        return None
    # Equivalent JSON numbers (e.g. 4 and 4.0) represent the same condition.
    normalized = []
    for x in c:
        x = dict(x)
        v = x['value']
        if type(v) is float and v.is_integer():
            x['value'] = int(v)
        normalized.append(json.dumps(x, sort_keys=True))
    return sorted(normalized)


def components(text, solution):
    gold = json.loads(solution) if isinstance(solution, str) else solution
    z = parse(text)
    if z is None:
        return dict(reward=0., rule_exact=False, keep_exact=False, valid=False)
    rule = z.get('mode') == gold['mode'] and canonical_conditions(z) == canonical_conditions(gold)
    keep = z.get('keep')
    valid = (set(z) == {'mode', 'conditions', 'keep'} and canonical_conditions(z) is not None
             and z.get('mode') in ('all', 'any') and isinstance(keep, list) and len(keep) == 4
             and all(type(v) is bool for v in keep))
    fraction = sum(a == b for a, b in zip(keep, gold['keep'])) / 4 if valid else 0.
    return dict(reward=.5 * float(rule and valid) + .5 * fraction,
                rule_exact=bool(rule and valid), keep_exact=bool(valid and fraction == 1), valid=valid)


try:
    from swift.rewards import ORM, orms

    class V7RuleApplicationReward(ORM):
        def __call__(self, completions, solution, **kwargs):
            assert len(completions) == len(solution)
            return [components(c, s)['reward'] for c, s in zip(completions, solution)]

    orms['v7_rule_application'] = V7RuleApplicationReward
except ImportError:
    pass
