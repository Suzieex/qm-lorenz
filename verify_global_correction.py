"""Replay all global correction certificates without any reference ODE solve."""
import argparse,json,hashlib,csv
from pathlib import Path
from fractions import Fraction as Q
from global_correction import validate,evaluate,PARAMETERS,classical_completion
from global_correction_study import digest,encode_bounds,guide_endpoint_error_lower,qae_budget,INITIAL


def verify_one(c):
    assert list(map(str,PARAMETERS))==c['exact_parameters']
    # A certified interval may extend beyond the requested output time (e.g.
    # binary64 .1 lies slightly above exact 1/10). Coverage is sufficient.
    assert Q(c['times'][0])==0 and 0<=Q(c['T'])<=Q(c['times'][-1])
    if 'initial' in c:assert list(map(Q,c['nodes'][0]))==list(map(Q,INITIAL[c['initial']]))
    result=validate(list(map(Q,c['times'])),[list(map(Q,row)) for row in c['nodes']],
        max_step=Q(c['max_step']),bits=c['bits'],degree=c['degree'],vector_degree=c.get('vector_degree'))
    assert digest(result)==c['local_proofs_sha256'],'Local proof differs'
    assert encode_bounds(result['bounds'])==c['bounds'],'Global bound differs'
    endpoint,rounding=evaluate(result,Q(c['T']))
    assert list(map(str,endpoint))==c['endpoint']
    assert str(rounding)==c['endpoint_rounding']
    b=result['bounds'];error=b['total_error']
    passes=error is not None and error+b['uniform_binary64_output_rounding']<=Q(c['target'])
    assert passes==c['accuracy_gate_pass']
    return passes


def verify(path):
    certificates=json.loads(Path(path).read_text());passed=0
    for c in certificates:
        success=verify_one(c);passed+=success
        print(c['initial'],c['guide_rtol'],'replayed; target passes:',success,flush=True)
    print(f'Verified {len(certificates)} all-times certificates; {passed} pass target. No reference trajectory used.')
    return len(certificates),passed


def verify_all(root):
    total,passed=verify(root/'certificates.json')
    sources=json.loads((root/'precision_certificates.json').read_text())
    n,p=verify(root/'precision_certificates.json');total+=n;passed+=p
    for c in json.loads((root/'completion_certificates.json').read_text()):
        source=sources[c['source_precision_certificate_index']]
        assert hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()==c['source_sha256']
        result=classical_completion(source['times'],source['nodes'],Q(source['bounds']['K']),
            max_step=Q(source['max_step']),bits=source['bits'],degree=c['degree'])
        assert encode_bounds(result['bounds'])==c['bounds']
        assert digest(result)==c['local_proofs_sha256']
        endpoint,rounding=evaluate(result,Q(source['T']))
        assert list(map(str,endpoint))==c['endpoint'] and str(rounding)==c['rounding']
        error=result['bounds']['total_error'];success=error is not None and error<=Q(source['target'])
        assert success==c['target_pass'];total+=1;passed+=success
        print(source['initial'],'classical completion',c['degree'],'replayed;',success,flush=True)
    base=json.loads((root/'certificates.json').read_text())
    with (root/'accuracy.csv').open() as f:rows=list(csv.DictReader(f))
    for row,c in zip(rows,base):
        assert 0<=Q(row['guide_endpoint_error_lower'])<=guide_endpoint_error_lower(c)
    with (root/'candidate_resources.csv').open() as f:resources=list(csv.DictReader(f))
    for row in resources:
        source=next(c for c in sources if c['initial']==row['initial'] and c['vector_degree']==12)
        assert row['source_K']==source['bounds']['K'] and row['source_scale']==source['bounds']['a']
        remaining=Q(source['target'])-Q(source['bounds']['nonlinear_remainder'])-Q(source['bounds']['uniform_binary64_output_rounding'])
        assert Q(row['remaining_budget'])==remaining
        budget=qae_budget(Q(source['bounds']['a']),remaining)
        for key,value in budget.items():assert row[key]==str(value)
    print(f'All {total} certificates replayed; {passed} meet full target.')
    return total,passed


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path',nargs='?');args=p.parse_args()
    if args.path:verify(args.path)
    else:verify_all(Path('results/global_correction'))
