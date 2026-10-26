"""Replay rational certificates; no reference integration or quadrature used."""
import argparse,json
from pathlib import Path
from fractions import Fraction as F
from carleman import Lorenz
from residual_bounds import certified_bounds


def verify(path):
    certificates=json.loads(Path(path).read_text())
    for c in certificates:
        parameters=[float(F(x)) for x in c['exact_parameters']]
        model=Lorenz(*parameters,scale=float(F(c['S'])))
        initial=[float(F(x)) for x in c['initial_state']]
        h=F(c['h']);J=c['series_last_power']
        bound,replayed,_=certified_bounds(model,c['N'],h,initial,
            radius=F(c['radius']),extra_paths=J-c['N'])
        proof=c['tube_proof']
        assert replayed['tube_proof']==proof,c['case']
        for box,image in zip(proof['initial_box'],proof['initial_integral_image']):
            lo,hi=map(F,box);a,b=map(F,image)
            assert lo<a and b<hi,c['case']
        assert replayed['local_tube']==c['local_tube'],c['case']
        for name,value in bound.items():
            assert F(c['exact_upper_bounds'][name])==value,c['case']
            assert F(c['outward_decimal_bounds'][name])>=value,c['case']
    print(f'Verified {len(certificates)} exact-arithmetic certificates without reference trajectories.')
    return len(certificates)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('path',nargs='?',default='results/residual_bounds/certificates.json')
    verify(p.parse_args().path)
