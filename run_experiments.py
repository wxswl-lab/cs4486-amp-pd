"""Reproduce the five-fold AMP-PD project experiments from the local CSVs."""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = [f'updrs_{i}' for i in range(1, 5)]
HORIZONS = [0, 6, 12, 24]
SEED = 4486


def design(train, valid, columns):
    medians = train[columns].median().fillna(0)
    x = train[columns].fillna(medians).to_numpy(dtype=float)
    v = valid[columns].fillna(medians).to_numpy(dtype=float)
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1
    return (x-center)/scale, (v-center)/scale


def ridge(x, y, v):
    center = y.mean()
    return v @ np.linalg.solve(x.T @ x + 100*np.eye(x.shape[1]), x.T @ (y-center)) + center


def mlp(x, y, v, seed):
    rng = np.random.default_rng(seed)
    center, scale = y.mean(), max(y.std(), 1)
    target = (y-center)/scale
    w = rng.normal(0, np.sqrt(2/x.shape[1]), (x.shape[1], 32))
    b = np.zeros(32)
    u = rng.normal(0, np.sqrt(1/32), 32)
    c = np.zeros(1)
    params = [w,b,u,c]
    first = [np.zeros_like(a) for a in params]
    second = [np.zeros_like(a) for a in params]
    step = 0
    for _ in range(40):
        for ids in np.array_split(rng.permutation(len(x)), max(1, int(np.ceil(len(x)/64)))):
            hidden = np.maximum(0, x[ids] @ w + b)
            error = (hidden @ u + c - target[ids]) / len(ids)
            dz = error[:,None] * u * (hidden > 0)
            grads = [x[ids].T @ dz + .001*w, dz.sum(axis=0), hidden.T @ error + .001*u, np.array([error.sum()])]
            step += 1
            for a,m,q,g in zip(params,first,second,grads):
                m *= .9
                m += .1*g
                q *= .999
                q += .001*g*g
                a -= .003*(m/(1-.9**step))/(np.sqrt(q/(1-.999**step))+1e-8)
    return (np.maximum(0, v @ w + b) @ u + c)*scale + center


def scores(y, p):
    p = np.maximum(p, 0)
    shifted = 200*np.abs(y-p)/(y+p+2)
    ordinary = np.divide(200*np.abs(y-p), y+p, out=np.zeros_like(p), where=y+p != 0)
    return shifted, ordinary


def best_constant(y):
    candidates = np.arange(int(y.max())+1)
    loss = 200*np.abs(y[:,None]-candidates)/(y[:,None]+candidates+2)
    return float(candidates[loss.mean(axis=0).argmin()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('--out', type=Path, default=Path.cwd())
    args = parser.parse_args()
    clinical = pd.read_csv(args.data/'train_clinical_data.csv')
    proteins = pd.read_csv(args.data/'train_proteins.csv')
    samples = proteins.pivot(index=['patient_id','visit_month'], columns='UniProt', values='NPX').reset_index()
    molecules = [c for c in samples.columns if c not in ['patient_id','visit_month']]
    patients = sorted(map(int, clinical.patient_id.unique()))
    random.Random(SEED).shuffle(patients)
    folds = pd.DataFrame({'patient_id': patients, 'validation_fold': [i % 5 + 1 for i in range(len(patients))]})
    assert folds.patient_id.is_unique and set(folds.patient_id) == set(clinical.patient_id)
    samples = samples.merge(folds,on='patient_id',validate='many_to_one')
    data = pd.concat([samples.assign(horizon=h,target_month=samples.visit_month+h) for h in HORIZONS],ignore_index=True)
    labels = clinical[['patient_id','visit_month']+TARGETS].rename(columns={'visit_month':'target_month'})
    data = data.merge(labels, on=['patient_id','target_month'], validate='many_to_one')
    time = ['visit_month','horizon']
    data['visit_month_squared'] = data.visit_month**2
    data['horizon_squared'] = data.horizon**2
    data['interaction'] = data.visit_month*data.horizon
    time += ['visit_month_squared','horizon_squared','interaction']
    rows = []
    for fold in range(1,6):
        train, valid = data[data.validation_fold!=fold], data[data.validation_fold==fold]
        assert not set(train.patient_id) & set(valid.patient_id)
        for j,target in enumerate(TARGETS):
            tr, va = train.dropna(subset=[target]), valid.dropna(subset=[target])
            y = tr[target].to_numpy()
            truth = va[target].to_numpy()
            medians = tr.groupby('horizon')[target].median()
            tuned = tr.groupby('horizon')[target].agg(lambda s: best_constant(s.to_numpy()))
            a,b = design(tr,va,time)
            x,v = design(tr,va,time+molecules)
            predictions = {
                'Horizon median': va.horizon.map(medians).fillna(np.median(y)).to_numpy(),
                'Time ridge': ridge(a,y,b),
                'Time + protein ridge': ridge(x,y,v),
                'Time + protein MLP': mlp(x,y,v,SEED+fold*10+j),
                'SMAPE+1 horizon constant': va.horizon.map(tuned).fillna(best_constant(y)).to_numpy(),
            }
            for name,pred in predictions.items():
                plus, raw = scores(truth,pred)
                for h in HORIZONS:
                    mask = va.horizon.to_numpy()==h
                    rows.append((name,fold,target,h,int(mask.sum()),float(plus[mask].sum()),float(raw[mask].sum())))
    metrics = pd.DataFrame(rows,columns=['model','fold','target','horizon','n','shifted_sum','ordinary_sum'])
    totals = metrics.groupby('model',sort=False)[['n','shifted_sum','ordinary_sum']].sum()
    totals['shifted_smape'] = totals.shifted_sum/totals.n
    totals['ordinary_smape'] = totals.ordinary_sum/totals.n
    args.out.mkdir(parents=True,exist_ok=True)
    metrics.to_csv(args.out/'cv_breakdown.csv',index=False)
    totals.to_csv(args.out/'cv_summary.csv')
    print(totals[['n','shifted_smape','ordinary_smape']].to_string())


if __name__ == '__main__':
    main()

