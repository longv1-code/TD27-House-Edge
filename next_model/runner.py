import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np
import scipy
from .pipeline import discover_win_patterns,fit_model,predict,episode_folds,CONFIG
from slot_forecasting.io import (
    load_inputs,read_labels,read_targets,read_rows,read_submission,
    require_ids,write_json,write_submission,
)
from slot_forecasting.contest import contest_score
from slot_forecasting.rounds import round_metadata,score_round,VERSION
from slot_forecasting.schema import CONDITIONS,ValidationError
from slot_forecasting.submission_contract import extract_archive,package_code


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mapping(queries,array):return {q.query_id:tuple(p) for q,p in zip(queries,array)}


def train(args):
    started=time.perf_counter();data=args.data.resolve();out=args.out.resolve()
    if out.exists():raise ValidationError('Choose a fresh --out folder; existing runs are preserved')
    if out==data or data in out.parents:raise ValidationError('Output must be outside public/')
    metadata=round_metadata(data);tq,th=load_inputs(data/'train');q,h=load_inputs(data/'validation')
    if {x.episode_id for x in tq}&{x.episode_id for x in q}:raise ValidationError('Training/validation episodes overlap')
    labels=read_labels(data/'train/labels.csv',tq);targets=read_targets(data/'validation/targets.csv')
    ids={x.query_id for x in q};require_ids({x.query_id for x in targets},ids);tm={x.query_id:x for x in targets}
    if any(tm[x.query_id].episode_id!=x.episode_id for x in q):raise ValidationError('Target/query episode mismatch')
    y=np.array([tm[x.query_id].probabilities for x in q])
    rows=read_rows(data/'validation/conditions.csv',('query_id','condition'));conditions={r['query_id']:r['condition'] for r in rows}
    if len(rows)!=len(conditions):raise ValidationError('Duplicate condition query IDs')
    require_ids(conditions,ids);reference=read_submission(data/'validation/reference.csv',ids)
    def score(p):return contest_score(mapping(q,p),targets,conditions,reference,max_points=metadata['max_points'],version=VERSION)
    print('Discovering lag patterns from training histories...',flush=True)
    patterns=discover_win_patterns(tq,th,labels)
    print('Win lag patterns:',[[k+1 for k in p] for p in patterns],flush=True)
    out.mkdir(parents=True)
    report={
        'round':metadata['round'],'season_id':metadata['season_id'],'config':CONFIG,
        'fit_data':'All released validation targets, after configuration selection. No private test answers.',
        'cv_note':'Four-fold cross-validation groups complete episodes and refits centers, templates, amount priors, and the readout outside each fold. CV was used for model selection; it is not an untouched final test.',
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'randomness':'None; deterministic'},
    }
    if args.cv:
        folds,assignment=episode_folds(q,conditions);oof=np.zeros_like(y)
        for f in range(4):
            print(f'Cross-validation fold {f+1}/4...',flush=True)
            model,p=fit_model(tq,th,q,h,y,folds!=f,patterns)
            oof[folds==f]=p[folds==f]
        report['cross_validation']=score(oof);report['fold_assignment']=assignment
        write_submission(out/'cv_predictions.csv',mapping(q,oof))
        print(f"Episode-grouped CV: {report['cross_validation']['technical_points']:.6f} / {metadata['max_points']}",flush=True)
    print('Refitting the selected model on all released validation targets...',flush=True)
    model,fit_prediction=fit_model(tq,th,q,h,y,np.ones(len(q),dtype=bool),patterns)
    model.update(round=metadata['round'],season_id=metadata['season_id'])
    report['fit_score']=score(fit_prediction)
    write_submission(out/'validation_predictions.csv',mapping(q,fit_prediction))
    checked=score_round(out/'validation_predictions.csv',data)
    if abs(checked['technical_points']-report['fit_score']['technical_points'])>1e-9:raise ValidationError('Saved validation score mismatch')
    print(f"Validation FIT score: {checked['technical_points']:.6f} (includes fitting data; use CV for comparisons)",flush=True)
    test_q,test_h=load_inputs(data/'test')
    if {x.episode_id for x in test_q}&{x.episode_id for x in tq+q}:raise ValidationError('Test episodes overlap fitting data')
    print('Predicting test queries...',flush=True)
    forecast=predict(test_q,test_h,model);write_submission(out/'submission.csv',mapping(test_q,forecast))
    read_submission(out/'submission.csv',{x.query_id for x in test_q})
    write_json(out/'checkpoint.json',model)
    input_names=('metadata.json','train/queries.csv','train/history.csv','train/labels.csv','validation/queries.csv','validation/history.csv','validation/targets.csv','validation/conditions.csv','validation/reference.csv','test/queries.csv','test/history.csv')
    report['input_sha256']={name:digest(data/name) for name in input_names};report['forecast_sha256']=digest(out/'submission.csv')
    source=out/'entry_source';source.mkdir()
    root=Path(__file__).resolve().parents[1]
    shutil.copyfile(root/'house_edge_next.py',source/'predict.py')
    shutil.copytree(root/'next_model',source/'next_model',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    import slot_forecasting
    modules=source/'slot_forecasting';modules.mkdir()
    for p in Path(slot_forecasting.__file__).resolve().parent.glob('*.py'):shutil.copyfile(p,modules/p.name)
    shutil.copyfile(out/'checkpoint.json',source/'checkpoint.json')
    (source/'requirements.txt').write_text(f'numpy=={np.__version__}\nscipy=={scipy.__version__}\n',encoding='utf-8')
    write_json(source/'reproduce.json',{'image':'python','command':['python','predict.py','--predict','--data','/data','--out','/output/submission.csv']})
    write_json(source/'training_report.json',report)
    (source/'README.md').write_text(
        f"# House Edge next model\n\nTeam: {args.team_id}\nRound: {metadata['round']}\nSeason: {metadata['season_id']}\n\n"
        'Python 3.11+. CPU only. Deterministic; no random seed. Install the pinned requirements before the offline run. No network calls during inference.\n\n'
        'Reproduce: python predict.py --predict --data /data --out /output/submission.csv\n\n'
        'Prediction reads checkpoint.json, metadata.json, and test queries/history only. It uses no forecast lookup table and no training/validation files. IDs only group histories and identify output rows. Missing ticks are unknown; future ticks are rejected.\n\n'
        'Training discovers win and payout-size lag patterns from training histories. Population odds and the residual readout are fitted on all released validation targets after model selection. The raw validation fit score is optimistic. Use the episode-grouped CV score in training_report.json for comparison. That CV was used in selection and is not an untouched final test.\n\n'
        'Whole episodes stay together in each CV fold. Centers, positive-payout templates, amount-pattern priors, and the residual readout are refitted outside each fold. No private test answers are used.\n\n'
        'To retrain from the participant kit, use the separate source bundle and: python house_edge_next.py --data public --out new-run --cv\n',encoding='utf-8')
    archive=package_code(source,out/'entry.zip')
    if archive.stat().st_size>100_000_000 or (out/'submission.csv').stat().st_size>10_000_000:raise ValidationError('Upload size limit exceeded')
    print('Verifying the actual ZIP with only test inputs available...',flush=True)
    with tempfile.TemporaryDirectory() as temporary:
        tmp=Path(temporary);unpacked=tmp/'code';extract_archive(archive.read_bytes(),unpacked)
        isolated=tmp/'data';(isolated/'test').mkdir(parents=True)
        shutil.copyfile(data/'metadata.json',isolated/'metadata.json')
        for name in ('queries.csv','history.csv'):shutil.copyfile(data/'test'/name,isolated/'test'/name)
        reproduced=tmp/'submission.csv'
        run=subprocess.run([sys.executable,'predict.py','--predict','--data',str(isolated),'--out',str(reproduced)],cwd=unpacked,capture_output=True,text=True,timeout=1800,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        if run.returncode:raise ValidationError(f'ZIP reproduction failed: {run.stdout}\n{run.stderr}')
        actual=read_submission(reproduced,{x.query_id for x in test_q});expected=read_submission(out/'submission.csv',{x.query_id for x in test_q})
        difference=max(abs(a-b) for k in expected for a,b in zip(expected[k],actual[k]))
        if difference>1e-8:raise ValidationError(f'ZIP mismatch: {difference}')
    report['verification']={'passed':True,'test_queries':len(test_q),'max_probability_difference':difference,'input_access':'Only metadata and test queries/history; no fitting data or saved forecast input'}
    report['seconds']=time.perf_counter()-started;report['entry_sha256']=digest(archive)
    write_json(out/'results.json',report)
    print(f'ZIP reproduction PASSED. Max probability difference: {difference:.3g}',flush=True)
    print(f"Runtime: {report['seconds']:.1f} seconds\nForecast: {out/'submission.csv'}\nMatching ZIP: {archive}")
    print('Nothing was uploaded.')


def main():
    root=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser(description='Train or reproduce House Edge next forecasts')
    ap.add_argument('--data',type=Path,default=root/'public');ap.add_argument('--out',type=Path,default=Path('strong-03'))
    ap.add_argument('--cv',action='store_true',help='Run four-fold episode-grouped validation before final fitting')
    ap.add_argument('--predict',action='store_true');ap.add_argument('--team-id',default='Select the registered team on the submission form')
    args=ap.parse_args()
    try:
        if args.predict:
            model=json.loads((root/'checkpoint.json').read_text(encoding='utf-8'));metadata=json.loads((args.data/'metadata.json').read_text(encoding='utf-8'))
            if any(metadata.get(k)!=model[k] for k in ('round','season_id')):raise ValidationError('Checkpoint belongs to another round/season; retrain on that kit')
            q,h=load_inputs(args.data/'test');write_submission(args.out,mapping(q,predict(q,h,model)))
        else:train(args)
    except (ValueError,OSError,subprocess.SubprocessError) as error:raise SystemExit(f'ERROR: {error}')
