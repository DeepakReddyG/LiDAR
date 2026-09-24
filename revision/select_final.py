"""Select once from all registered development outcomes under objective.md."""
import json
import subprocess
from pathlib import Path
import numpy as np
from revision.runner import ROOT, guard_for, sha


def choose(candidates):
    if not candidates:
        raise ValueError('No completed candidates')
    if any(not m['integrity']['geometry_checked'] or m['integrity']['missing_reviewed_points'] or m['primary_loss'] is None for _,m in candidates):
        raise ValueError('Candidate integrity or reviewed evidence is incomplete')
    best=min(m['primary_loss'] for _,m in candidates)
    tied=[(name,m) for name,m in candidates if abs(m['primary_loss']-best)<=1e-12]
    def complexity(item):
        name,m=item;cfg=m['configuration']
        learned=int(cfg.get('source')=='sam3')
        stages=1 if 'pointwise' in cfg else 3+int(cfg['constraints'])+int(cfg['smoothing'])+int(cfg['transfer']=='height_aware')
        return learned,stages,m['seconds_downstream'],name
    return min(tied,key=complexity)


def main():
    destination=ROOT/'revision_work/manifests/final_manifest.json'
    if destination.exists():raise FileExistsError('Final selection must not be overwritten')
    candidates=[];locations={};parent_paths={}
    for name in ['fixed_anchor','grid_025','grid_1','rgb_band_075','rgb_band_3']:
        subdir='analysis_full' if name=='fixed_anchor' else 'analysis'
        path=ROOT/f'revision_work/runs/{name}/{subdir}/metrics.json'
        parent=ROOT/f'revision_work/manifests/{name}.json'
        manifest=json.loads(parent.read_text());guard_for(manifest).verify_locked_files()
        result=json.loads(path.read_text())
        for method,m in result['methods'].items():
            key=name+'/'+method;candidates.append((key,m));locations[key]=path;parent_paths[key]=parent
    name,m=choose(candidates);parent=parent_paths[name];final=json.loads(parent.read_text())
    exp,method=name.split('/',1);subdir=locations[name].parent
    with np.load(subdir/(method+'_predictions.npz')) as prediction, np.load(ROOT/'revision_work/runs/fixed_anchor/analysis/pilot_arrays.npz') as anchor:
        if not np.array_equal(prediction['ids'],anchor['ids']):raise ValueError('Selected result is not the identical pilot population')
    final.update(phase=4,selected_configuration=m['configuration'],selected_candidate=name,selected_primary_loss=m['primary_loss'],selection_objective='objective.md; wrong and unlabelled cost 1:1',development_only=True,selection_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),selected_parent_manifest=str(parent.relative_to(ROOT)),selected_parent_manifest_sha256=sha(parent),selected_prediction_artifact=str((subdir/(method+'_predictions.npz')).relative_to(ROOT)),selected_prediction_sha256=sha(subdir/(method+'_predictions.npz')),approved_reference_records=[],phase5_instruction='After committed freeze, consume the gate once. No approved independent reference records are known. Inspect locked candidates, create blind annotation tasks and stop without prediction or scoring if approval absent.',selection_candidates=[{'id':k,'primary_loss':v['primary_loss'],'errors':v['error_points'],'metric_artifact':str(locations[k].relative_to(ROOT)),'metric_sha256':sha(locations[k])} for k,v in candidates])
    if m['configuration'].get('pointwise')=='geometry':
        final['selected_required_stages']=['bounded grid','minimum-elevation decimation','CSF terrain and HAG','pointwise geometry-only vegetation proxy','LAS semantic crosswalk']
        final['unused_in_selected_method']=['RGB/ExG classification','orthographic RGB','SAM3','mask constraints','pixel smoothing','image-to-point transfer']
        final['scope_warning']='Selected by pilot objective only. Restricted grass/tree/abstain proxy; no positive prediction capability for buildings, pavement or vehicles. The height-separated and geometrically grouped pilot favors this proxy. It is not a validated multiclass replacement or deployable asset inventory.'
    else:final['scope_warning']='Pilot-selected configuration, not independently validated.'
    final['phase5'] = {'gate_path':'revision_work/phase5_gate.json','annotation_output':'research_paper_documents_and_drafts/supporting/revision_experiments_2026-09-23/holdout_annotation','approved_reference_records':[]}
    final['source_hashes']['revision/select_final.py'] = sha(Path(__file__))
    final['source_hashes']['revision/final_gate.py'] = sha(ROOT/'revision/final_gate.py')
    final['source_hashes']['revision/annotation_package.py'] = sha(ROOT/'revision/annotation_package.py')
    destination.write_text(json.dumps(final,indent=2)+'\n')
    print(json.dumps({'selected':name,'errors':m['error_points'],'balanced_error_percent':100*m['primary_loss'],'candidate_count':len(candidates)},indent=2))

if __name__=='__main__':main()
