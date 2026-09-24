from revision.select_final import choose
import pytest

def candidate(loss,**cfg):
    return {'primary_loss':loss,'integrity':{'geometry_checked':True,'missing_reviewed_points':0},'configuration':cfg,'seconds_downstream':1.}

def test_primary_loss_precedes_complexity():
    assert choose([('simple',candidate(.02,pointwise='geometry')),('complex',candidate(.01,source='sam3',constraints=True,smoothing=True,transfer='height_aware'))])[0]=='complex'

def test_exact_tie_prefers_no_learned_model_and_fewer_stages():
    assert choose([('sam',candidate(.01,source='sam3',constraints=False,smoothing=False,transfer='naive')),('rules',candidate(.01,pointwise='geometry'))])[0]=='rules'

def test_incomplete_candidate_fails_instead_of_being_silently_removed():
    bad=candidate(.001,pointwise='geometry');bad['integrity']['geometry_checked']=False
    with pytest.raises(ValueError,match='integrity'):choose([('bad',bad),('good',candidate(.1,pointwise='rules'))])
