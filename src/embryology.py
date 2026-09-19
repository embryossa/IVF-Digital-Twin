# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""Shared L1 parameters, transfer scenario and exact conditioning of the stage chain.

Provenance of the 2026-09-11 constants (scripts and fit reports in
IVF_Clinic_Development_7.1/work/layer-flow-fix-2026-09-11):
* stage 1 — zero-truncated NB with log link, Fertimed stimulated cycles
  2025-07.2026 (n=687 with OCC>=1, 5-fold patient CV). Blank OCC cells are
  explained by NB zeros plus a 5.8% unrecorded rate, so no separate structural
  zero component is identifiable; BMI is not recorded and has no term;
* stage 4 — zero-inflated beta-binomial on culture-observed protocols_15k
  cycles, excluding day-5/6/7 transfers recorded with Bl=0 (owner decision:
  partly unfilled blastocyst fields), n=7919;
* fair-quality transfer odds ratio — day-5 single blastocyst transfers,
  adjusted for age and ln(2PN), n=2975 (399 fair-quality only);
* follicle count, day-5 embryos and frozen embryos — the training-data
  definitions of the KAT/GAT/CSDI inputs that the simulation does not observe.
"""
import numpy as np
from scipy.stats import binom,betabinom,nbinom

STAGES=('okk','mii','pn2','blasts','good','euploid')

# Stage 1: mu = exp(c0 + c_amh*ln(AMH+0.1) + c_afc*ln(AFC+1) + c_age*(age-35)).
OCC_C0, OCC_C_AMH, OCC_C_AFC, OCC_C_AGE, OCC_THETA = 0.059760569570942425, 0.26842852018697644, 0.768917489602832, -0.003929881848384461, 5.772194027999357

# Stage 4 (ZIBB) and stage 5 (BB).
BLAST_B0, BLAST_B_AGE, BLAST_B_N, BLAST_N_CAP = 0.6143088201494307, -0.016956173666476206, 0.019593039761272916, 5.0
BLAST_KAPPA, BLAST_PI_ZERO = 7.823023379292471, 0.009542306619502793
GOOD_C0, GOOD_C_AGE, GOOD_C_N, GOOD_KAPPA = 0.0882450191913956, -0.06351627768526279, 0.3572181541653332, 4.1309470589977195

WARMING_SURVIVAL = 0.95
# Adjusted odds ratio of pregnancy for a fair- versus good-quality blastocyst transfer (95% CI 0.35-0.75).
FAIR_QUALITY_LOG_OR = -0.6664093102660372
# protocols_15k: median OCC / punctured follicles.
OKK_PER_FOLLICLE = 0.8461538461538461


def blast_mean(age,n):
    n=np.maximum(np.asarray(n,dtype=float),1.)
    return 1/(1+np.exp(-(BLAST_B0+BLAST_B_AGE*max(0.,age-35)+BLAST_B_N*np.log(np.minimum(n,BLAST_N_CAP)))))


def good_mean(age,n):
    n=np.maximum(np.asarray(n,dtype=float),1.)
    return 1/(1+np.exp(-(GOOD_C0+GOOD_C_AGE*max(0.,age-35)+GOOD_C_N*np.log(n))))


def oocyte_mean(age,amh,afc):
    return float(np.exp(OCC_C0+OCC_C_AMH*np.log(amh+.1)+OCC_C_AFC*np.log(afc+1.)+OCC_C_AGE*(age-35.)))


def parameters(patient):
    age,amh,afc=patient.female_age,patient.amh,patient.afc
    sigmoid=lambda x:1/(1+np.exp(-x))
    euploid=next(p for upper,p in ((30,.70),(35,.65),(38,.55),(40,.35),(42,.18),(float('inf'),.10)) if age<upper)
    return dict(mu=oocyte_mean(age,amh,afc),theta=OCC_THETA,zero=0.,
        maturity=float(sigmoid(2.4665+.005*age-.782+.24*amh-.069)),
        fertilisation=float(sigmoid(1.1678+.004*age-.303-.051)),
        euploid=euploid,euploid_k=6.)


def impute_follicles(okk):
    """Punctured follicles implied by a retrieved-oocyte count (training definition)."""
    return np.rint(np.asarray(okk,dtype=float)/OKK_PER_FOLLICLE)


def follicle_kpi_score(age,follicles,mii,fert_rate,good):
    """KPIScore exactly as recorded in the KAT/GAT/CSDI training data (follicle-based)."""
    follicles=np.asarray(follicles,dtype=float);mii=np.asarray(mii,dtype=float)
    fert_rate=np.asarray(fert_rate,dtype=float);good=np.asarray(good,dtype=float)
    a=1 if age>=40 else (5 if age<=36 else 3)
    b=np.where(follicles>15,5,np.where(follicles>=8,3,1))
    c=np.where(mii<=3,1,np.where(mii<=7,3,5))
    d=np.where(fert_rate<.5,1,np.where(fert_rate<=.65,3,5))
    e=np.where(good==0,1,np.where(good<=2,3,5))
    return (a+b+c+d+e).astype(int)


def transfer_candidates(blasts,good,euploid,pgt):
    """Embryos that survive warming, split into good- and fair-quality transfers.

    Without PGT-A every blastocyst is transferable (good quality first). With
    PGT-A only euploid embryos are transferable.
    """
    good=np.asarray(good);blasts=np.asarray(blasts);euploid=np.asarray(euploid)
    if pgt:
        return np.random.binomial(euploid,WARMING_SURVIVAL),np.zeros(len(euploid),dtype=int)
    return np.random.binomial(good,WARMING_SURVIVAL),np.random.binomial(blasts-good,WARMING_SURVIVAL)


def fair_probability(p):
    p=np.clip(np.asarray(p,dtype=float),1e-9,1-1e-9)
    return 1/(1+np.exp(-(np.log(p/(1-p))+FAIR_QUALITY_LOG_OR)))


# Shared logit-normal random effect across the transfers of one cycle. Fitted on
# GGRC retrievals that record the outcome of every transfer (1703 retrievals,
# 2406 transfers, 461 with >=2): sigma=0.746, 95% profile CI 0.29-1.13, latent
# ICC 0.145. Adjusting for age and blastocyst count leaves it at 0.732, so this
# is residual correlation, not heterogeneity the covariates already explain; the
# single-embryo-transfer subset gives 0.681. A structural never-implants share
# was not identifiable (pi=0, 95% CI 0-0.05) and is therefore not modelled.
# Script, report and limitations: IVF_Clinic_Development_7.1/work/frailty-2026-09-12.
TRANSFER_FRAILTY_SIGMA=0.746
_FRAILTY_NODES,_FRAILTY_WEIGHTS=np.polynomial.hermite_e.hermegauss(32)
_FRAILTY_WEIGHTS=_FRAILTY_WEIGHTS/np.sqrt(2*np.pi)
_FRAILTY_GRID={}


def _frailty_location(p,sigma):
    """Logit location whose mean over the shared effect equals p."""
    if sigma not in _FRAILTY_GRID:
        grid=np.linspace(-20.,20.,4001)
        mean=(_FRAILTY_WEIGHTS*1/(1+np.exp(-(grid[:,None]+sigma*_FRAILTY_NODES[None,:])))).sum(axis=1)
        _FRAILTY_GRID[sigma]=(mean,grid)
    mean,grid=_FRAILTY_GRID[sigma]
    return np.interp(np.clip(p,1e-9,1-1e-9),mean,grid)


def frailty_transfer_probability(p,z,sigma=None):
    """Per-transfer probability for one draw z of the cycle's shared effect.

    Averaged over z it returns p, so sampled transfer outcomes stay consistent
    with cycle_probabilities while sharing one effect within a cycle.
    """
    sigma=TRANSFER_FRAILTY_SIGMA if sigma is None else float(sigma)
    if sigma==0:return np.asarray(p,dtype=float)
    return 1/(1+np.exp(-(_frailty_location(p,sigma)+sigma*np.asarray(z,dtype=float))))


def cycle_probabilities(p,n_good,n_fair,sigma=None):
    """P(>=1 pregnancy) across the transferable embryos of each scenario.

    p is the per-transfer probability of the scenario's first (best) embryo.
    Fair-quality embryos transferred after a good-quality one carry the
    fair-quality odds ratio; in fair-only scenarios p already describes them.

    Transfers of one cycle share a random effect, so successive transfers are
    positively correlated and add less than an independent chain would. Every
    per-transfer probability keeps its marginal value: a single embryo returns p
    exactly, and sigma=0 restores the independent chain.
    """
    p=np.asarray(p,dtype=float);n_good=np.asarray(n_good,dtype=float);n_fair=np.asarray(n_fair,dtype=float)
    if not (p.shape==n_good.shape==n_fair.shape):raise ValueError('Invalid transfer arrays')
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid transfer probabilities')
    for n in (n_good,n_fair):
        if not np.isfinite(n).all() or (n<0).any() or (n!=np.floor(n)).any():raise ValueError('Invalid transfer counts')
    sigma=TRANSFER_FRAILTY_SIGMA if sigma is None else float(sigma)
    if not np.isfinite(sigma) or sigma<0:raise ValueError('Invalid transfer frailty')
    p_fair=np.where(n_good>=1,fair_probability(p),p)
    if sigma==0:
        return np.where(n_good+n_fair==0,0.,1-np.power(1-p,n_good)*np.power(1-p_fair,n_fair))
    shape=p.shape;flat=lambda v:np.asarray(v,dtype=float).reshape(-1)
    effect=sigma*_FRAILTY_NODES[None,:]
    good=1/(1+np.exp(-(_frailty_location(flat(p),sigma)[:,None]+effect)))
    fair=1/(1+np.exp(-(_frailty_location(flat(p_fair),sigma)[:,None]+effect)))
    shared=(_FRAILTY_WEIGHTS*(1-np.power(1-good,flat(n_good)[:,None])*np.power(1-fair,flat(n_fair)[:,None]))).sum(axis=1)
    # The quadrature is exact only to its node count, so the two identities the rest
    # of the pipeline relies on are taken directly: one transfer is its own marginal,
    # and an impossible transfer stays impossible.
    return np.where((n_good+n_fair==0)|(p<=0),0.,np.where(n_good+n_fair==1,p,shared.reshape(shape)))


def anchored_cycle_probability(scenario_p,n_good,n_fair,per_transfer):
    """Cycle probability consistent with a final per-transfer probability.

    Scenario probabilities are shifted by one logit constant so that their
    mean over scenarios with a transfer equals per_transfer; heterogeneity
    between scenarios is preserved.
    """
    from scipy.optimize import brentq
    p=np.clip(np.asarray(scenario_p,dtype=float),1e-9,1-1e-9);n_good=np.asarray(n_good);n_fair=np.asarray(n_fair)
    mask=(n_good+n_fair)>=1
    if per_transfer is None or not mask.any():return 0. if not mask.any() else None
    target=float(np.clip(per_transfer,1e-9,1-1e-9));lp=np.log(p/(1-p))
    shifted=lambda delta:1/(1+np.exp(-(lp+delta)))
    delta=brentq(lambda v:shifted(v)[mask].mean()-target,-30,30)
    return float(cycle_probabilities(shifted(delta),n_good,n_fair).mean())


def sample_conditioned_counts(patient,known,n):
    """Forward filtering/backward sampling under all observed count evidence.

    Replacing an observed child alone is not conditioning: its ancestors must
    be reweighted too. This finite-state calculation retains all feasible
    trajectories and does not clip/invent unknown upstream counts.
    """
    observed=[getattr(known,k) for k in STAGES]
    last=None
    for value in observed:
        if value is None:continue
        if isinstance(value,bool) or not np.isfinite(value) or value!=int(value) or not 0<=value<=100:
            raise ValueError('Invalid observed embryo count')
        if last is not None and value>last:raise ValueError('Inconsistent observed stage counts')
        last=value
    # Preserve the original prior support; accommodate explicitly observed
    # high counts without discarding the NB tail above the usual cap of 50.
    cap=100 if max(v or 0 for v in observed)>50 else 50
    states=np.arange(cap+1);p=parameters(patient)
    prior=(1-p['zero'])*nbinom.pmf(states,p['theta'],p['theta']/(p['theta']+p['mu']))
    prior[0]+=p['zero']
    prior[-1]+=(1-p['zero'])*nbinom.sf(cap,p['theta'],p['theta']/(p['theta']+p['mu']))
    if observed[0] is not None:
        prior[:]=0;prior[int(observed[0])]=1
    n_parent=states[:,None];children=states[None,:]
    transitions=[binom.pmf(children,n_parent,p[k]) for k in ('maturity','fertilisation')]
    m4=blast_mean(patient.female_age,states)[:,None];m5=good_mean(patient.female_age,states)[:,None]
    t4=(1-BLAST_PI_ZERO)*betabinom.pmf(children,n_parent,m4*BLAST_KAPPA,(1-m4)*BLAST_KAPPA)+BLAST_PI_ZERO*(children==0)
    transitions += [t4,betabinom.pmf(children,n_parent,m5*GOOD_KAPPA,(1-m5)*GOOD_KAPPA),betabinom.pmf(children,n_parent,p['euploid']*p['euploid_k'],(1-p['euploid'])*p['euploid_k'])]
    filtered=[prior/prior.sum()]
    for j,t in enumerate(transitions,1):
        f=filtered[-1]@t
        if observed[j] is not None:f=np.where(states==observed[j],f,0.)
        total=f.sum()
        if not np.isfinite(total) or total<=0:raise ValueError('Observations outside model support')
        filtered.append(f/total)
    draws=np.empty((6,n),dtype=int)
    draws[-1]=np.random.choice(states,size=n,p=filtered[-1])
    for j in range(4,-1,-1):
        for child in np.unique(draws[j+1]):
            mask=draws[j+1]==child
            weights=filtered[j]*transitions[j][:,child]
            weights/=weights.sum()
            draws[j,mask]=np.random.choice(states,size=int(mask.sum()),p=weights)
    return tuple(draws)
