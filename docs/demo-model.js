/* Reduced Digital Twin 7.1 for the demo: L1 cascade, L2 FORTUNE+KPI, L3 = closed-form KAN branch of KAT, L7 = BEFE with L1-L3 only.
   Ported from IVF_Clinic_Desktop 7.1 (src/embryology.py, src/ivf_digital_twin.py, src/ivf_core.transfer_view, befe.py, befe_app.py).
   Not ported: FT-Transformer and isotonic calibration of KAT, CSDI (L5), GAT (L6), cluster (L4), OOD, reliability, centre calibration. */
(function(root) {
  'use strict';
  const presets = {
    balanced:{name:'Сбалансированный пример',age:35,amh:2.5,afc:15,facts:[12,10,7,4,3]},
    lower:{name:'Меньший овариальный резерв',age:39,amh:1,afc:8,facts:[6,4,3,1,1]},
    higher:{name:'Больший овариальный резерв',age:30,amh:4.5,afc:23,facts:[20,16,12,7,5]}
  };
  const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
  const limits={age:[25,43,1],amh:[.5,5,.1],afc:[5,25,1],stage:[0,4,1],transfers:[1,3,1]};
  const BMI=22.04,ATTEMPT=1,N_SIM=4000,SEED=42,CAP=50;
  function validate(s) {
    if(!s || !Object.hasOwn(presets,s.preset))throw new Error('Выберите один из трёх учебных примеров.');
    for(const [k,[a,b,step]] of Object.entries(limits)) {
      if(typeof s[k]!=='number'||!Number.isFinite(s[k])||s[k]<a||s[k]>b||Math.abs((s[k]-a)/step-Math.round((s[k]-a)/step))>1e-6)throw new Error('Недопустимое значение: '+k);
    }
    return s;
  }
  function initial(preset='balanced') {const p=presets[preset];if(!p)throw new Error('Неизвестный пример');return {preset,age:p.age,amh:p.amh,afc:p.afc,stage:0,transfers:2};}

  // ── numerics ──────────────────────────────────────────────────────────
  const sig=x=>1/(1+Math.exp(-x)),logit=p=>Math.log(p/(1-p)),silu=x=>x/(1+Math.exp(-x));
  const LANCZOS=[0.99999999999980993,676.5203681218851,-1259.1392167224028,771.32342877765313,-176.61502916214059,12.507343278686905,-0.13857109526572012,9.9843695780195716e-6,1.5056327351493116e-7];
  function lgamma(x){if(x<.5)return Math.log(Math.PI/Math.abs(Math.sin(Math.PI*x)))-lgamma(1-x);x-=1;let a=LANCZOS[0];const t=x+7.5;for(let i=1;i<9;i++)a+=LANCZOS[i]/(x+i);return .5*Math.log(2*Math.PI)+(x+.5)*Math.log(t)-t+Math.log(a);}
  const lchoose=(n,k)=>lgamma(n+1)-lgamma(k+1)-lgamma(n-k+1),lbeta=(a,b)=>lgamma(a)+lgamma(b)-lgamma(a+b);
  function rng(seed){let a=seed>>>0;return ()=>{a=(a+0x6D2B79F5)>>>0;let t=a;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};}
  function normal(u){let a=u();while(a<=0)a=u();return Math.sqrt(-2*Math.log(a))*Math.cos(2*Math.PI*u());}
  function gamma(k,u){if(k<1)return gamma(k+1,u)*Math.pow(u()||1e-300,1/k);const d=k-1/3,c=1/Math.sqrt(9*d);for(;;){let x,v;do{x=normal(u);v=1+c*x;}while(v<=0);v=v*v*v;const w=u();if(w<1-.0331*x*x*x*x||Math.log(w)<.5*x*x+d*(1-v+Math.log(v)))return d*v;}}
  const betaDraw=(a,b,u)=>{const x=gamma(a,u);return x/(x+gamma(b,u));};
  // Inverse-CDF draw from a pmf row; rows are exact model laws, so one uniform per draw.
  function draw(cdf,u){const r=u()*cdf[cdf.length-1];let lo=0,hi=cdf.length-1;while(lo<hi){const m=(lo+hi)>>1;if(cdf[m]<r)lo=m+1;else hi=m;}return lo;}
  const cumulate=pmf=>{const c=new Float64Array(pmf.length);let s=0;pmf.forEach((v,i)=>c[i]=s+=v);return c;};
  const binomRow=(n,p)=>cumulate(Array.from({length:n+1},(_,k)=>p<=0?Number(k===0):p>=1?Number(k===n):Math.exp(lchoose(n,k)+k*Math.log(p)+(n-k)*Math.log(1-p))));
  const bbRow=(n,a,b)=>cumulate(Array.from({length:n+1},(_,k)=>Math.exp(lchoose(n,k)+lbeta(k+a,n-k+b)-lbeta(a,b))));
  function quantile(sorted,q){if(!sorted.length)return null;const h=(sorted.length-1)*q,i=Math.floor(h);return i+1<sorted.length?sorted[i]+(h-i)*(sorted[i+1]-sorted[i]):sorted[i];}
  const sorted=a=>Float64Array.from(a).sort(),mean=a=>{let s=0;for(const v of a)s+=v;return a.length?s/a.length:NaN;};
  const variance=a=>{const m=mean(a);let s=0;for(const v of a)s+=(v-m)*(v-m);return s/a.length;};
  const rint=x=>{const f=Math.floor(x),d=x-f;return d>.5?f+1:d<.5?f:(f%2===0?f:f+1);};   // numpy rint: half to even

  // ── L1: stage laws (src/embryology.py, fit of 2026-09-11) ─────────────
  const OCC=[0.059760569570942425,0.26842852018697644,0.768917489602832,-0.003929881848384461,5.772194027999357];
  const BLAST=[0.6143088201494307,-0.016956173666476206,0.019593039761272916,5.0,7.823023379292471,0.009542306619502793];
  const GOOD=[0.0882450191913956,-0.06351627768526279,0.3572181541653332,4.1309470589977195];
  const WARMING_SURVIVAL=.95,FAIR_QUALITY_LOG_OR=-0.6664093102660372,OKK_PER_FOLLICLE=0.8461538461538461;
  const blastMean=(age,n)=>sig(BLAST[0]+BLAST[1]*Math.max(0,age-35)+BLAST[2]*Math.log(Math.min(Math.max(n,1),BLAST[3])));
  const goodMean=(age,n)=>sig(GOOD[0]+GOOD[1]*Math.max(0,age-35)+GOOD[2]*Math.log(Math.max(n,1)));
  function parameters(age,amh,afc){
    const euploid=[[30,.70],[35,.65],[38,.55],[40,.35],[42,.18],[Infinity,.10]].find(([upper])=>age<upper)[1];
    return {mu:Math.exp(OCC[0]+OCC[1]*Math.log(amh+.1)+OCC[2]*Math.log(afc+1)+OCC[3]*(age-35)),theta:OCC[4],
      maturity:sig(2.4665+.005*age-.782+.24*amh-.069),fertilisation:sig(1.1678+.004*age-.303-.051),euploid};
  }
  // NB oocyte law over 0..50; the tail above 50 is clipped to 50 as in stage1_oocytes.
  function oocyteRow(mu,theta){const p=theta/(theta+mu),pmf=[];let s=0;for(let k=0;k<CAP;k++){const v=Math.exp(lgamma(k+theta)-lgamma(theta)-lgamma(k+1)+theta*Math.log(p)+k*Math.log(1-p));pmf.push(v);s+=v;}pmf.push(Math.max(0,1-s));return cumulate(pmf);}

  // ── L2: FORTUNE + KPIScore (stage7_pregnancy_cycle) ───────────────────
  const KPI_CI={5:[.012,.08],6:[.016,.09],7:[.02,.10],8:[.03,.12],9:[.04,.14],10:[.05,.15],11:[.06,.18],12:[.08,.20],13:[.10,.22],14:[.13,.26],15:[.16,.30],16:[.20,.32],17:[.24,.36],18:[.28,.41],19:[.33,.46],20:[.38,.51],21:[.42,.57],22:[.47,.63],23:[.51,.69],24:[.56,.74],25:[.59,.79]};
  const KPI_BETA=Object.fromEntries(Object.entries(KPI_CI).map(([s,[lo,hi]])=>{const m=(lo+hi)/2,v=((hi-lo)/3.92)**2,k=m*(1-m)/v-1;return [s,[m*k,(1-m)*k]];}));
  const KPI_WEIGHT=.5;
  const fortuneLogit=(age,amh)=>.40-.55*((age-36.3)/5.5)+.15*((Math.log(amh+.1)-Math.log(2.1))/1.2)-.20*((BMI-24)/4.2);
  function kpiScore(age,amh,mii,fertPct,good){return (age>=40?1:age<=36?5:3)+(amh<1?1:amh<2?3:5)+(mii<=3?1:mii<7?3:5)+(fertPct<50?1:fertPct<=65?3:5)+(good===0?1:good<=2?3:5);}

  // ── L3: KAN branch of KAT as a closed formula ──────────────────────────
  // Prediction_KAN.pth (SHA-256 c7178818…e6e4, identical to the shipped 7.1 artifact). Its B-spline
  // terms are ~0 (RMS 4.0e-5 / 1.6e-4), so the network IS this formula: max |ΔP| vs torch 1.5e-4.
  //   h = W·SiLU(x);  h̃ = γ·(h−mean h)/sqrt(var h+ε)+β;  logit = v·SiLU(h̃);  P = σ(logit)
  // Raw inputs (the KAN was trained on unscaled features). In 7.1 the KAN is only one branch:
  // P_KAT = isotonic(0.279·σ(KAN) + 0.721·P_FTT); FT-Transformer and the calibrator are not in the demo.
  const KAN={
    features:['Возраст','№ попытки','Количество фолликулов','Число ОКК','Число инсеминированных','2 pN','Число дробящихся на 3 день','Число Bl','Число Bl хор.кач-ва','Частота оплодотворения','Частота дробления','Частота формирования бластоцист','Частота формирования бластоцист хорошего качества','Частота получения ОКК','Число эмбрионов 5 дня','Заморожено эмбрионов','Перенесено эмбрионов','KPIScore'],
    W:[[0.134840757,0.188098356,0.10291402,-0.0107433069,0.103713557,0.142426595,-0.0348568112,0.0613383912,-0.198162228,0.139162913,0.0195489004,-0.0957883224,-0.22045517,0.1526981,-0.145776093,0.224987224,-0.0998843461,-0.0484573506],[0.0470892191,0.0340457931,0.16739355,-0.108873859,0.12146917,0.15910925,-0.149149016,0.279411644,0.0478589013,-0.15881516,0.0850369707,-0.0008999118,-0.138217211,-0.12843734,-0.0896401256,0.167511106,-0.234311536,-0.0652154386],[0.21122101,-0.0051486208,0.238954008,-0.195571646,-0.0042325063,0.19336763,-0.0064448328,-0.0140114548,0.018089259,0.282440871,0.0253611412,0.0760162547,0.0975760818,0.0203329604,-0.126765296,-0.157883882,-0.142220303,-0.0925993025],[0.198384538,-0.0071096052,0.186491564,0.05414249,0.127100453,0.0102990549,-0.211242408,-0.0005139245,-0.0718772933,-0.177675396,0.139575422,-0.125408933,0.0228636283,-0.0012914281,0.157713682,-0.142298892,0.204482704,0.0763295442],[0.0365882814,-0.0263622608,-0.129863501,-0.277408719,0.136338681,0.158178777,-0.255265236,0.0355494134,0.260693699,0.0083318092,-0.28281334,-0.0447003283,-0.0036777372,0.0624441803,-0.112083033,-0.231154561,0.0766190886,-0.0112362811],[0.0911054835,0.231715754,0.191292673,0.22840181,-0.158452138,0.169078112,-0.211054757,-0.297052145,-0.0416553468,-0.0872100741,-0.0161488857,-0.204010397,-0.0114666764,0.285029501,-0.0674427822,-0.203679651,-0.173104242,-0.015823178],[-0.106751263,0.105304249,0.28648746,0.209207028,0.157671958,0.127813846,0.0191233028,0.0316776149,0.033552818,-0.141029835,0.305530191,0.164710134,-0.0779938027,0.0681657717,0.318559259,0.296360016,0.0969015062,0.223687738],[-0.0633886904,0.0958723947,-0.0206636749,0.244725719,0.252257407,-0.0907760412,-0.0817539766,0.0273637436,-0.253011078,0.220347121,0.040002685,-0.0365979522,-0.30176118,0.0732349381,-0.0101127056,0.192297995,-0.0552874021,0.0877462476],[-0.032653667,0.142587408,-0.169176951,-0.196526513,0.212374628,0.231265575,0.236382946,0.0955050737,0.11723233,-0.151553318,0.0234880652,0.144869998,-0.155972213,0.198123157,0.0007465406,0.284249335,0.105015196,0.0968293697],[-0.112072155,0.229855984,0.0327953435,0.14901422,-0.128644258,-0.0899955854,-0.0227284785,0.0832818523,0.175856501,0.231938153,-0.0991462022,-0.136162564,-0.0389190204,-0.0702478811,-0.0997770354,0.131658003,-0.155511901,-0.092839554]],
    v:[-0.271954685,0.233985901,-0.304859012,-0.254699141,0.0028735627,-0.243485242,-0.0099084349,-0.209559515,-0.16277352,-0.240196422],
    gamma:[1.10346746,0.965121448,1.09830976,1.07539773,0.974614382,1.11004305,0.928918541,0.924571037,0.925150096,1.06271923],
    beta:[0.0802276954,-0.0516693741,0.079028666,0.0703720525,-0.0344525315,0.0877126828,0.0425539538,0.0423660614,0.0690707862,-0.0625727028],
    eps:1e-5
  };
  function kanTerms(x){
    const s=x.map(silu),h=KAN.W.map(row=>row.reduce((a,w,i)=>a+w*s[i],0)),m=mean(h),sd=Math.sqrt(variance(h)+KAN.eps);
    const terms=h.map((v,j)=>KAN.v[j]*silu((v-m)/sd*KAN.gamma[j]+KAN.beta[j]));
    return {terms,logit:terms.reduce((a,b)=>a+b,0)};
  }
  // build_nn_features: unobserved training columns use their protocols_15k definitions (single blastocyst transfer).
  function kanFeatures(age,okk,mii,pn2,blasts,good){
    const foll=rint(okk/OKK_PER_FOLLICLE),fert=mii>0?pn2/Math.max(mii,1):0;
    const kpi=(age>=40?1:age<=36?5:3)+(foll>15?5:foll>=8?3:1)+(mii<=3?1:mii<=7?3:5)+(fert<.5?1:fert<=.65?3:5)+(good===0?1:good<=2?3:5);
    return [age,ATTEMPT,foll,okk,mii,pn2,pn2,blasts,good,fert,pn2>0?1:0,pn2>0?blasts/Math.max(pn2,1):0,pn2>0?good/Math.max(pn2,1):0,okk/Math.max(foll,1),rint((blasts+pn2)/2),Math.max(good-1,0),1,kpi];
  }

  // ── transfers: fair-quality odds ratio and shared frailty (σ=0.746) ─────
  const FRAILTY_SIGMA=0.746;
  const HN=[-10.0774227,-9.06439921,-8.21972877,-7.46075575,-6.75593083,-6.08896431,-5.45003327,-4.83260461,-4.23202111,-3.64478125,-3.06813517,-2.49984042,-1.93800491,-1.3809802,-0.827284904,-0.275546419,0.275546419,0.827284904,1.3809802,1.93800491,2.49984042,3.06813517,3.64478125,4.23202111,4.83260461,5.45003327,6.08896431,6.75593083,7.46075575,8.21972877,9.06439921,10.0774227];
  const HW=[4.12460749e-23,5.20844959e-19,6.75529022e-16,2.37806486e-13,3.34750124e-11,2.31251841e-09,8.88129071e-08,2.0596221e-06,3.05598031e-05,0.000302557026,0.00206205105,0.0099034617,0.0341098477,0.0853448083,0.156538994,0.21170557,0.21170557,0.156538994,0.0853448083,0.0341098477,0.0099034617,0.00206205105,0.000302557026,3.05598031e-05,2.0596221e-06,8.88129071e-08,2.31251841e-09,3.34750124e-11,2.37806486e-13,6.75529022e-16,5.20844959e-19,4.12460749e-23];
  let frailtyGrid=null;
  function frailtyLocation(p){
    if(!frailtyGrid){const g=new Float64Array(4001),m=new Float64Array(4001);for(let i=0;i<4001;i++){g[i]=-20+i*.01;let s=0;for(let k=0;k<32;k++)s+=HW[k]/(1+Math.exp(-(g[i]+FRAILTY_SIGMA*HN[k])));m[i]=s;}frailtyGrid={g,m};}
    const {g,m}=frailtyGrid;p=clamp(p,1e-9,1-1e-9);if(p<=m[0])return g[0];if(p>=m[4000])return g[4000];
    let lo=0,hi=4000;while(hi-lo>1){const c=(lo+hi)>>1;if(m[c]<=p)lo=c;else hi=c;}return g[lo]+(p-m[lo])/(m[hi]-m[lo])*(g[hi]-g[lo]);
  }
  const fairProbability=p=>sig(logit(clamp(p,1e-9,1-1e-9))+FAIR_QUALITY_LOG_OR);
  function cycleProbability(p,nGood,nFair){
    const n=nGood+nFair;if(n===0||p<=0)return 0;if(n===1)return p;
    const lg=frailtyLocation(p),lf=frailtyLocation(nGood>=1?fairProbability(p):p);let s=0;
    for(let k=0;k<32;k++){const e=FRAILTY_SIGMA*HN[k];s+=HW[k]*(1-Math.pow(1-sig(lg+e),nGood)*Math.pow(1-sig(lf+e),nFair));}
    return s;
  }

  // ── Monte Carlo cycle (run_pipeline with the demo's prefix observations) ─
  // The demo reveals stages in order, so every unknown stage lies downstream of the last
  // observed one; forward sampling is then the exact conditional law (no FFBS needed).
  function simulate(age,amh,afc,facts,cap,n){
    const u=rng(SEED),P=parameters(age,amh,afc),rows={},row=(key,k,make)=>(rows[key]||(rows[key]=[]))[k]||(rows[key][k]=make(k));
    const occRow=oocyteRow(P.mu,P.theta),fl=fortuneLogit(age,amh);
    const S={okk:new Int16Array(n),mii:new Int16Array(n),pn2:new Int16Array(n),blasts:new Int16Array(n),good:new Int16Array(n),euploid:new Int16Array(n),nGood:new Int16Array(n),nFair:new Int16Array(n),comb:new Float64Array(n),fortune:new Float64Array(n),kpi:new Float64Array(n),kpiScore:new Int8Array(n),kan:new Float64Array(n),kanLogit:new Float64Array(n)};
    const terms=new Float64Array(10),feat=new Float64Array(18);
    for(let i=0;i<n;i++){
      const okk=facts[0]??draw(occRow,u);
      const mii=facts[1]??draw(row('mii',okk,k=>binomRow(k,P.maturity)),u);
      const pn2=facts[2]??draw(row('pn2',mii,k=>binomRow(k,P.fertilisation)),u);
      let blasts=facts[3];if(blasts==null){const m=blastMean(age,pn2);blasts=u()<BLAST[5]?0:draw(row('bl',pn2,k=>bbRow(k,m*BLAST[4],(1-m)*BLAST[4])),u);}
      let good=facts[4];if(good==null){const m=goodMean(age,blasts);good=draw(row('good',blasts,k=>bbRow(k,m*GOOD[3],(1-m)*GOOD[3])),u);}
      const euploid=draw(row('eup',good,k=>bbRow(k,P.euploid*6,(1-P.euploid)*6)),u);
      // L2 per scenario
      const pf=clamp(sig(fl+.07*normal(u)),.01,.99),score=kpiScore(age,amh,mii,mii>0?pn2/Math.max(mii,1)*100:0,good),[a,b]=KPI_BETA[score];
      const pk=clamp(clamp(betaDraw(a,b,u),.005,.95),.01,.99),pc=clamp(sig((1-KPI_WEIGHT)*logit(pf)+KPI_WEIGHT*logit(pk)),.01,.99);
      // transfers: without PGT-A every blastocyst is transferable, good quality first
      const wg=draw(row('warm',good,k=>binomRow(k,WARMING_SURVIVAL)),u),wf=draw(row('warm',blasts-good,k=>binomRow(k,WARMING_SURVIVAL)),u);
      const ng=Math.min(wg,cap),nf=Math.min(wf,cap-ng);
      const x=kanFeatures(age,okk,mii,pn2,blasts,good),kt=kanTerms(x);
      for(let j=0;j<10;j++)terms[j]+=kt.terms[j];for(let j=0;j<18;j++)feat[j]+=x[j];
      S.okk[i]=okk;S.mii[i]=mii;S.pn2[i]=pn2;S.blasts[i]=blasts;S.good[i]=good;S.euploid[i]=euploid;S.nGood[i]=ng;S.nFair[i]=nf;
      S.fortune[i]=pf;S.kpi[i]=pk;S.kpiScore[i]=score;S.comb[i]=pc;S.kanLogit[i]=kt.logit;S.kan[i]=sig(kt.logit);
    }
    S.kanTermsMean=Array.from(terms,v=>v/n);S.kanFeaturesMean=Array.from(feat,v=>v/n);
    return S;
  }
  const stageKeys=['okk','mii','pn2','blasts','good','euploid'];
  function stageSummary(S){return stageKeys.map(key=>{const s=sorted(S[key]);return {key,mean:mean(S[key]),median:quantile(s,.5),low:quantile(s,.025),high:quantile(s,.975)};});}

  // ── L7: BEFE with the L1-L2 prior and the KAN as the only evidence ────
  // befe.py constants; GAT/CSDI absent (precision 1e-6 in 7.1, omitted), no OOD deflation, diffusion neutral.
  const TAU_PRIOR_BASE=1,TAU_KAT=2.4;
  function befe(comb,kan){
    const sc=sorted(comb),sk=sorted(kan),pPrior=mean(comb),pKan=mean(kan);
    const ciPrior=[quantile(sc,.025),quantile(sc,.975)],ciKan=[quantile(sk,.025),quantile(sk,.975)];
    const tauPrior=Math.max(TAU_PRIOR_BASE*Math.exp(-4*(ciPrior[1]-ciPrior[0])),1e-6),tauKan=TAU_KAT/(1+8*((ciKan[1]-ciKan[0])/3.92)**2);
    const lp=logit(clamp(pPrior,1e-6,1-1e-6)),lk=logit(clamp(pKan,1e-6,1-1e-6)),tauPost=tauPrior+tauKan,mu=(tauPrior*lp+tauKan*lk)/tauPost;
    const wPrior=tauPrior/tauPost,wKan=tauKan/tauPost;
    // _uncertainty_range: scenario spread of the fused logits + DerSimonian-Laird model variance
    const fused=Array.from(comb,(p,i)=>wPrior*logit(p)+wKan*logit(clamp(kan[i],1e-6,1-1e-6)));
    const vScen=variance(fused),src=[[tauPrior,lp],[tauKan,lk]],ts=tauPost,tq=tauPrior**2+tauKan**2;
    const q=src.reduce((a,[t,x])=>a+t*(x-mu)**2,0),den=ts-tq/ts,tauB2=den>1e-9?Math.max(0,(q-1)/den):0;
    const vModel=1/src.reduce((a,[t])=>a+1/(1/t+tauB2),0),sd=Math.sqrt(vScen+vModel);
    return {posterior:sig(mu),interval:[sig(mu-1.96*sd),sig(mu+1.96*sd)],pPrior,pKan,ciPrior,ciKan,tauPrior,tauKan,priorPull:wPrior,evidencePull:wKan,
      rangeScenarioSd:Math.sqrt(vScen),rangeModelSd:Math.sqrt(vModel),heteroSd:Math.sqrt(tauB2),fused};
  }
  // embryology.anchored_cycle_probability: shift scenario logits so their transfer mean equals the headline.
  function anchoredCycle(S,target){
    const n=S.comb.length,idx=[];for(let i=0;i<n;i++)if(S.nGood[i]+S.nFair[i]>=1)idx.push(i);
    if(!idx.length)return {pCycle:0,scenario:new Float64Array(n)};
    const lp=Array.from(S.comb,p=>logit(clamp(p,1e-9,1-1e-9))),t=clamp(target,1e-9,1-1e-9),m=d=>idx.reduce((a,i)=>a+sig(lp[i]+d),0)/idx.length;
    let lo=-30,hi=30;for(let k=0;k<100;k++){const c=(lo+hi)/2;if(m(c)<t)lo=c;else hi=c;}
    const d=(lo+hi)/2,scenario=Float64Array.from(lp,(l,i)=>cycleProbability(sig(l+d),S.nGood[i],S.nFair[i]));
    return {pCycle:mean(scenario),scenario};
  }
  const baselineCache=new Map();
  function baseline(age,amh,afc,n){const key=[age,amh,afc,n].join('|');if(!baselineCache.has(key)){if(baselineCache.size>200)baselineCache.clear();baselineCache.set(key,stageSummary(simulate(age,amh,afc,[null,null,null,null,null],1,n)));}return baselineCache.get(key);}

  function calculate(input,options={}) {
    const s=validate(input),p=presets[s.preset],knownCount=[0,2,3,5,5][s.stage],n=options.n||N_SIM;
    const facts=p.facts.map((v,i)=>i<knownCount?v:null);
    const S=simulate(s.age,s.amh,s.afc,facts,s.transfers,n),stages=stageSummary(S),base=baseline(s.age,s.amh,s.afc,n);
    // ivf_core.transfer_view: the prior and the KAN describe only scenarios with a transfer
    const mask=[];for(let i=0;i<n;i++)if(S.nGood[i]+S.nFair[i]>=1)mask.push(i);
    const pick=a=>mask.length?Float64Array.from(mask,i=>a[i]):a,comb=pick(S.comb),kan=pick(S.kan),fusion=befe(comb,kan);
    const {pCycle,scenario}=anchoredCycle(S,fusion.posterior);
    const l1Cycle=mean(Array.from(S.comb,(q,i)=>cycleProbability(q,S.nGood[i],S.nFair[i])));
    const nTx=sorted(Array.from(S.nGood,(g,i)=>g+S.nFair[i])),share=mask.length/n,okk=Array.from(S.okk);
    const kpiSorted=sorted(pick(S.kpiScore));
    return {
      baseline:base.slice(0,5).map(v=>v.mean),ranges:base.slice(0,5).map(v=>[v.low,v.high]),current:stages.slice(0,5).map(v=>v.mean),
      stages,transferable:{median:quantile(nTx,.5),low:quantile(nTx,.025),high:quantile(nTx,.975)},facts,knownCount,
      available:share,pTransfer:fusion.posterior,interval:fusion.interval,pCycle,transfers:Math.min(s.transfers,Math.max(1,Math.round(quantile(nTx,.5)))),observed:s.stage>=3,
      l1:{perTransfer:fusion.pPrior,ci:fusion.ciPrior,fortune:mean(pick(S.fortune)),kpi:mean(pick(S.kpi)),kpiScore:quantile(kpiSorted,.5),kpiScoreRange:[quantile(kpiSorted,.025),quantile(kpiSorted,.975)],cycle:l1Cycle},
      kan:{probability:fusion.pKan,ci:fusion.ciKan,features:KAN.features,featuresMean:S.kanFeaturesMean,terms:S.kanTermsMean,logitMean:mean(S.kanLogit)},
      fusion:{priorPull:fusion.priorPull,evidencePull:fusion.evidencePull,tauPrior:fusion.tauPrior,tauKan:fusion.tauKan,rangeScenarioSd:fusion.rangeScenarioSd,rangeModelSd:fusion.rangeModelSd,heteroSd:fusion.heteroSd},
      empty:{p_no_blast:mean(Array.from(S.blasts,v=>v<1)),p_no_good_blast:mean(Array.from(S.good,v=>v<1)),p_no_transfer:1-share},
      risks:{p_moderate_ohss:mean(okk.map(v=>v>=15&&v<20)),p_severe_ohss:mean(okk.map(v=>v>=20))},
      scenarios:{perTransfer:fusion.fused.map(sig),perCycle:scenario},
      n,source:'dt71-reduced-L1-L3-v1'
    };
  }
  const api={presets,limits,initial,validate,calculate,kan:KAN,kanTerms,kanFeatures};
  root.DemoModel=api;if(typeof module!=='undefined')module.exports=api;
})(typeof globalThis!=='undefined'?globalThis:this);
