#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tier 4.3+ — optuna_realr.py
---------------------------------
Kern: Hyperparameter-Optimierung (Optuna) für Strategie/ML-Parameter.

Features:
- SQLite-Storage (./data/optuna/{study}.db)
- Ergebnisse: ./models/best_params_{symbol}_{timeframe}.json
- Trial-Logs: ./logs/optuna_sessions/{timestamp}/
- Trials-CSV: ./results/optuna_trials/{study}_{timestamp}.csv
- Regime-aware (optional) via models/rl.joblib (auto-fallback)
- Fallback-Backtest (EMA-Crossover), falls ml_core nicht verfügbar
"""

from __future__ import annotations
import os, json, math, argparse, datetime as dt, logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from types import ModuleType

import numpy as np, pandas as pd, optuna
from optuna.pruners import BasePruner

# ------------------------------------------------------------
# Global Setup
use_hyperband: bool = False
disable_pruning: bool = False

BASE = Path(__file__).resolve().parent
DIRS = {
    "logs": BASE / "logs" / "optuna_sessions",
    "optuna": BASE / "data" / "optuna",
    "models": BASE / "models",
    "results": BASE / "results" / "optuna_trials",
}
for d in DIRS.values(): d.mkdir(parents=True, exist_ok=True)

ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = DIRS["logs"] / ts
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "optuna_realr.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("optuna_realr")

# ------------------------------------------------------------
# Safe Imports
def _safe_imports() -> Tuple[Optional[ModuleType], Optional[ModuleType], Optional[ModuleType]]:
    fs_mod = ml_mod = rl_mod = None
    try:
        import features_store as _fs_mod; fs_mod = _fs_mod
    except Exception as e: log.warning("⚠ features_store Importfehler: %s", e)
    try:
        import ml_core as _ml_mod; ml_mod = _ml_mod
    except Exception as e: log.warning("⚠ ml_core Importfehler: %s", e)
    try:
        import regime_learner as _rl_mod; rl_mod = _rl_mod
    except Exception as e: log.info("ℹ regime_learner optional: %s", e)
    return fs_mod, ml_mod, rl_mod
features_store, ml_core, regime_learner = _safe_imports()

# ------------------------------------------------------------
# Helper
def _env_default(k: str, d: str) -> str: return os.getenv(k, d)
def _pct(x: pd.Series) -> pd.Series: return x.pct_change().fillna(0.0)
def _json_sanitize(o: Any) -> Any:
    if o is None or isinstance(o, (bool, int, float, str)): return o
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, dict): return {k: _json_sanitize(v) for k,v in o.items()}
    if isinstance(o, (list,tuple)): return [_json_sanitize(v) for v in o]
    return None

# ------------------------------------------------------------
# Load Data
def _load_data(symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    if not (features_store and hasattr(features_store, "load_ohlcv")):
        raise RuntimeError("❌ features_store fehlt.")
    df = features_store.load_ohlcv(symbol=symbol, timeframe=timeframe, start=start, end=end)
    if df is None or len(df)==0: raise RuntimeError("❌ Keine Daten geladen.")
    if hasattr(features_store, "build_features"):
        try: df = features_store.build_features(df, params=None)
        except Exception as e: log.warning("build_features skipped: %s", e)
    return df

# ------------------------------------------------------------
# Adaptive Regime Filter
def _apply_regime_adaptive(df: pd.DataFrame, model_path: Optional[str], regime_filter: str,
                           min_conf: float, min_rows:int=200,
                           relax_step:float=0.1, floor:float=0.2) -> Tuple[pd.DataFrame,Dict[str,Any]]:
    meta = {"requested":regime_filter,"effective":"all","min_conf_used":0.0,"relaxed":False}
    if not model_path or not regime_learner or not hasattr(regime_learner,"RegimeLearner"): return df, meta
    try:
        rl = regime_learner.RegimeLearner.load(model_path); df2 = rl.transform(df)
    except Exception as e:
        log.warning("⚠ Regime-Transform fehlgeschlagen: %s", e); return df, meta
    if regime_filter=="all": return df2, meta
    cur = float(min_conf)
    while cur >= floor:
        m = (df2["regime_label"].str.lower()==regime_filter.lower())&(df2["regime_conf"]>=cur)
        if m.sum()>=min_rows:
            meta.update({"effective":regime_filter,"min_conf_used":cur}); return df2.loc[m].copy(), meta
        log.info("Regime-Relax %.2f→%.2f: nur %d Zeilen",cur,cur-relax_step,int(m.sum()))
        cur-=relax_step; meta["relaxed"]=True
    log.info("ℹ Regime-Fallback auf 'all'"); return df2, meta

# ------------------------------------------------------------
# Simple EMA Backtest
def _fallback_generate_signals(df: pd.DataFrame,p:Dict[str,Any])->pd.Series:
    ef,es=int(p.get("ema_fast",12)),int(p.get("ema_slow",26))
    emaf,emas=df["close"].ewm(span=ef,adjust=False).mean(),df["close"].ewm(span=es,adjust=False).mean()
    sig=np.where(emaf>emas,1,-1)
    if "regime_conf" in df.columns: sig=np.where(df["regime_conf"]<0.5,0,sig)
    return pd.Series(sig,index=df.index)

def _fallback_backtest(df:pd.DataFrame,sig:pd.Series,p:Dict[str,Any])->Dict[str,Any]:
    fee=float(p.get("fee_bps",6)); rets=_pct(df["close"])
    gross=sig.shift(1).fillna(0)*rets; turns=(sig.diff().abs()>0).astype(float)
    pnl=gross-turns*(fee/1e4); eq=(1+pnl).cumprod()
    dd=(eq/eq.cummax()-1).min(); trades=int(turns.sum())
    net=float(pnl.sum()); win=float((pnl>0).sum()/max((sig!=0).sum(),1))*100
    sharpe=float(pnl.mean()/max(pnl.std(ddof=1),1e-9)*np.sqrt(252*24*4))
    pf=float((pnl[pnl>0].sum()/-min(pnl[pnl<0].sum(),-1e-9))) if (pnl<0).any() else 1.0
    return {"sharpe":sharpe,"profit_factor":pf,"max_drawdown":abs(dd)*100,
            "win_rate":win,"net_profit":net,"trades":trades}

# ------------------------------------------------------------
# Unified Backtest
def _run_backtest(df:pd.DataFrame,p:Dict[str,Any])->Dict[str,Any]:
    if ml_core:
        if hasattr(ml_core,"run_backtest"): return ml_core.run_backtest(df=df,params=p)
        if hasattr(ml_core,"generate_signals") and hasattr(ml_core,"backtest"):
            s=ml_core.generate_signals(df=df,params=p)
            return ml_core.backtest(df=df,signals=s,params=p)
    return _fallback_backtest(df,_fallback_generate_signals(df,p),p)

# ------------------------------------------------------------
# Score Function
def _score(m:Dict[str,Any])->float:
    s,pf,dd,win,net=[float(m.get(k,0) or 0) for k in["sharpe","profit_factor","max_drawdown","win_rate","net_profit"]]
    if win>1: win/=100; pf=min(pf,5); s=max(min(s,5),-2)
    return s+0.6*pf+0.25*math.log1p(max(net,-0.999))+0.2*win-1.2*(0.5*(dd/100))

# ------------------------------------------------------------
# Param Grid
def _suggest_params(t:optuna.Trial)->Dict[str,Any]:
    return {"ema_fast":t.suggest_int("ema_fast",5,30),
            "ema_slow":t.suggest_int("ema_slow",20,120),
            "fee_bps":t.suggest_float("fee_bps",2,10)}

# ------------------------------------------------------------
# Main Optimization
def run_optuna(symbol:str,timeframe:str,start:str,end:str,study_name:str,
               n_trials:int=100,storage_path:Optional[Path]=None,pruner:str="median",
               regime_model_path:Optional[str]=None,regime_filter:str="all",
               min_conf:float=0.0)->Tuple[Dict[str,Any],Path]:
    if storage_path is None: storage_path=DIRS["optuna"]/f"{study_name}.db"
    storage_url=f"sqlite:///{storage_path}"
    pruner_obj={"median":optuna.pruners.MedianPruner(),
                "hyperband":optuna.pruners.HyperbandPruner(),
                "none":optuna.pruners.NopPruner()}.get(pruner,optuna.pruners.NopPruner())
    sampler=optuna.samplers.TPESampler(seed=1337)
    log.info(f"🚀 Storage: {storage_url}")
    df_raw=_load_data(symbol,timeframe,start,end)
    df,meta=_apply_regime_adaptive(df_raw,regime_model_path,regime_filter,float(min_conf))
    if len(df)<50: raise RuntimeError(f"❌ Zu wenige Zeilen nach Filter ({len(df)})")
    eff_study=f"{study_name}__{meta['effective']}_c{meta['min_conf_used']:.2f}"
    study=optuna.create_study(study_name=eff_study,storage=storage_url,
                              load_if_exists=True,direction="maximize",
                              pruner=pruner_obj,sampler=sampler)

    def obj(trial:optuna.Trial)->float:
        p=_suggest_params(trial); 
        try: m=_run_backtest(df,p)
        except Exception as e: log.warning("Trial %d fail: %s",trial.number,e); return -10
        s=_score(m); 
        trial.set_user_attr("metrics",_json_sanitize(m)); trial.set_user_attr("score",s); return s

    log.info(f"▶ Starte {eff_study} ({n_trials} Trials)")
    study.optimize(obj,n_trials=n_trials,n_jobs=1,show_progress_bar=False)
    log.info(f"✅ Beste Trial #{study.best_trial.number} Score={study.best_value:.3f}")
    best=dict(study.best_trial.params)
    best["_meta"]={"symbol":symbol,"tf":timeframe,"study":eff_study,
                   "regime":meta,"timestamp":dt.datetime.now().isoformat()}
    outp=DIRS["models"]/f"best_params_{symbol}_{timeframe}.json"
    json.dump(best,open(outp,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    log.info(f"💾 Saved {outp}")

    # Export Trials
    rows=[]
    for t in study.trials:
        m=_json_sanitize(t.user_attrs.get("metrics") or {})
        rows.append({"n":t.number,"v":t.value,"state":str(t.state),
                     "score":t.user_attrs.get("score"),**t.params,**(m or {})})
    pd.DataFrame(rows).to_csv(DIRS["results"]/f"{eff_study}_{ts}.csv",index=False)
    return best,outp

# ------------------------------------------------------------
# CLI
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--symbol",default=_env_default("OPTUNA_SYMBOL","BTCUSDT"))
    p.add_argument("--timeframe",default=_env_default("OPTUNA_TIMEFRAME","15m"))
    p.add_argument("--start",default=_env_default("OPTUNA_START","2023-01-01"))
    p.add_argument("--end",default=_env_default("OPTUNA_END","2025-01-01"))
    p.add_argument("--study",default=_env_default("OPTUNA_STUDY_NAME","tier43_plus"))
    p.add_argument("--trials",type=int,default=int(_env_default("OPTUNA_N_TRIALS","100")))
    p.add_argument("--pruner",default=_env_default("OPTUNA_PRUNER","median"),
                   choices=["median","hyperband","none"])
    p.add_argument("--regime_model",default=_env_default("REGIME_MODEL",str(DIRS["models"]/ "rl.joblib")))
    p.add_argument("--regime_filter",default=_env_default("REGIME_FILTER","all"),
                   choices=["all","bull","bear","sideways"])
    p.add_argument("--min_conf",type=float,default=float(_env_default("REGIME_MIN_CONF","0.0")))
    a=p.parse_args()
    rm=a.regime_model if a.regime_model and os.path.exists(a.regime_model) else None
    best,_=run_optuna(a.symbol,a.timeframe,a.start,a.end,a.study,a.trials,a.pruner,rm,a.regime_filter,a.min_conf)
    print(json.dumps({"ok":True,"best":best.get('_meta',{})},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
