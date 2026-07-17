"""Refit recorded aggregate statistical models and save real validation forecast diagnostics."""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ARIMA_OUT = ROOT / "models/classical_statistical_time_series/arima/figures"
SARIMA_OUT = ROOT / "models/classical_statistical_time_series/sarima/figures"
BLUE, GREEN, ORANGE, RED, PURPLE = "#2563EB", "#16A34A", "#F59E0B", "#DC2626", "#7C3AED"


def weekly_total(frame):
    return frame.groupby("Date")["Weekly_Sales"].sum().sort_index().asfreq("W-FRI")


def build_repaired_exog(features):
    frame = features.copy(); markdown = [c for c in frame if c.startswith("MarkDown")]
    frame[markdown] = frame[markdown].fillna(0.0)
    for col in ["Temperature", "Fuel_Price", "CPI", "Unemployment"]:
        frame[col] = frame[col].fillna(frame[col].median())
    aggregation = {"IsHoliday":"mean", "Temperature":"mean", "Fuel_Price":"mean", "CPI":"mean", "Unemployment":"mean", **{c:"sum" for c in markdown}}
    weekly = frame.groupby("Date").agg(aggregation).sort_index().asfreq("W-FRI").rename(columns={"IsHoliday":"holiday_share"})
    weekly["total_markdown"] = weekly[markdown].sum(axis=1)
    week = weekly.index.isocalendar().week.astype(int)
    weekly["week_sin"] = np.sin(2*np.pi*week/52); weekly["week_cos"] = np.cos(2*np.pi*week/52)
    weekly["week_sin_2"] = np.sin(4*np.pi*week/52); weekly["week_cos_2"] = np.cos(4*np.pi*week/52)
    weekly["log_total_markdown"] = np.log1p(weekly["total_markdown"].clip(lower=0))
    return weekly.replace([np.inf,-np.inf],np.nan).ffill().bfill().fillna(0)


def build_old_sarimax_exog(features, train_dates, target):
    frame=features.copy(); markdown=[c for c in frame if c.startswith("MarkDown")]
    frame[markdown]=frame[markdown].fillna(0.0)
    for col in ["Temperature","Fuel_Price","CPI","Unemployment"]:
        frame[col]=frame[col].fillna(frame[col].median())
    aggregation={"IsHoliday":"mean","Temperature":"mean","Fuel_Price":"mean","CPI":"mean","Unemployment":"mean",**{c:"sum" for c in markdown}}
    weekly=frame.groupby("Date").agg(aggregation).sort_index().asfreq("W-FRI").rename(columns={"IsHoliday":"holiday_share"})
    weekly["total_markdown"]=weekly[markdown].sum(axis=1); weekly["month"]=weekly.index.month
    week=weekly.index.isocalendar().week.astype(int); weekly["weekofyear"]=week
    weekly["week_sin"]=np.sin(2*np.pi*week/52); weekly["week_cos"]=np.cos(2*np.pi*week/52); weekly["is_december"]=(weekly["month"]==12).astype(int)
    weekly=weekly.replace([np.inf,-np.inf],np.nan).ffill().bfill().fillna(0)
    train_x=weekly.loc[train_dates]; ranked=[]
    y=pd.Series(target,index=train_dates,dtype=float)
    for col in train_x:
        values=train_x[col].astype(float)
        if values.nunique()<=1: continue
        corr=values.corr(y)
        if pd.notna(corr): ranked.append((col,abs(float(corr))))
    selected=[c for c,v in sorted(ranked,key=lambda z:z[1],reverse=True) if v>=.05]
    if not selected: selected=[c for c,_ in sorted(ranked,key=lambda z:z[1],reverse=True)[:8]]
    corr=train_x[selected].corr().abs(); pruned=[]
    for col in selected:
        if len(pruned)>=8: break
        if all(corr.loc[col,prev]<.95 for prev in pruned): pruned.append(col)
    return weekly,pruned


def allocate(target, history, aggregate):
    rows=target[["Store","Dept","Date","IsHoliday","Weekly_Sales"]].copy()
    prior=history[["Store","Dept","Date","Weekly_Sales"]].copy(); prior["Date"] += pd.Timedelta(days=364); prior=prior.rename(columns={"Weekly_Sales":"last_year"})
    fallback=history.groupby(["Store","Dept"],as_index=False)["Weekly_Sales"].mean().rename(columns={"Weekly_Sales":"series_mean"})
    rows=rows.merge(prior,on=["Store","Dept","Date"],how="left").merge(fallback,on=["Store","Dept"],how="left")
    rows["base"]=rows["last_year"].fillna(rows["series_mean"]).fillna(0).clip(lower=0)
    denominator=rows.groupby("Date")["base"].transform("sum"); count=rows.groupby("Date")["base"].transform("size")
    rows["share"]=np.where(denominator>0,rows["base"]/denominator,1/count)
    rows["AggregateForecast"]=rows["Date"].map(aggregate); rows["Prediction"]=(rows["AggregateForecast"]*rows["share"]).clip(lower=0)
    rows["AbsoluteError"]=(rows["Weekly_Sales"]-rows["Prediction"]).abs()
    return rows


def forecast_models(train_part, val_part, features):
    totals=weekly_total(train_part); dates=sorted(val_part.Date.unique()); results={}
    arima=ARIMA(totals,order=(1,0,2),enforce_stationarity=False,enforce_invertibility=False).fit()
    aggregate=pd.Series(np.asarray(arima.forecast(len(dates))),index=dates).clip(lower=0)
    results["ARIMA"]=(allocate(val_part,train_part,aggregate),aggregate,"Best ARIMA (1,0,2)",ARIMA_OUT/"forecast_diagnostics_arima")
    sarima=SARIMAX(totals,order=(1,0,2),enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
    aggregate=pd.Series(np.asarray(sarima.forecast(len(dates))),index=dates).clip(lower=0)
    results["SARIMA"]=(allocate(val_part,train_part,aggregate),aggregate,"SARIMA-folder best (1,0,2); seasonal order absent",SARIMA_OUT/"forecast_diagnostics_sarima")
    repaired=build_repaired_exog(features); cols=["week_sin","week_cos","week_sin_2","week_cos_2","holiday_share","log_total_markdown"]
    x_train=repaired.loc[totals.index,cols].astype(float); x_val=repaired.loc[dates,cols].astype(float); center=x_train.mean(); scale=x_train.std().replace(0,1)
    model=SARIMAX(np.log1p(totals.clip(lower=0)),order=(1,0,1),seasonal_order=(0,0,0,0),exog=(x_train-center)/scale,enforce_stationarity=False,enforce_invertibility=False).fit(disp=False,maxiter=300)
    aggregate=pd.Series(np.expm1(np.asarray(model.get_forecast(len(dates),exog=(x_val-center)/scale).predicted_mean)),index=dates).clip(lower=0)
    results["ARIMAX"]=(allocate(val_part,train_part,aggregate),aggregate,"Repaired ARIMAX (1,0,1)",ARIMA_OUT/"forecast_diagnostics_arimax")
    old,cols=build_old_sarimax_exog(features,totals.index,totals); x_train=old.loc[totals.index,cols]; x_val=old.loc[dates,cols]
    model=SARIMAX(totals,order=(0,0,0),seasonal_order=(0,0,0,0),exog=x_train,enforce_stationarity=False,enforce_invertibility=False).fit(disp=False,maxiter=200)
    aggregate=pd.Series(np.asarray(model.get_forecast(len(dates),exog=x_val).predicted_mean),index=dates).clip(lower=0)
    results["SARIMAX"]=(allocate(val_part,train_part,aggregate),aggregate,f"SARIMAX-folder (0,0,0), {len(cols)} selected exogenous features",SARIMA_OUT/"forecast_diagnostics_sarimax")
    return results


def save_fig(fig,out,name):
    out.mkdir(parents=True,exist_ok=True); fig.tight_layout(); fig.savefig(out/name,dpi=230,bbox_inches="tight",facecolor="white"); plt.close(fig)


def diagnostics(name, rows, aggregate, title, out):
    weekly=rows.groupby("Date",as_index=False).agg(Actual=("Weekly_Sales","sum"),Forecast=("Prediction","sum"),MAE=("AbsoluteError","mean"),IsHoliday=("IsHoliday","max"))
    fig,ax=plt.subplots(figsize=(14,6)); ax.plot(weekly.Date,weekly.Actual/1e6,"o-",label="Actual",color=BLUE); ax.plot(weekly.Date,weekly.Forecast/1e6,"o-",label=name,color=ORANGE)
    holiday=weekly[weekly.IsHoliday]; ax.scatter(holiday.Date,holiday.Actual/1e6,color=RED,s=80,label="Holiday actual",zorder=4); ax.set(title=f"{title}: aggregate validation forecast",xlabel="Date",ylabel="Total weekly sales (million)"); ax.grid(alpha=.2); ax.legend(); save_fig(fig,out,"01_aggregate_actual_vs_forecast.png")
    stats=rows.groupby(["Store","Dept"],as_index=False).agg(MeanSales=("Weekly_Sales","mean"),MAE=("AbsoluteError","mean")); high=tuple(stats.loc[stats.MeanSales.idxmax(),["Store","Dept"]].astype(int)); ordered=stats.sort_values("MeanSales").reset_index(drop=True); median=tuple(ordered.loc[len(ordered)//2,["Store","Dept"]].astype(int)); worst=tuple(stats.loc[stats.MAE.idxmax(),["Store","Dept"]].astype(int))
    fig,axes=plt.subplots(3,1,figsize=(14,12),sharex=True)
    for ax,(label,key) in zip(axes,[("Highest sales volume",high),("Median sales volume",median),("Highest validation MAE",worst)]):
        subset=rows[(rows.Store==key[0])&(rows.Dept==key[1])].sort_values("Date"); ax.plot(subset.Date,subset.Weekly_Sales,"o-",label="Actual"); ax.plot(subset.Date,subset.Prediction,"o-",label="Forecast"); ax.set_title(f"{label}: Store {key[0]}, Dept {key[1]}"); ax.set_ylabel("Weekly_Sales"); ax.grid(alpha=.2); ax.legend()
    axes[-1].set_xlabel("Date"); fig.suptitle(f"{title}: representative validation forecasts",fontsize=16,weight="bold"); save_fig(fig,out,"02_representative_series.png")
    weekly["Horizon"]=np.arange(1,len(weekly)+1); fig,ax=plt.subplots(figsize=(13,6)); ax.plot(weekly.Horizon,weekly.MAE,"o-",color=BLUE); h=weekly[weekly.IsHoliday]; ax.scatter(h.Horizon,h.MAE,color=RED,s=85,label="Holiday"); ax.set(title=f"{title}: error by forecast horizon",xlabel="Forecast horizon (week)",ylabel="Row-level MAE"); ax.grid(alpha=.2); ax.legend(); save_fig(fig,out,"03_error_by_forecast_horizon.png")
    groups=rows.groupby("IsHoliday").AbsoluteError.mean(); labels=["Non-holiday","Holiday"]; values=[groups.get(False,np.nan),groups.get(True,np.nan)]; fig,ax=plt.subplots(figsize=(9,6)); bars=ax.bar(labels,values,color=[BLUE,ORANGE]);
    for bar,value in zip(bars,values): ax.text(bar.get_x()+bar.get_width()/2,value+40,f"{value:,.1f}",ha="center",weight="bold")
    ax.set(title=f"{title}: holiday vs non-holiday error",ylabel="MAE on original sales scale"); ax.grid(axis="y",alpha=.2); save_fig(fig,out,"04_holiday_vs_nonholiday.png")
    weekly.to_csv(out/"weekly_validation_errors.csv",index=False)


if __name__ == "__main__":
    train=pd.read_csv(DATA/"train.csv",parse_dates=["Date"]).sort_values(["Date","Store","Dept"]); features=pd.read_csv(DATA/"features.csv",parse_dates=["Date"])
    dates=np.array(sorted(train.Date.unique())); start=dates[-39]; train_part=train[train.Date<start].copy(); val_part=train[train.Date>=start].copy()
    for name,(rows,aggregate,title,out) in forecast_models(train_part,val_part,features).items():
        print(f"Generating {name} diagnostics -> {out}"); diagnostics(name,rows,aggregate,title,out)
