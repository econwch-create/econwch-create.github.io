from __future__ import annotations

import csv
import math
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from huggingface_hub import hf_hub_download

OUT = Path('outputs')
OUT.mkdir(exist_ok=True)

CITY_CODES = [110000,120000,130100,130200,130300,130400,130500,130600,130700,130800,130900,131000,131100,140100,140200,140300,140400,140500,140600,140700,140800,140900,141000,141100,150100,150200,150300,150400,150500,150600,150700,150800,150900,210100,210200,210300,210400,210500,210600,210700,210800,210900,211000,211100,211200,211300,211400,220100,220200,220300,220400,220500,220600,220700,220800,230100,230200,230300,230400,230500,230600,230700,230800,230900,231000,231100,231200,310000,320100,320200,320300,320400,320500,320600,320700,320800,320900,321000,321100,321200,321300,330100,330200,330300,330400,330500,330600,330700,330800,330900,331000,331100,340100,340200,340300,340400,340500,340600,340700,340800,341000,341100,341200,341300,341500,341600,341700,341800,350100,350200,350300,350400,350500,350600,350700,350800,350900,360100,360200,360300,360400,360500,360600,360700,360800,360900,361000,361100,370100,370200,370300,370400,370500,370600,370700,370800,370900,371000,371100,371300,371400,371500,371600,371700,410100,410200,410300,410400,410500,410600,410700,410800,410900,411000,411100,411200,411300,411400,411500,411600,411700,420100,420200,420300,420500,420600,420700,420800,420900,421000,421100,421200,421300,430100,430200,430300,430400,430500,430600,430700,430800,430900,431000,431100,431200,431300,440100,440200,440300,440400,440500,440600,440700,440800,440900,441200,441300,441400,441500,441600,441700,441800,441900,442000,445100,445200,445300,450100,450200,450300,450400,450500,450600,450700,450800,450900,451000,451100,451200,451300,451400,460100,460200,460300,460400,500000,510100,510300,510400,510500,510600,510700,510800,510900,511000,511100,511300,511400,511500,511600,511700,511800,511900,512000,520100,520200,520300,520400,520500,520600,530100,530300,530400,530500,530600,530700,530800,530900,540100,540200,540300,540400,540500,540600,610100,610200,610300,610400,610500,610600,610700,610800,610900,611000,620100,620200,620300,620400,620500,620600,620700,620800,620900,621000,621100,621200,630100,630200,632500,640100,640200,640300,640400,640500,650100,650200,650400,650500]

FALLBACK = {
    460300: ('海南省','三沙市',112.348820,16.831039),
    460400: ('海南省','儋州市',109.576782,19.517486),
    520500: ('贵州省','毕节市',105.285010,27.301693),
    520600: ('贵州省','铜仁市',109.191555,27.718346),
    540200: ('西藏自治区','日喀则市',88.885148,29.267519),
    540300: ('西藏自治区','昌都市',97.178452,31.136875),
    540400: ('西藏自治区','林芝市',94.362348,29.654693),
    540500: ('西藏自治区','山南市',91.766529,29.236023),
    540600: ('西藏自治区','那曲市',92.060214,31.476004),
    630200: ('青海省','海东市',102.103270,36.502916),
    650400: ('新疆维吾尔自治区','吐鲁番市',89.184078,42.947613),
    650500: ('新疆维吾尔自治区','哈密市',93.513160,42.833248),
}

AI_TERMS = ['人工智能','机器学习','深度学习','神经网络','卷积神经网络','循环神经网络','图神经网络','强化学习','迁移学习','联邦学习','支持向量机','随机森林','决策树','贝叶斯网络','计算机视觉','机器视觉','图像识别','图像分类','目标检测','目标识别','语义分割','实例分割','模式识别','自然语言处理','知识图谱','专家系统','智能决策','智能诊断','智能预测','BP神经网络','LSTM','Transformer','YOLO']
CROP_TERMS = ['农业','农田','农场','农作物','作物','种植','播种','育苗','插秧','耕作','灌溉','施肥','水肥一体化','土壤墒情','病虫害','植保','农业遥感','农情监测','长势监测','农业无人机','植保无人机','农业机器人','智能农机','无人农机','收割','收获','脱粒','产量预测','杂草识别','智慧农业','精准农业','数字农业']
GRAIN_TERMS = ['粮食','粮食作物','谷物','稻谷','稻米','水稻','稻田','小麦','麦田','玉米','玉米田','大豆','高粱','谷子','青稞','燕麦','荞麦','薯类','马铃薯','甘薯','主粮']
NONCROP_TERMS = ['畜牧','养殖','生猪','猪舍','肉牛','奶牛','家禽','蛋鸡','肉鸡','水产','鱼塘','养鱼','渔业','宠物','兽医','屠宰','食品加工','食品包装','乳制品','肉制品']
DIGITAL_TERMS = ['物联网','大数据','云计算','区块链','遥感','北斗','传感器','无人机','数字孪生','GIS','5G']
AI_IPC = ('G06N','G06V','G06T7','G06K9','G06F18')
CROP_IPC = ('A01B','A01C','A01D','A01F','A01G','A01H')


def pat(terms: list[str]) -> re.Pattern[str]:
    return re.compile('|'.join(sorted(map(re.escape, terms), key=len, reverse=True)), re.I)

P_AI, P_CROP, P_GRAIN, P_NONCROP, P_DIGITAL = map(pat, [AI_TERMS, CROP_TERMS, GRAIN_TERMS, NONCROP_TERMS, DIGITAL_TERMS])


def fetch_cities() -> list[dict[str, Any]]:
    url = 'https://raw.githubusercontent.com/yhdjyyzk/GeoJSON_data/master/%E5%85%A8%E5%9B%BD%E5%8E%BF%E7%BA%A7%E4%BB%A5%E4%B8%8A%E5%9C%B0%E5%90%8D%E4%BB%A3%E7%A0%81%E5%8F%8A%E7%BB%8F%E7%BA%AC%E5%BA%A6.csv'
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    text = r.content.decode('gb18030', errors='replace')
    rows = list(csv.DictReader(text.splitlines()))
    by_code = {}
    for row in rows:
        try:
            code = int(row['行政代码'])
            by_code[code] = row
        except Exception:
            pass
    cities = []
    for code in CITY_CODES:
        if code in by_code:
            row = by_code[code]
            cities.append({'省份': '', '城市': row['地名'].strip(), '城市代码': code, '经度': float(row['东经']), '纬度': float(row['北纬'])})
        elif code in FALLBACK:
            province, city, lon, lat = FALLBACK[code]
            cities.append({'省份': province, '城市': city, '城市代码': code, '经度': lon, '纬度': lat})
        else:
            raise RuntimeError(f'Coordinate missing: {code}')
    pd.DataFrame(cities).to_csv(OUT/'研究城市坐标_298市.csv', index=False, encoding='utf-8-sig')
    return cities


def filter_patents() -> None:
    print('Downloading public patent sample...')
    source = hf_hub_download(repo_id='LiuHongwei1992/520634PatentsDataset', repo_type='dataset', filename='patents_sample_500k.csv')
    selected = []
    summary_parts = []
    chunks = pd.read_csv(source, chunksize=50000, low_memory=False)
    for n, df in enumerate(chunks, 1):
        title = df.get('专利名称', '').fillna('').astype(str)
        abstract = df.get('摘要文本', '').fillna('').astype(str)
        text = title + ' ' + abstract
        ipc = df.get('IPC主分类号', '').fillna('').astype(str).str.upper().str.replace(' ', '', regex=False)
        has_ai = text.str.contains(P_AI, na=False) | ipc.str.startswith(AI_IPC)
        crop = text.str.contains(P_CROP, na=False) | ipc.str.startswith(CROP_IPC)
        grain = text.str.contains(P_GRAIN, na=False)
        noncrop = text.str.contains(P_NONCROP, na=False) & ~crop & ~grain
        digital = text.str.contains(P_DIGITAL, na=False)
        df['农业AI_宽口径'] = has_ai & (crop | grain | ipc.str.startswith('A01'))
        df['作物AI_主口径'] = has_ai & crop & ~noncrop
        df['粮食AI_严格口径'] = has_ai & grain
        df['数字农业非AI'] = ~has_ai & digital & (crop | grain | ipc.str.startswith('A01'))
        df['AI文本证据'] = text.str.contains(P_AI, na=False)
        df['AI_IPC证据'] = ipc.str.startswith(AI_IPC)
        df['作物证据'] = crop
        df['粮食证据'] = grain
        keep = df['农业AI_宽口径'] | df['作物AI_主口径'] | df['粮食AI_严格口径'] | df['数字农业非AI']
        out = df.loc[keep].copy()
        selected.append(out)
        grp = out.groupby(['申请人城市','申请年份'], dropna=False)[['农业AI_宽口径','作物AI_主口径','粮食AI_严格口径','数字农业非AI']].sum().reset_index()
        summary_parts.append(grp)
        print(f'Patent chunk {n}: selected {len(out):,}')
    patents = pd.concat(selected, ignore_index=True)
    patents.to_csv(OUT/'公开样本_农业AI专利明细.csv.gz', index=False, encoding='utf-8-sig', compression='gzip')
    city_year = pd.concat(summary_parts).groupby(['申请人城市','申请年份'], as_index=False)[['农业AI_宽口径','作物AI_主口径','粮食AI_严格口径','数字农业非AI']].sum()
    city_year.to_csv(OUT/'公开样本_农业AI城市年度计数.csv', index=False, encoding='utf-8-sig')
    overview = pd.DataFrame({'指标':['原始抽样专利数','筛选后记录数','宽口径农业AI','作物AI主口径','粮食AI严格口径','数字农业非AI','覆盖城市数','年份最小值','年份最大值'],'数值':[520634,len(patents),int(patents['农业AI_宽口径'].sum()),int(patents['作物AI_主口径'].sum()),int(patents['粮食AI_严格口径'].sum()),int(patents['数字农业非AI'].sum()),int(patents['申请人城市'].nunique()),int(pd.to_numeric(patents['申请年份'], errors='coerce').min()),int(pd.to_numeric(patents['申请年份'], errors='coerce').max())]})
    overview['说明'] = ['Hugging Face分层抽样数据','至少命中农业AI或数字农业规则','','','','','', '', '']
    overview.to_csv(OUT/'公开样本_农业AI筛选摘要.csv', index=False, encoding='utf-8-sig')
    cols = [c for c in ['专利类型','专利名称','摘要文本','申请人类型','申请人城市','申请号','申请日','申请年份','IPC主分类号','农业AI_宽口径','作物AI_主口径','粮食AI_严格口径','数字农业非AI'] if c in patents.columns]
    for name, mask in [('作物AI',patents['作物AI_主口径']),('粮食AI',patents['粮食AI_严格口径']),('边界案例',patents['农业AI_宽口径'] & ~patents['作物AI_主口径'])]:
        subset = patents.loc[mask, cols]
        if len(subset): subset.sample(min(200,len(subset)), random_state=42).to_csv(OUT/f'人工核验样本_{name}.csv',index=False,encoding='utf-8-sig')


def safe(v: Any) -> float | None:
    try:
        x=float(v)
        return None if x <= -900 or not math.isfinite(x) else x
    except Exception:
        return None


def heatwave(vals: list[tuple[datetime,float|None]], threshold=35.0, minrun=3) -> tuple[int,int]:
    events=days=run=0
    prev=None
    def end_run(r): return (1,r) if r>=minrun else (0,0)
    for d,v in sorted(vals):
        consecutive = prev is not None and (d-prev).days==1
        if v is not None and v>threshold:
            if consecutive: run += 1
            else:
                e,dd=end_run(run); events+=e; days+=dd; run=1
        else:
            e,dd=end_run(run); events+=e; days+=dd; run=0
        prev=d
    e,dd=end_run(run); return events+e,days+dd


def summarize_city(city: dict[str,Any], p: dict[str,dict[str,Any]]) -> list[dict[str,Any]]:
    dates=sorted(set(p.get('T2M_MAX',{})) | set(p.get('PRECTOTCORR',{})))
    by=defaultdict(list)
    for key in dates:
        if not (len(key)==8 and key.isdigit()): continue
        d=datetime.strptime(key,'%Y%m%d')
        if 2001<=d.year<=2023:
            by[d.year].append((d,safe(p.get('T2M_MAX',{}).get(key)),safe(p.get('PRECTOTCORR',{}).get(key))))
    rows=[]
    for y in range(2001,2024):
        obs=by[y]; t=[x[1] for x in obs if x[1] is not None]; pr=[x[2] for x in obs if x[2] is not None]
        grow=[x for x in obs if 4<=x[0].month<=10]; gt=[x[1] for x in grow if x[1] is not None]; gp=[x[2] for x in grow if x[2] is not None]
        he,hd=heatwave([(x[0],x[1]) for x in obs]); ghe,ghd=heatwave([(x[0],x[1]) for x in grow])
        rows.append({**city,'年份':y,'有效气温天数':len(t),'有效降水天数':len(pr),'tmax_mean_C':sum(t)/len(t) if t else None,'tmax_max_C':max(t) if t else None,'hot_days_30':sum(v>30 for v in t),'hot_days_32':sum(v>32 for v in t),'hot_days_35':sum(v>35 for v in t),'heat_degree_days_30':sum(max(v-30,0) for v in t),'heat_degree_days_35':sum(max(v-35,0) for v in t),'heatwave_events_35_3d':he,'heatwave_days_35_3d':hd,'precip_total_mm':sum(pr) if pr else None,'rain_days_ge1mm':sum(v>=1 for v in pr),'heavy_rain_days_25mm':sum(v>=25 for v in pr),'heavy_rain_days_50mm':sum(v>=50 for v in pr),'dry_days_lt1mm':sum(v<1 for v in pr),'growseason_tmax_mean_C':sum(gt)/len(gt) if gt else None,'growseason_tmax_max_C':max(gt) if gt else None,'growseason_hot_days_30':sum(v>30 for v in gt),'growseason_hot_days_32':sum(v>32 for v in gt),'growseason_hot_days_35':sum(v>35 for v in gt),'growseason_heat_degree_days_30':sum(max(v-30,0) for v in gt),'growseason_heat_degree_days_35':sum(max(v-35,0) for v in gt),'growseason_heatwave_events_35_3d':ghe,'growseason_heatwave_days_35_3d':ghd,'growseason_precip_total_mm':sum(gp) if gp else None,'growseason_rain_days_ge1mm':sum(v>=1 for v in gp),'growseason_heavy_rain_days_25mm':sum(v>=25 for v in gp),'growseason_heavy_rain_days_50mm':sum(v>=50 for v in gp),'growseason_dry_days_lt1mm':sum(v<1 for v in gp),'数据源':'NASA POWER Daily Point API','空间口径':'城市代表点，生长季4-10月'})
    return rows


def fetch_nasa_one(city: dict[str,Any]) -> tuple[list[dict[str,Any]],str|None]:
    url='https://power.larc.nasa.gov/api/temporal/daily/point'
    params={'parameters':'T2M_MAX,PRECTOTCORR','community':'AG','longitude':city['经度'],'latitude':city['纬度'],'start':'20010101','end':'20231231','format':'JSON'}
    last=None
    for attempt in range(6):
        try:
            r=requests.get(url,params=params,timeout=240)
            r.raise_for_status(); p=r.json()['properties']['parameter']
            return summarize_city(city,p),None
        except Exception as e:
            last=repr(e); time.sleep(min(60,2**attempt))
    return [],last


def fetch_climate(cities: list[dict[str,Any]]) -> None:
    allrows=[]; failures=[]
    with ThreadPoolExecutor(max_workers=8) as ex:
        future={ex.submit(fetch_nasa_one,c):c for c in cities}
        for i,f in enumerate(as_completed(future),1):
            c=future[f]
            rows,err=f.result()
            if err: failures.append({**c,'错误':err})
            else: allrows.extend(rows)
            if i%10==0: print(f'NASA POWER cities: {i}/{len(cities)}')
    pd.DataFrame(allrows).sort_values(['城市代码','年份']).to_csv(OUT/'NASA_POWER_298市_2001_2023_年度高温降水.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(failures).to_csv(OUT/'NASA_POWER_下载失败城市.csv',index=False,encoding='utf-8-sig')


def main():
    cities=fetch_cities()
    filter_patents()
    fetch_climate(cities)
    manifest=pd.DataFrame([['公开样本_农业AI专利明细.csv.gz','专利级','2015-2024抽样','可用于分类校准和人工核验，不可作为全国总体回归数量'],['公开样本_农业AI城市年度计数.csv','城市-年份','2015-2024抽样','仅描述抽样分布，不得将样本0当作真实0'],['NASA_POWER_298市_2001_2023_年度高温降水.csv','城市-年份','2001-2023','可直接与现有面板按城市代码+年份合并；城市代表点口径']],columns=['文件','层级','覆盖','用途与限制'])
    manifest.to_csv(OUT/'数据清单.csv',index=False,encoding='utf-8-sig')
    print('Done')

if __name__=='__main__': main()
