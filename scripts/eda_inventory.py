"""Inventory PathSnap trip data and write simple GPS/road/route overlays.

Usage:
  python scripts/eda_inventory.py --zip "data/Student folder.zip" --out results/eda
"""
from __future__ import annotations
import argparse, csv, math, os, pathlib, zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont

EXTS = ('.track', '.nodes', '.arcs', '.route')

def read_array(zf, path, cols=None, dtype=float):
    arr = np.loadtxt(zf.open(path), dtype=dtype)
    if cols is not None:
        arr = arr.reshape(-1, cols)
    else:
        arr = np.atleast_1d(arr)
    return arr

def trip_paths(zf):
    trips = {}
    for name in zf.namelist():
        p = name.split('/')
        if len(p) == 4 and p[-1]:
            ext = os.path.splitext(p[-1])[1]
            if ext in EXTS:
                trips.setdefault((p[1], p[2]), {})[ext] = name
    return trips

def inventory(zf, trips):
    rows=[]
    for (part, tid), files in sorted(trips.items()):
        track=read_array(zf,files['.track'],3)
        nodes=read_array(zf,files['.nodes'],2)
        arcs=read_array(zf,files['.arcs'],2,int)
        route=read_array(zf,files['.route'])
        dt=np.diff(track[:,2])
        finite=np.isfinite(track).all(axis=1)
        route_ok=(route>=0)&(route<len(arcs))
        badnode=((arcs<0)|(arcs>=len(nodes))).any(axis=1)
        rows.append({'partition':part,'trip_id':tid,'gps_points':len(track),'nodes':len(nodes),'arcs':len(arcs),'route_arcs':len(route),'dt_eq_1_pct':round(100*float(np.mean(dt==1)),2) if len(dt) else 0,'dt_gt_5_count':int((dt>5).sum()),'invalid_gps_rows':int((~finite).sum()),'route_arc_ids_invalid':int((~route_ok).sum()),'arcs_bad_node_ids':int(badnode.sum()),'lon_min':float(np.nanmin(track[:,0])),'lon_max':float(np.nanmax(track[:,0])),'lat_min':float(np.nanmin(track[:,1])),'lat_max':float(np.nanmax(track[:,1]))})
    return rows

def summarize(rows):
    out=[]
    for part in sorted({r['partition'] for r in rows}):
        rr=[r for r in rows if r['partition']==part]
        gps=[r['gps_points'] for r in rr]
        out.append({'partition':part,'trips':len(rr),'gps_total':sum(gps),'gps_median':int(np.median(gps)),'gps_min':min(gps),'gps_max':max(gps),'nodes_median':int(np.median([r['nodes'] for r in rr])),'arcs_median':int(np.median([r['arcs'] for r in rr])),'route_arcs_median':int(np.median([r['route_arcs'] for r in rr])),'trips_with_gap_gt_5s':sum(r['dt_gt_5_count']>0 for r in rr),'invalid_gps_rows':sum(r['invalid_gps_rows'] for r in rr),'invalid_route_arc_refs':sum(r['route_arc_ids_invalid'] for r in rr),'arcs_with_bad_node_ids':sum(r['arcs_bad_node_ids'] for r in rr)})
    return out

def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def overlay(zf, files, part, tid, out):
    prefix=f'Student folder/{part}/{tid}/{tid}'
    track=read_array(zf,prefix+'.track',3);nodes=read_array(zf,prefix+'.nodes',2);arcs=read_array(zf,prefix+'.arcs',2,int);route=read_array(zf,prefix+'.route').astype(int)
    lat0=float(np.mean(track[:,1]));lon0=float(np.mean(track[:,0]));sx=111320*math.cos(math.radians(lat0));sy=110574
    def xy(p): return np.column_stack(((p[:,0]-lon0)*sx,(p[:,1]-lat0)*sy))
    ns=xy(nodes);gps=xy(track[:,:2]);road=ns[arcs];ids=route[(route>=0)&(route<len(arcs))];rseg=road[ids]
    extent=np.vstack([road.reshape(-1,2),gps,rseg.reshape(-1,2)]);mn=extent.min(0);mx=extent.max(0);pad=np.maximum((mx-mn)*.035,10);mn-=pad;mx+=pad
    W,H=1400,1000;im=Image.new('RGB',(W,H),'white');dr=ImageDraw.Draw(im)
    try: font=ImageFont.truetype('arial.ttf',23);small=ImageFont.truetype('arial.ttf',18)
    except OSError: font=small=ImageFont.load_default()
    dr.text((30,18),f'{part} trip {tid}: full network + supplied route + GPS',fill='#111111',font=font)
    def proj(p): return np.column_stack((50+(p[:,0]-mn[0])/(mx[0]-mn[0])*(W-100),H-70-(p[:,1]-mn[1])/(mx[1]-mn[1])*(H-140))).astype(int)
    a=proj(road[:,0]);b=proj(road[:,1])
    for p,q in zip(a,b): dr.line([tuple(p),tuple(q)],fill='#d5dbe0',width=1)
    r=proj(rseg.reshape(-1,2)).reshape(-1,2,2)
    for p,q in r: dr.line([tuple(p),tuple(q)],fill='#079447',width=3)
    g=proj(gps);step=max(1,len(g)//2500)
    for p in g[::step]: dr.ellipse((p[0]-2,p[1]-2,p[0]+2,p[1]+2),fill='#2778c8')
    dr.text((50,H-48),f'Blue GPS ({len(gps)} pts)   Green ordered .route ({len(ids)} arcs)   Grey network ({len(road)} arcs)',fill='#222222',font=small)
    im.save(out/f'overlay_{part}_{tid}.png')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--zip',required=True);ap.add_argument('--out',default='results/eda');args=ap.parse_args();out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.zip) as zf:
        trips=trip_paths(zf);rows=inventory(zf,trips)
        write_csv(out/'trip_inventory.csv',rows);write_csv(out/'partition_summary.csv',summarize(rows))
        for part,tid in [('student_test','00000040'),('student_test','00000076'),('train','00000004')]: overlay(zf,trips[(part,tid)],part,tid,out)
    print(f'Wrote inventory and overlays for {len(rows)} trips to {out}')
if __name__=='__main__': main()
