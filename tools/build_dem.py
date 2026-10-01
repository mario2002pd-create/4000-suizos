"""Genera datos reales de relieve para cuatromiles-suizos.html.

Entradas (descargadas de internet):
  - Mosaicos AWS Terrarium z10 (modelo digital de elevación abierto: SRTM, ASTER, EU-DEM, etc.)
  - Contornos de cantones (click_that_hood / Natural Earth-like, GeoJSON)
Salida: tools/dem.json con {png: data-URI PNG, rings...}. El PNG codifica
  R,G = (altura+500 m)*8 en 16 bits (0.125 m) · B = bits: 1 agua, 2 dentro de Suiza, 4 dentro de Graubünden.
"""
import io, json, math, sys, urllib.request, concurrent.futures as cf
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates, maximum_filter, minimum_filter

Z=10; LO0,LO1,LA0,LA1=5.85,10.65,45.72,47.88; NX,NZ=1600,1070
def tile_xy(lon,lat):
    n=2**Z; x=(lon+180)/360*n; y=(1-math.log(math.tan(math.radians(lat))+1/math.cos(math.radians(lat)))/math.pi)/2*n; return x,y
x0,y1=tile_xy(LO0,LA0); x1,y0=tile_xy(LO1,LA1)
tx0,tx1,ty0,ty1=int(x0),int(x1),int(y0),int(y1)
print("tiles",tx1-tx0+1,"x",ty1-ty0+1,file=sys.stderr)
def get(t):
    x,y=t
    for _ in range(4):
        try:
            b=urllib.request.urlopen(f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{Z}/{x}/{y}.png",timeout=60).read()
            im=np.asarray(Image.open(io.BytesIO(b)).convert("RGB"),dtype=np.float64)
            return t,im[...,0]*256+im[...,1]+im[...,2]/256-32768
        except Exception as e: err=e
    raise err
tiles=[(x,y) for x in range(tx0,tx1+1) for y in range(ty0,ty1+1)]
mos=np.zeros(((ty1-ty0+1)*256,(tx1-tx0+1)*256))
with cf.ThreadPoolExecutor(8) as ex:
    for (x,y),h in ex.map(get,tiles): mos[(y-ty0)*256:(y-ty0+1)*256,(x-tx0)*256:(x-tx0+1)*256]=h
# remuestreo a rejilla lat/lon regular (fila 0 = norte)
lon=LO0+np.arange(NX+1)/NX*(LO1-LO0); lat=LA1-np.arange(NZ+1)/NZ*(LA1-LA0)
LON,LAT=np.meshgrid(lon,lat)
n=2**Z; PX=((LON+180)/360*n-tx0)*256; PY=((1-np.log(np.tan(np.radians(LAT))+1/np.cos(np.radians(LAT)))/math.pi)/2*n-ty0)*256
H=map_coordinates(mos,[PY-.5,PX-.5],order=1,mode="nearest")
# agua: zonas planas a baja altitud (los lagos son planos en el DEM)
rng=maximum_filter(mos,9)-minimum_filter(mos,9)
flat=map_coordinates((rng<1.2).astype(float),[PY-.5,PX-.5],order=1)>.6
water=flat&(H<1900)
# polígonos
gj=json.load(open(sys.argv[1]))
def rings(feats):
    out=[]
    for f in feats:
        g=f["geometry"]; polys=g["coordinates"] if g["type"]=="MultiPolygon" else [g["coordinates"]]
        for p in polys: out.append(p[0])
    return out
def dp(pts,eps):
    if len(pts)<3: return pts
    a,b=np.array(pts[0]),np.array(pts[-1]); d=b-a; L=np.hypot(*d) or 1e-12
    dist=[abs(d[0]*(p[1]-a[1])-d[1]*(p[0]-a[0]))/L for p in pts[1:-1]]
    i=int(np.argmax(dist)); 
    if dist[i]>eps: return dp(pts[:i+2],eps)[:-1]+dp(pts[i+1:],eps)
    return [pts[0],pts[-1]]
def simp_ring(r,eps):
    pts=[tuple(p) for p in r]; m=len(pts)//2          # anillo cerrado: se divide en dos mitades
    return dp(pts[:m+1],eps)[:-1]+dp(pts[m:],eps)[:-1]
def simp(rs,eps=.0012): return [[[round(la,4),round(lo,4)] for lo,la in simp_ring(r,eps)] for r in rs]
feats=gj["features"]; CH=rings(feats); GR=rings([f for f in feats if f["properties"]["name"]=="Graubünden"])
def raster(rs):
    im=Image.new("L",(NX+1,NZ+1),0); d=ImageDraw.Draw(im)
    for r in rs: d.polygon([((lo-LO0)/(LO1-LO0)*NX,(LA1-la)/(LA1-LA0)*NZ) for lo,la in r],fill=1)
    return np.asarray(im,dtype=np.uint8)
inCH,inGR=raster(CH),raster(GR)
v=np.clip(np.round((H+500)*8),0,65535).astype(np.uint32)
flags=(water&(inCH|True)).astype(np.uint8)*1+inCH*2+inGR*4
rgb=np.dstack([v>>8,v&255,flags]).astype(np.uint8)
buf=io.BytesIO(); Image.fromarray(rgb).save(buf,"PNG",optimize=True)
import base64
out={"NX":NX,"NZ":NZ,"LO0":LO0,"LO1":LO1,"LA0":LA0,"LA1":LA1,"png":"data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode(),
     "CH":simp(CH),"GR":simp(GR)}
json.dump(out,open("tools/dem.json","w"),separators=(",",":"))
print("png KB",len(buf.getvalue())//1024,"rings",len(out["CH"]),len(out["GR"]),"H range",H.min(),H.max(),"water %",water.mean()*100,file=sys.stderr)
np.save("/tmp/H.npy",H)
