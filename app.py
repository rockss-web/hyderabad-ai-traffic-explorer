import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import Fullscreen, MiniMap
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from pathlib import Path

st.set_page_config(page_title="Hyderabad Traffic AI", page_icon="🚦", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
.stApp {background:#0b1220;color:#edf3ff;font-family:'DM Sans',sans-serif}
section[data-testid="stSidebar"]{background:#111c2e}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important}
.hero{background:linear-gradient(120deg,#142c46,#101b2c);padding:24px;border-radius:18px;border:1px solid #29415f;margin-bottom:18px}
.hero .tag{color:#63e6c4;font-weight:700;letter-spacing:2px;font-size:11px}
.hero h1{font-size:34px;margin:7px 0}
.hero p{color:#bdcce0;margin:0}
.kpi{background:#142238;border:1px solid #2a405c;border-radius:14px;padding:15px}
.klabel{font-size:11px;letter-spacing:1px;color:#a7bad3}
.kvalue{font-size:27px;font-weight:700;color:#fff;margin-top:5px}
.knote{font-size:11px;color:#63e6c4}
div[data-testid="stMetric"]{background:#142238;border:1px solid #2a405c;padding:12px;border-radius:12px}
</style>
""", unsafe_allow_html=True)

# Demo road corridors and approximate map coordinates. These are not live road measurements.
ROADS = [
    {"road":"HITEC City Main Road","area":"HITEC City","base":30,"load":.82,"coords":[(17.4435,78.3772),(17.4470,78.3810),(17.4500,78.3865),(17.4508,78.3915)]},
    {"road":"Madhapur Road","area":"Madhapur","base":34,"load":.72,"coords":[(17.4483,78.3915),(17.4447,78.3950),(17.4395,78.3995),(17.4350,78.4045)]},
    {"road":"Gachibowli Link Road","area":"Gachibowli","base":40,"load":.63,"coords":[(17.4401,78.3489),(17.4390,78.3590),(17.4405,78.3695),(17.4435,78.3772)]},
    {"road":"Jubilee Hills Road No. 36","area":"Jubilee Hills","base":36,"load":.59,"coords":[(17.4310,78.4070),(17.4350,78.4120),(17.4400,78.4170),(17.4445,78.4210)]},
    {"road":"Kukatpally Main Road","area":"Kukatpally","base":27,"load":.84,"coords":[(17.4948,78.3996),(17.4890,78.4050),(17.4820,78.4100),(17.4750,78.4140)]},
    {"road":"Begumpet Main Road","area":"Begumpet","base":29,"load":.78,"coords":[(17.4440,78.4670),(17.4490,78.4630),(17.4540,78.4590),(17.4590,78.4550)]},
    {"road":"Tank Bund corridor","area":"Tank Bund","base":31,"load":.70,"coords":[(17.4235,78.4740),(17.4300,78.4750),(17.4380,78.4770),(17.4450,78.4790)]},
    {"road":"Financial District corridor","area":"Financial District","base":39,"load":.57,"coords":[(17.4150,78.3400),(17.4210,78.3450),(17.4300,78.3470),(17.4401,78.3489)]},
]

@st.cache_resource
def train_ai():
    rng=np.random.default_rng(7)
    n=5000
    hour=rng.integers(0,24,n)
    rain=rng.uniform(0,10,n)
    load=rng.uniform(.35,.95,n)
    incident=rng.integers(0,3,n)
    weekend=rng.integers(0,2,n)
    rush=(((hour>=7)&(hour<=10))|((hour>=16)&(hour<=20))).astype(int)
    baseline=rng.choice([25,28,30,34,36,40],n)
    speed=baseline*(1-.22*load-.28*rush+.06*weekend)-.85*rain-incident*2+rng.normal(0,2.5,n)
    speed=np.clip(speed,4,baseline)
    X=pd.DataFrame({"hour":hour,"rain":rain,"load":load,"incident":incident,"weekend":weekend,"rush":rush,"baseline":baseline})
    Xtr,Xte,ytr,yte=train_test_split(X,speed,test_size=.2,random_state=7)
    m=RandomForestRegressor(n_estimators=100,max_depth=12,min_samples_leaf=2,random_state=7,n_jobs=-1)
    m.fit(Xtr,ytr)
    mae=mean_absolute_error(yte,m.predict(Xte))
    return m,float(mae)

model,mae=train_ai()
with st.sidebar:
    st.markdown("## 🚦 Traffic controls")
    st.caption("Change the scenario and watch predicted road colours update.")
    hour=st.slider("Time of day",0,23,18,format="%d:00")
    rain=st.slider("Rain intensity",0,10,2)
    incident=st.selectbox("Incident scenario",["No incident","Minor incident","Major incident"])
    weekend=st.toggle("Weekend",value=False)
    st.divider()
    st.warning("DEMO MODE: AI predictions use synthetic training data, not live traffic feeds.")

st.markdown("""
<div class="hero">
<div class="tag">SMART CITY · HYDERABAD</div>
<h1>AI Traffic Explorer</h1>
<p>Explore a Hyderabad map with AI-estimated traffic conditions. Adjust the scenario, inspect coloured road corridors, and compare predicted speeds.</p>
</div>
""",unsafe_allow_html=True)

inc_val={"No incident":0,"Minor incident":1,"Major incident":2}[incident]
rush=int((7<=hour<=10) or (16<=hour<=20))
results=[]
for r in ROADS:
    x=pd.DataFrame([{"hour":hour,"rain":rain,"load":r["load"],"incident":inc_val,"weekend":int(weekend),"rush":rush,"baseline":r["base"]}])
    speed=float(np.clip(model.predict(x)[0],4,r["base"]))
    congestion=int(np.clip(round(100*(1-speed/r["base"])),0,100))
    level="Light" if congestion<30 else ("Moderate" if congestion<60 else "Heavy")
    color={"Light":"#20c997","Moderate":"#ffb020","Heavy":"#ff4d67"}[level]
    results.append({**r,"speed":round(speed,1),"congestion":congestion,"level":level,"color":color})
df=pd.DataFrame(results)
cols=st.columns(4)
kpis=[
("ROAD CORRIDORS",str(len(df)),"Sample map segments"),
("AVERAGE PREDICTED SPEED",f'{df.speed.mean():.1f} km/h',"AI estimate"),
("AVERAGE CONGESTION",f'{df.congestion.mean():.0f}%',"Scenario score"),
("HEAVY TRAFFIC",str((df.level=="Heavy").sum()),"Segments in this scenario")]
for c,(label,value,note) in zip(cols,kpis):
    c.markdown(f'<div class="kpi"><div class="klabel">{label}</div><div class="kvalue">{value}</div><div class="knote">{note}</div></div>',unsafe_allow_html=True)

st.write("")
left,right=st.columns([2.1,1])
with left:
    st.subheader("📍 Hyderabad traffic map")
    st.caption("Pan and zoom. Click a coloured road line for its prediction. Map tiles need internet access.")
    traffic_map=folium.Map(location=[17.4435,78.4050],zoom_start=12,tiles="CartoDB positron",control_scale=True)
    Fullscreen(position="topleft").add_to(traffic_map)
    MiniMap(toggle_display=True).add_to(traffic_map)
    for r in results:
        popup=f"<b>{r['road']}</b><br>Area: {r['area']}<br>Predicted speed: {r['speed']} km/h<br>Congestion: {r['congestion']}%<br>Status: {r['level']}<br><small>AI demo estimate, not live data</small>"
        folium.PolyLine(r["coords"],color=r["color"],weight=8,opacity=.9,tooltip=f"{r['road']} · {r['level']} traffic",popup=folium.Popup(popup,max_width=250)).add_to(traffic_map)
        folium.CircleMarker(r["coords"][len(r["coords"])//2],radius=5,color=r["color"],fill=True,fill_color=r["color"],fill_opacity=1,tooltip=r["area"]).add_to(traffic_map)
    Fullscreen().add_to(traffic_map)
    st_folium(traffic_map,use_container_width=True,height=570,returned_objects=[],key="hyd_map")
    st.markdown("🟢 **Light** &nbsp;&nbsp; 🟠 **Moderate** &nbsp;&nbsp; 🔴 **Heavy**",unsafe_allow_html=True)
with right:
    st.subheader("🚗 Road conditions")
    for r in sorted(results,key=lambda z:z["congestion"],reverse=True):
        with st.container(border=True):
            st.markdown(f"**{r['road']}**")
            st.markdown(f"<span style='color:{r['color']};font-weight:700'>● {r['level']} traffic</span>",unsafe_allow_html=True)
            st.metric("Predicted speed",f"{r['speed']} km/h",f"{r['congestion']}% congestion")
st.subheader("Congestion comparison")
chart=df[["road","congestion","level"]].sort_values("congestion")
import plotly.express as px
fig=px.bar(chart,x="congestion",y="road",orientation="h",color="level",text="congestion",
 color_discrete_map={"Light":"#20c997","Moderate":"#ffb020","Heavy":"#ff4d67"})
fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=350,
 margin=dict(l=5,r=15,t=10,b=10),xaxis_title="Congestion score (%)",yaxis_title="",legend_title_text="")
fig.update_traces(textposition="outside",cliponaxis=False)
st.plotly_chart(fig,use_container_width=True)
with st.expander("About the AI and its limits"):
    st.write("A Random Forest regression model estimates road speed from time, rain, road load, incident severity, weekend and rush-hour inputs.")
    st.write(f"Demo holdout MAE: {mae:.2f} km/h. This score is based on synthetic data and is not evidence of real-world accuracy.")
    st.write("Road lines are illustrative corridors around Hyderabad. The map does not use Google Maps live traffic, verified live road speeds, GPS or a live routing service.")
st.caption("AI Traffic Explorer · Student prototype · SDG 11: Sustainable Cities and Communities")
