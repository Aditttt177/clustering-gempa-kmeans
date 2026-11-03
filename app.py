import streamlit as st, pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Clustering Gempa K-Means", layout="wide")
FITUR = ["depth", "mag", "sig", "nst", "dmin", "rms", "gap"]
df = pd.read_csv("data_dashboard.csv")

def siapkan(fitur):
    X = df[fitur].copy()
    for c in ["depth", "dmin"]:
        if c in X:
            X[c] = np.log1p(X[c])
    return pd.DataFrame(StandardScaler().fit_transform(X), columns=fitur)

@st.cache_data
def evaluasi(fitur):
    Z = siapkan(list(fitur)); out = []
    for k in range(2, 11):
        km = KMeans(k, n_init=10, random_state=42).fit(Z)
        out.append((k, km.inertia_, silhouette_score(Z, km.labels_, sample_size=3000, random_state=42)))
    return pd.DataFrame(out, columns=["K", "Inertia", "Silhouette"])

@st.cache_data
def klaster(fitur, k):
    Z = siapkan(list(fitur))
    km = KMeans(k, n_init=10, random_state=42).fit(Z)
    sil = silhouette_score(Z, km.labels_, sample_size=3000, random_state=42)
    return km.labels_, km.inertia_, sil, Z

# ---------- Sidebar (opsi interaktif) ----------
st.sidebar.header("Pengaturan")
K = st.sidebar.slider("Jumlah klaster (K)", 2, 8, 4)
fitur_pakai = st.sidebar.multiselect("Fitur untuk clustering", FITUR, default=FITUR)
if len(fitur_pakai) < 2:
    st.warning("Pilih minimal 2 fitur untuk clustering."); st.stop()
fx = st.sidebar.selectbox("Fitur sumbu X (visualisasi)", fitur_pakai, index=0)
fy = st.sidebar.selectbox("Fitur sumbu Y (visualisasi)", fitur_pakai, index=min(1, len(fitur_pakai) - 1))

fp = tuple(fitur_pakai)
lab, inertia, sil, Z = klaster(fp, K)
d = df.copy(); d["cluster"] = lab

# ---------- Judul dan ringkasan ----------
st.title("Analisis Clustering Gempa Bumi Global dengan K-Means")
st.write("Dashboard ini mengelompokkan gempa bumi (Global Earthquakes Seismic Telemetry, Kaggle) "
         "berdasarkan kedalaman, magnitudo, signifikansi, dan kualitas rekaman seismik.")
a, b, c, e = st.columns(4)
a.metric("Jumlah data", len(df)); b.metric("Jumlah fitur", len(fitur_pakai))
c.metric("Inertia", f"{inertia:,.0f}"); e.metric("Silhouette", f"{sil:.3f}")
st.dataframe(df[FITUR].describe().T.round(2))
st.caption("Data setelah pembersihan: 9296 gempa. Outlier sig, nst, dmin, rms, gap dibuang dengan batas 3x IQR; "
           "depth dan mag dipertahankan karena gempa dalam dan besar adalah kejadian nyata.")

# ---------- EDA ----------
st.header("Distribusi Fitur dan Korelasi")
f1, f2 = st.columns(2)
fig, ax = plt.subplots(figsize=(6, 4)); sns.histplot(df[fx], bins=40, ax=ax, color="steelblue")
ax.set_title(f"Distribusi {fx}"); f1.pyplot(fig)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(df[fitur_pakai].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Korelasi antar fitur"); f2.pyplot(fig)
st.caption("Sebagian besar fitur miring ke kanan (gempa kecil dan dangkal mendominasi). mag dan sig berkorelasi kuat (0,80) "
           "karena sig dihitung sebagian dari magnitudo; gap berkorelasi negatif dengan mag dan nst; "
           "depth, dmin, dan rms hampir tidak berkorelasi dengan fitur lain.")

# ---------- Elbow dan Silhouette ----------
st.header("Elbow Method dan Silhouette Score")
ev = evaluasi(fp)
g1, g2 = st.columns(2)
fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(ev.K, ev.Inertia, "o-"); ax.axvline(K, color="red", ls="--")
ax.set_title("Elbow (Inertia)"); ax.set_xlabel("K"); g1.pyplot(fig)
fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(ev.K, ev.Silhouette, "o-", color="darkorange"); ax.axvline(K, color="red", ls="--")
ax.set_title("Silhouette Score"); ax.set_xlabel("K"); g2.pyplot(fig)
st.dataframe(ev.round(4))
st.caption("Inertia turun mulus tanpa siku tegas. Silhouette tertinggi di K=2 lalu turun dan mendatar di sekitar 0,16-0,18 untuk K>=3, "
           "artinya struktur klaster lemah (data lebih berupa kontinum). Garis merah menandai K yang dipilih; "
           "K=4 dipakai sebagai default karena klasternya paling mudah ditafsirkan.")

# ---------- Hasil klaster ----------
st.header("Visualisasi Hasil Klaster")
h1, h2 = st.columns(2)
P = PCA(n_components=2, random_state=42).fit_transform(Z)
fig, ax = plt.subplots(figsize=(6, 5)); sns.scatterplot(x=P[:, 0], y=P[:, 1], hue=lab, palette="Set1", s=10, alpha=0.7, ax=ax)
ax.set_title("Scatter PCA 2D"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); h1.pyplot(fig)
fig, ax = plt.subplots(figsize=(6, 5)); sns.scatterplot(data=d, x=fx, y=fy, hue="cluster", palette="Set1", s=10, alpha=0.7, ax=ax)
ax.set_title(f"{fx} vs {fy}"); h2.pyplot(fig)
st.caption("Pada K=4, klaster gempa kuat (mag > 5) terpisah di sisi kanan PCA dan klaster gempa dalam di bagian atas, "
           "sedangkan dua klaster gempa dangkal kecil saling bertumpuk karena hanya berbeda pada rms dan dmin. "
           "Di plot kedalaman vs magnitudo, gempa di atas 300 km hanya berasal dari klaster gempa dalam.")

# ---------- Karakteristik klaster ----------
st.header("Karakteristik Tiap Klaster")
tab = d.groupby("cluster")[fitur_pakai].mean().round(2)
tab.insert(0, "jumlah", d.cluster.value_counts().sort_index())
st.dataframe(tab)
i1, i2 = st.columns(2)
fig, ax = plt.subplots(figsize=(6, 4)); sns.heatmap(Z.assign(cluster=lab).groupby("cluster").mean(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Profil klaster (nilai standar)"); i1.pyplot(fig)
fig, ax = plt.subplots(figsize=(6, 4)); sns.countplot(x=lab, hue=lab, palette="Set1", legend=False, ax=ax)
ax.set_title("Jumlah gempa per klaster"); i2.pyplot(fig)
st.caption("Pada K=4: klaster 0 = gempa kuat (mag 5,26, sig 429, nst 112, gap kecil); klaster 1 = dangkal kecil dengan stasiun terdekat jauh (dmin 4,3, rms rendah); "
           "klaster 2 = dangkal kecil dengan residual tinggi (rms 0,99); klaster 3 = gempa dalam (rata-rata 163 km). "
           "Jumlah tiap klaster cukup seimbang (20-31%).")
st.header("Hubungan dengan Label Bawaan (interpretasi)")
st.dataframe(pd.crosstab(d.cluster, d.focal_depth_regime))
st.caption("Label focal_depth_regime tidak dipakai saat clustering. Klaster gempa dalam memuat sebagian besar gempa deep_mantle_slab, "
           "yang menunjukkan klaster yang terbentuk sesuai dengan klasifikasi kedalaman geofisika.")
