"""
app.py
============================================================
PoleOptimizer Pro - Streamlit Giriş Noktası (Main Module)
============================================================

Bu dosya, projenin GERÇEK Streamlit uygulamasıdır ve deploy ayarlarında
"main module" olarak BU dosya seçilmelidir (corridor_data_collector.py
değil — o dosya saf bir backend modülüdür, arayüz içermez).

Şu an sadece Modül 1'i (CorridorDataCollector) interaktif olarak test
etmeye yarayan minimal bir arayüz sunuyor. Sonraki modüller (DEM/eğim
kontrolü, optimizasyon motoru) eklendikçe bu dosya genişletilecek.
"""

import streamlit as st
import folium
from streamlit_folium import st_folium

from corridor_data_collector import CorridorDataCollector, GeoPoint

st.set_page_config(page_title="PoleOptimizer Pro", layout="wide")

st.title("⚡ PoleOptimizer Pro")
st.caption("Modül 1 Test Arayüzü: Koridor Veri Toplama & Sanal Düğüm Üretimi")

with st.sidebar:
    st.header("Girdi Parametreleri")

    st.subheader("Başlangıç Noktası (A)")
    start_lat = st.number_input("A - Enlem (lat)", value=40.5506, format="%.6f")
    start_lon = st.number_input("A - Boylam (lon)", value=34.9556, format="%.6f")

    st.subheader("Bitiş Noktası (B)")
    end_lat = st.number_input("B - Enlem (lat)", value=40.5650, format="%.6f")
    end_lon = st.number_input("B - Boylam (lon)", value=34.9700, format="%.6f")

    st.subheader("Koridor Ayarları")
    buffer_m = st.slider("Koridor genişliği (m)", 50, 500, 150, step=10)
    spacing_m = st.slider("Direk aralığı / span (m)", 20, 100, 40, step=5)

    st.subheader("Direk Konumlandırma")
    offset_m = st.slider(
        "Yol kenarı offseti (m)", 0, 15, 5, step=1,
        help="Direklerin yol orta hattından kaldırım kenarına doğru ne kadar kaydırılacağı.",
    )
    offset_side = st.radio(
        "Offset yönü (A→B'ye bakarken)", options=["right", "left"],
        format_func=lambda v: "Sağ" if v == "right" else "Sol",
        horizontal=True,
    )

    run_button = st.button("🚀 Koridor Verisini Getir", type="primary")

if run_button:
    with st.spinner("OSM verisi çekiliyor ve sanal düğümler üretiliyor... (biraz sürebilir)"):
        try:
            collector = CorridorDataCollector(
                start=GeoPoint(lat=start_lat, lon=start_lon),
                end=GeoPoint(lat=end_lat, lon=end_lon),
                corridor_buffer_m=float(buffer_m),
                node_spacing_m=float(spacing_m),
                pole_offset_m=float(offset_m),
                offset_side=offset_side,
            )
            data = collector.run()
            st.session_state["corridor_data"] = data
        except Exception as exc:  # noqa: BLE001
            st.error(f"Veri toplama sırasında bir hata oluştu: {exc}")
            st.session_state.pop("corridor_data", None)

data = st.session_state.get("corridor_data")

if data is None:
    st.info("Sol panelden A/B koordinatlarını girip **'Koridor Verisini Getir'** butonuna basın.")
else:
    if data.route_source == "road_snapped":
        st.success("✅ Direkler yol ağı üzerinden hesaplanan rotaya oturtuldu (kenar offsetli).")
    else:
        st.warning(
            "⚠️ Yol ağı üzerinden rota bulunamadı — A-B düz hattına geri "
            "dönüldü. Bu durumda düğümler bina/arazi üzerinden geçebilir; "
            "sonuç yalnızca kaba bir ön izlemedir."
        )

    num_corners = sum(1 for n in data.virtual_nodes if n.is_corner)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Bina Sayısı", len(data.buildings_gdf))
    col2.metric("Engel Poligonu", len(data.obstacles_gdf))
    col3.metric("Yol Segmenti", len(data.roads_gdf))
    col4.metric("Sanal Düğüm", len(data.virtual_nodes))
    col5.metric("Köşe/Kavşak Direği", num_corners)

    mid_lat = (data.start.lat + data.end.lat) / 2
    mid_lon = (data.start.lon + data.end.lon) / 2

    fmap = folium.Map(location=[mid_lat, mid_lon], zoom_start=15, tiles="OpenStreetMap")

    folium.Marker(
        [data.start.lat, data.start.lon], tooltip="A - Başlangıç",
        icon=folium.Icon(color="green"),
    ).add_to(fmap)
    folium.Marker(
        [data.end.lat, data.end.lon], tooltip="B - Bitiş",
        icon=folium.Icon(color="red"),
    ).add_to(fmap)

    for node in data.virtual_nodes:
        is_corner = node.is_corner
        folium.CircleMarker(
            [node.point.lat, node.point.lon],
            radius=7 if is_corner else 4,
            color="#d62728" if is_corner else "#1f77b4",
            fill=True,
            fill_opacity=0.9 if is_corner else 0.8,
            tooltip=(
                f"{'🔴 Köşe/Kavşak Direği' if is_corner else 'Ara Direk'} "
                f"#{node.node_id} ({node.cumulative_distance_m:.0f}m)"
            ),
        ).add_to(fmap)

    if data.corridor_polygon is not None:
        folium.GeoJson(
            data.corridor_polygon.__geo_interface__,
            style_function=lambda _: {"color": "#ff7f0e", "fillOpacity": 0.05},
        ).add_to(fmap)

    st_folium(fmap, width=None, height=600, returned_objects=[])
