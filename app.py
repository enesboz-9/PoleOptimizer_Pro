"""
app.py
============================================================
PoleOptimizer Pro - Streamlit Giriş Noktası (Main Module)
============================================================

Bu dosya, projenin GERÇEK Streamlit uygulamasıdır ve deploy ayarlarında
"main module" olarak BU dosya seçilmelidir (corridor_data_collector.py
değil — o dosya saf bir backend modülüdür, arayüz içermez).

Modül 1'i (CorridorDataCollector) interaktif olarak test etmeye yarayan
arayüz iki giriş yöntemi sunar:
    1) Elle koordinat gir  -> A ve B noktalarını sayısal olarak girip
       en kısa yol ağı rotasını hesaplatma (eski davranış).
    2) Haritada çiz (kalem) -> Kullanıcı haritada başlangıçtan bitişe
       serbest elle (eğik/yamuk olabilen) bir kroki çizer; uygulama bu
       krokiyi gerçek OSM yol ağına harita-eşleştirme (map-matching) ile
       oturtur ve direkleri bu eşleştirilmiş rotanın kenarına yerleştirir.

Her iki modda da, sonuç haritasındaki her direğe TIKLANDIĞINDA (hover
değil, click) koordinatları bir popup içinde gösterilir.
"""

import streamlit as st
import folium
import osmnx as ox
from folium.plugins import Draw
from streamlit_folium import st_folium

from corridor_data_collector import CorridorDataCollector, GeoPoint

st.set_page_config(page_title="PoleOptimizer Pro", layout="wide")


def geocode_place(query: str):
    """İl/ilçe/köy/mahalle/adres metnini OSM Nominatim üzerinden
    (lat, lon) koordinatına çevirir.

    Türkiye'deki küçük yerleşimlerde (köy, mahalle) belirsizliği azaltmak
    için sorguya otomatik olarak ", Türkiye" eklenir (kullanıcı zaten bir
    ülke adı yazmadıysa).

    Args:
        query: Kullanıcının yazdığı yer adı (örn. "Bafra", "Görele köyü,
            Amasya").

    Returns:
        (lat, lon) tuple'ı, ya da konum bulunamazsa/hata oluşursa None.
    """
    query = query.strip()
    if not query:
        return None
    if "türkiye" not in query.lower() and "turkey" not in query.lower():
        query = f"{query}, Türkiye"
    try:
        lat, lon = ox.geocode(query)
        return lat, lon
    except Exception:  # noqa: BLE001
        return None


# Harita merkezi, arama kutusuyla güncellenebilmesi için session_state'te
# tutulur (varsayılan: Çorum/Bahçelievler civarı — mevcut örnek verilerle
# aynı bölge).
st.session_state.setdefault("map_center_lat", 40.5578)
st.session_state.setdefault("map_center_lon", 34.9628)
st.session_state.setdefault("map_zoom", 16)

st.title("⚡ PoleOptimizer Pro")
st.caption("Modül 1 Test Arayüzü: Koridor Veri Toplama & Sanal Düğüm Üretimi")

# ------------------------------------------------------------------ #
# Sidebar: ortak parametreler + giriş yöntemi seçimi
# ------------------------------------------------------------------ #
with st.sidebar:
    st.header("Girdi Parametreleri")

    st.subheader("🔍 Konum Ara")
    st.caption("İl, ilçe, köy, mahalle veya adres yazıp haritayı o konuma taşıyın.")
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        search_query = st.text_input(
            "Konum Ara",
            placeholder="Örn: Bafra, Samsun ya da Görele köyü",
            label_visibility="collapsed",
            key="location_search_query",
        )
    with search_col2:
        search_clicked = st.button("Ara", use_container_width=True)

    if search_clicked:
        if not search_query.strip():
            st.warning("Lütfen bir yer adı yazın.")
        else:
            with st.spinner(f"'{search_query}' aranıyor..."):
                result = geocode_place(search_query)
            if result is None:
                st.error(
                    "Bu konum bulunamadı. Daha spesifik yazmayı deneyin "
                    "(örn. 'Köy adı, İlçe adı, İl adı')."
                )
            else:
                st.session_state["map_center_lat"] = result[0]
                st.session_state["map_center_lon"] = result[1]
                st.session_state["map_zoom"] = 14
                st.success(f"📍 Bulundu: {result[0]:.5f}, {result[1]:.5f}")
                st.rerun()

    st.divider()

    st.subheader("Giriş Yöntemi")
    input_mode = st.radio(
        "Hat nasıl belirlenecek?",
        options=["sketch", "manual"],
        format_func=lambda v: "✏️ Haritada Çiz (Kalem)" if v == "sketch" else "🔢 Elle Koordinat Gir",
    )

    if input_mode == "manual":
        st.subheader("Başlangıç Noktası (A)")
        start_lat = st.number_input("A - Enlem (lat)", value=40.5506, format="%.6f")
        start_lon = st.number_input("A - Boylam (lon)", value=34.9556, format="%.6f")

        st.subheader("Bitiş Noktası (B)")
        end_lat = st.number_input("B - Enlem (lat)", value=40.5650, format="%.6f")
        end_lon = st.number_input("B - Boylam (lon)", value=34.9700, format="%.6f")
    else:
        st.subheader("Harita Merkezi (Çizim İçin)")
        st.caption("Yukarıdan bir yer arayabilir ya da aşağıdan elle ayarlayabilirsiniz.")
        center_lat = st.number_input("Merkez - Enlem (lat)", format="%.6f", key="map_center_lat")
        center_lon = st.number_input("Merkez - Boylam (lon)", format="%.6f", key="map_center_lon")
        zoom_level = st.slider("Yakınlaştırma", 12, 19, key="map_zoom")

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

# ------------------------------------------------------------------ #
# Ana alan: giriş yöntemine göre çizim haritası ya da doğrudan buton
# ------------------------------------------------------------------ #
sketch_points = None

if input_mode == "manual":
    with st.expander("🗺️ Aranan konumu haritada göster / koordinat bul", expanded=True):
        st.caption(
            "Sol panelden bir yer aradıysanız harita o konuma gelir. Haritada "
            "istediğiniz noktaya **tıklayarak** koordinatlarını görüp A/B "
            "alanlarına kopyalayabilirsiniz."
        )
        locator_map = folium.Map(
            location=[st.session_state["map_center_lat"], st.session_state["map_center_lon"]],
            zoom_start=st.session_state["map_zoom"],
            tiles="OpenStreetMap",
        )
        folium.Marker(
            [st.session_state["map_center_lat"], st.session_state["map_center_lon"]],
            tooltip="Aranan konum",
            icon=folium.Icon(color="blue", icon="search"),
        ).add_to(locator_map)
        locator_map.add_child(folium.LatLngPopup())
        st_folium(locator_map, width=None, height=400, returned_objects=[], key="locator_map")

if input_mode == "sketch":
    st.subheader("✏️ Kroki: Hattı Kalemle Çizin")
    st.caption(
        "Haritanın sol üstündeki **çizgi (polyline)** aracına tıklayın, "
        "başlangıç noktasından bitişe doğru sokakları kabaca takip ederek "
        "tıklaya tıklaya ilerleyin, son noktada **çift tıklayarak** çizimi "
        "bitirin. Çiziminiz eğik, yamuk ya da tam yol üzerinde olmasa bile "
        "sorun değil — uygulama sizin anlatmak istediğiniz hattı en yakın "
        "gerçek yol ağına oturtacak. Gerekirse kalem simgesinin yanındaki "
        "düzenleme/silme araçlarıyla çizimi düzeltebilirsiniz."
    )

    draw_map = folium.Map(
        location=[center_lat, center_lon], zoom_start=zoom_level, tiles="OpenStreetMap"
    )
    Draw(
        export=False,
        draw_options={
            "polyline": {"shapeOptions": {"color": "#e91e63", "weight": 4}},
            "polygon": False,
            "rectangle": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(draw_map)

    draw_result = st_folium(
        draw_map,
        key="sketch_map",
        width=None,
        height=500,
        returned_objects=["all_drawings"],
    )

    sketch_coords_lonlat = None
    if draw_result:
        drawings = draw_result.get("all_drawings") or []
        line_drawings = [
            d for d in drawings if d.get("geometry", {}).get("type") == "LineString"
        ]
        if line_drawings:
            # Kullanıcı birden fazla çizgi çizmiş olabilir; en son çizileni kullan.
            sketch_coords_lonlat = line_drawings[-1]["geometry"]["coordinates"]

    if sketch_coords_lonlat and len(sketch_coords_lonlat) >= 2:
        st.success(f"✅ Çizim algılandı: {len(sketch_coords_lonlat)} nokta. Hesaplamaya hazır.")
        sketch_points = [GeoPoint(lat=lat, lon=lon) for lon, lat in sketch_coords_lonlat]
    else:
        st.info("Henüz bir çizgi çizilmedi. Haritadan bir hat çizin.")

    run_button = st.button(
        "🚀 Bu Çizimden Direkleri Hesapla",
        type="primary",
        disabled=sketch_points is None,
    )
else:
    run_button = st.button("🚀 Koridor Verisini Getir", type="primary")

# ------------------------------------------------------------------ #
# Hesaplama
# ------------------------------------------------------------------ #
if run_button:
    with st.spinner("OSM verisi çekiliyor ve sanal düğümler üretiliyor... (biraz sürebilir)"):
        try:
            if input_mode == "sketch":
                start_pt = sketch_points[0]
                end_pt = sketch_points[-1]
                collector = CorridorDataCollector(
                    start=start_pt,
                    end=end_pt,
                    corridor_buffer_m=float(buffer_m),
                    node_spacing_m=float(spacing_m),
                    pole_offset_m=float(offset_m),
                    offset_side=offset_side,
                )
                data = collector.run(sketch_points=sketch_points)
            else:
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

# ------------------------------------------------------------------ #
# Sonuçlar
# ------------------------------------------------------------------ #
if data is None:
    if input_mode == "manual":
        st.info("Sol panelden A/B koordinatlarını girip **'Koridor Verisini Getir'** butonuna basın.")
else:
    if data.route_source == "sketch_matched":
        st.success(
            "✅ Çiziminiz gerçek yol ağına oturtuldu (map-matching) ve "
            "direkler bu rotanın kenarına offsetli olarak yerleştirildi."
        )
    elif data.route_source == "road_snapped":
        st.success("✅ Direkler yol ağı üzerinden hesaplanan rotaya oturtuldu (kenar offsetli).")
    else:
        st.warning(
            "⚠️ Yol ağı üzerinden geçerli bir rota bulunamadı — düz hatta "
            "geri dönüldü. Bu durumda düğümler bina/arazi üzerinden "
            "geçebilir; sonuç yalnızca kaba bir ön izlemedir."
        )

    num_corners = sum(1 for n in data.virtual_nodes if n.is_corner)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Bina Sayısı", len(data.buildings_gdf))
    col2.metric("Engel Poligonu", len(data.obstacles_gdf))
    col3.metric("Yol Segmenti", len(data.roads_gdf))
    col4.metric("Sanal Düğüm", len(data.virtual_nodes))
    col5.metric("Köşe/Kavşak Direği", num_corners)

    st.caption("💡 Haritadaki herhangi bir direk simgesine **tıklayarak** koordinatlarını görebilirsiniz.")

    mid_lat = (data.start.lat + data.end.lat) / 2
    mid_lon = (data.start.lon + data.end.lon) / 2

    fmap = folium.Map(location=[mid_lat, mid_lon], zoom_start=15, tiles="OpenStreetMap")

    folium.Marker(
        [data.start.lat, data.start.lon],
        tooltip="A - Başlangıç",
        popup=folium.Popup(
            f"<b>A - Başlangıç</b><br>Lat: {data.start.lat:.6f}<br>Lon: {data.start.lon:.6f}",
            max_width=250,
        ),
        icon=folium.Icon(color="green"),
    ).add_to(fmap)
    folium.Marker(
        [data.end.lat, data.end.lon],
        tooltip="B - Bitiş",
        popup=folium.Popup(
            f"<b>B - Bitiş</b><br>Lat: {data.end.lat:.6f}<br>Lon: {data.end.lon:.6f}",
            max_width=250,
        ),
        icon=folium.Icon(color="red"),
    ).add_to(fmap)

    # Kullanıcının çizdiği ham krokiyi de referans olarak (soluk, kesikli
    # çizgi) haritaya ekleyelim ki hesaplanan rotayla karşılaştırılabilsin.
    if sketch_points:
        folium.PolyLine(
            [(p.lat, p.lon) for p in sketch_points],
            color="#e91e63",
            weight=3,
            opacity=0.5,
            dash_array="6,8",
            tooltip="Sizin çiziminiz (ham kroki)",
        ).add_to(fmap)

    for node in data.virtual_nodes:
        is_corner = node.is_corner
        label = "🔴 Köşe/Kavşak Direği" if is_corner else "🔵 Ara Direk"
        popup_html = (
            f"<b>{label} #{node.node_id}</b><br>"
            f"Enlem (lat): {node.point.lat:.6f}<br>"
            f"Boylam (lon): {node.point.lon:.6f}<br>"
            f"A'dan mesafe: {node.cumulative_distance_m:.0f} m"
        )
        folium.CircleMarker(
            [node.point.lat, node.point.lon],
            radius=7 if is_corner else 4,
            color="#d62728" if is_corner else "#1f77b4",
            fill=True,
            fill_opacity=0.9 if is_corner else 0.8,
            tooltip=f"{label} #{node.node_id} ({node.cumulative_distance_m:.0f}m) — koordinat için tıklayın",
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(fmap)

    if data.corridor_polygon is not None:
        folium.GeoJson(
            data.corridor_polygon.__geo_interface__,
            style_function=lambda _: {"color": "#ff7f0e", "fillOpacity": 0.05},
        ).add_to(fmap)

    st_folium(fmap, width=None, height=600, returned_objects=[], key="result_map")
