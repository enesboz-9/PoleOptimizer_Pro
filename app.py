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
       serbest elle (eğik/yamuk olabilen) bir kroki çizer; uygulama
       varsayılan olarak bu krokiyi gerçek OSM yol ağına harita-
       eşleştirme (map-matching) ile oturtur ve direkleri bu eşleştirilmiş
       rotanın kenarına yerleştirir. Çizilen güzergahta OSM'de hiç yol/
       patika verisi yoksa (orman, arazi, mesire alanı vb.), kullanıcı
       "Çizdiğim hattı olduğu gibi kullan" seçeneğini işaretleyerek
       map-matching'i tamamen devre dışı bırakabilir; bu durumda direkler
       doğrudan çizilen hat üzerine yerleştirilir.

Her iki modda da, sonuç haritasındaki her direğe TIKLANDIĞINDA (hover
değil, click) koordinatları ve bir önceki direğe olan mesafesi (span) bir
popup içinde gösterilir. Ayrıca "Direkler arası mesafeyi göster"
seçeneği işaretlenirse, her direk çifti arasındaki mesafe haritada
segment üzerinde etiket olarak da görüntülenir.
"""

import streamlit as st
import folium
import osmnx as ox
from branca.element import MacroElement, Template
from folium.plugins import Draw
from streamlit_folium import st_folium

from corridor_data_collector import CorridorDataCollector, GeoPoint, haversine_distance_m

st.set_page_config(page_title="PoleOptimizer Pro", layout="wide")


class SketchLineTools(MacroElement):
    """Kroki haritasına 3 ekstra araç ekler (saf Leaflet/JS, sunucuya gitmez):

    1) 🎯 Çizime Odaklan  -> çizilmiş tüm çizgilerin sınırlarına otomatik
       zoom/pan yapar (harita köşesindeki buton).
    2) ✂️ Çizgiyi Kırp    -> araç aktifken haritaya tıklandığında, tıklanan
       noktaya en yakın çizginin, tıklanan noktaya en yakın UCU o noktaya
       kadar kısaltılır (trim).
    3) Otomatik uç birleştirme -> yeni çizilen bir çizginin ucu, mevcut bir
       çizginin ucuna (piksel bazlı bir eşiğin içinde) denk gelirse iki
       çizgi otomatik olarak TEK bir çizgide birleştirilir (birden fazla
       aday uç varsa piksel mesafesine göre EN YAKIN olan seçilir). Bu,
       kapalı bir döngü (örn. bir bina adasının/apartman bloğunun
       çevresini dolaşıp başlangıcına yakın biten bir döngü besleme hattı)
       oluşturacak birleşmeler için de geçerlidir — böyle güzergahlar
       gerçek ve yaygın bir kullanım biçimi olduğundan artık normal şekilde
       birleştiriliyor. Çizim sırasında imleç böyle bir uca yaklaşınca
       yeşil bir halka ile "🔗 Birleştirilecek" ipucu gösterilir.
    """

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        (function() {
            var map = {{ this._parent.get_name() }};
            var fg = {{ this.feature_group_var }};
            var SNAP_PX = {{ this.snap_px }};
            var TRIM_PX = {{ this.trim_px }};

            // ---------- 1) Otomatik odaklama butonu ----------
            var focusCtrl = L.control({position: 'topright'});
            focusCtrl.onAdd = function() {
                var div = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
                div.innerHTML = '<a href="#" title="Çizime odaklan" ' +
                    'style="font-size:18px;line-height:30px;text-align:center;' +
                    'width:30px;height:30px;display:block;background:#fff;">🎯</a>';
                L.DomEvent.disableClickPropagation(div);
                L.DomEvent.on(div, 'click', function(e) {
                    L.DomEvent.preventDefault(e);
                    if (fg.getLayers().length > 0) {
                        map.fitBounds(fg.getBounds(), {padding: [40, 40]});
                    }
                });
                return div;
            };
            focusCtrl.addTo(map);

            // ---------- 2) Çizgi kırpma (trim) butonu ----------
            var trimActive = false;
            var trimCtrl = L.control({position: 'topleft'});
            var trimLink;
            trimCtrl.onAdd = function() {
                var div = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
                trimLink = L.DomUtil.create('a', '', div);
                trimLink.href = '#';
                trimLink.title = 'Çizgiyi kırp (tıklanan noktaya en yakın ucu kısaltır)';
                trimLink.innerHTML = '✂️';
                trimLink.style.cssText = 'font-size:16px;line-height:30px;' +
                    'text-align:center;width:30px;height:30px;display:block;background:#fff;';
                L.DomEvent.disableClickPropagation(div);
                L.DomEvent.on(trimLink, 'click', function(e) {
                    L.DomEvent.preventDefault(e);
                    trimActive = !trimActive;
                    trimLink.style.background = trimActive ? '#ffca28' : '#fff';
                    map.getContainer().style.cursor = trimActive ? 'crosshair' : '';
                });
                return div;
            };
            trimCtrl.addTo(map);

            function toPx(latlng) { return map.latLngToLayerPoint(latlng); }

            function nearestOnLine(latlngs, clickLatLng) {
                var p = toPx(clickLatLng);
                var pts = latlngs.map(toPx);
                var segLens = [];
                var totalLen = 0;
                for (var i = 0; i < pts.length - 1; i++) {
                    var segLen = pts[i].distanceTo(pts[i + 1]);
                    segLens.push(segLen);
                    totalLen += segLen;
                }
                var best = null, running = 0;
                for (var i = 0; i < pts.length - 1; i++) {
                    var a = pts[i], b = pts[i + 1];
                    var l2 = (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y);
                    var t = 0;
                    if (l2 > 0) {
                        t = ((p.x - a.x) * (b.x - a.x) + (p.y - a.y) * (b.y - a.y)) / l2;
                        t = Math.max(0, Math.min(1, t));
                    }
                    var proj = {x: a.x + t * (b.x - a.x), y: a.y + t * (b.y - a.y)};
                    var dx = p.x - proj.x, dy = p.y - proj.y;
                    var distSq = dx * dx + dy * dy;
                    if (best === null || distSq < best.distSq) {
                        best = {distSq: distSq, segIndex: i, t: t, cumDist: running + t * segLens[i]};
                    }
                    running += segLens[i];
                }
                if (best === null) return null;
                best.distPx = Math.sqrt(best.distSq);
                best.totalLen = totalLen;
                return best;
            }

            map.on('click', function(e) {
                if (!trimActive) return;
                var bestLayer = null, bestInfo = null;
                fg.eachLayer(function(layer) {
                    if (!(layer instanceof L.Polyline) || layer instanceof L.Polygon) return;
                    var latlngs = layer.getLatLngs();
                    if (!Array.isArray(latlngs) || latlngs.length < 2 || Array.isArray(latlngs[0])) return;
                    var info = nearestOnLine(latlngs, e.latlng);
                    if (info && info.distPx < TRIM_PX && (!bestInfo || info.distPx < bestInfo.distPx)) {
                        bestLayer = layer; bestInfo = info;
                    }
                });
                if (!bestLayer) return;
                var latlngs = bestLayer.getLatLngs();
                var i = bestInfo.segIndex, t = bestInfo.t;
                var a = latlngs[i], b = latlngs[i + 1];
                var cutPoint = L.latLng(a.lat + (b.lat - a.lat) * t, a.lng + (b.lng - a.lng) * t);
                var newLatLngs;
                if (bestInfo.cumDist < bestInfo.totalLen / 2) {
                    newLatLngs = [cutPoint].concat(latlngs.slice(i + 1));
                } else {
                    newLatLngs = latlngs.slice(0, i + 1).concat([cutPoint]);
                }
                if (newLatLngs.length >= 2) {
                    bestLayer.setLatLngs(newLatLngs);
                }
            });

            // ---------- 3) Çizim sırasında yakın uca "birleştirilecek" ipucu ----------
            var snapMarker = null;
            function clearSnap() {
                if (snapMarker) { map.removeLayer(snapMarker); snapMarker = null; }
            }
            function onDrawMouseMove(e) {
                var cursorPx = toPx(e.latlng);
                var found = null;
                fg.eachLayer(function(layer) {
                    if (!(layer instanceof L.Polyline) || layer instanceof L.Polygon) return;
                    var latlngs = layer.getLatLngs();
                    if (!Array.isArray(latlngs) || latlngs.length < 2 || Array.isArray(latlngs[0])) return;
                    [latlngs[0], latlngs[latlngs.length - 1]].forEach(function(endpoint) {
                        var d = cursorPx.distanceTo(toPx(endpoint));
                        if (d < SNAP_PX && (!found || d < found.dist)) {
                            found = {latlng: endpoint, dist: d};
                        }
                    });
                });
                if (found) {
                    if (!snapMarker) {
                        snapMarker = L.circleMarker(found.latlng, {
                            radius: 9, color: '#43a047', weight: 3, fill: false
                        }).addTo(map);
                        snapMarker.bindTooltip('🔗 Birleştirilecek', {
                            permanent: true, direction: 'top', offset: [0, -8]
                        }).openTooltip();
                    } else {
                        snapMarker.setLatLng(found.latlng);
                    }
                } else {
                    clearSnap();
                }
            }
            map.on('draw:drawstart', function(e) {
                if (e.layerType === 'polyline') map.on('mousemove', onDrawMouseMove);
            });
            map.on('draw:drawstop', function() {
                map.off('mousemove', onDrawMouseMove);
                clearSnap();
            });

            // ---------- 4) Uç uca yakın gelen iki çizgiyi otomatik birleştir ----------
            //
            // NOT (hata düzeltmesi): Bu blok önceden iki soruna yol açıyordu:
            //  a) fg.eachLayer() içinde koşulu sağlayan İLK çizgiyle
            //     birleşiyordu (en YAKIN olanla değil) — birden fazla aday
            //     çizgi/uç varken (örn. bir bina adasının etrafını dolaşan
            //     bir kroki, kendi başlangıç noktasına yakınlaşınca) yanlış
            //     uca birleşip hattı bir döngüye/dikdörtgene kilitleyebiliyordu.
            //  b) Tüketilen `newLayer`, `setTimeout(..., 0)` ile GECİKMELİ
            //     siliniyordu. streamlit-folium'un kendi 'draw:created'
            //     dinleyicisi (bizimkinden SONRA, ama AYNI olay turunda,
            //     senkron çalışır) `window.drawnItems.toGeoJSON()` ile TÜM
            //     katmanları o an serileştirir — silme henüz gerçekleşmediği
            //     için hem birleştirilmiş asıl çizgi HEM DE silinmeyi
            //     bekleyen küçük parça birlikte Python'a "all_drawings"
            //     olarak gönderiliyordu; bu da "otomatik algıla" adımının
            //     haritada görünenle tutarsız/hayalet bir geometri seçmesine
            //     yol açabiliyordu. Çözüm: silmeyi SENKRON (aynı tık
            //     içinde) yapmak.
            map.on('draw:created', function(e) {
                if (e.layerType !== 'polyline') return;
                clearSnap();
                var newLayer = e.layer;
                var newLatLngs = newLayer.getLatLngs();
                if (!Array.isArray(newLatLngs) || newLatLngs.length < 2) return;
                var newStart = newLatLngs[0], newEnd = newLatLngs[newLatLngs.length - 1];

                // Tüm mevcut çizgilerin HER iki ucu ile yeni çizginin HER iki
                // ucu arasındaki 4 olası eşleşmeyi piksel mesafesine göre
                // karşılaştırıp GLOBAL olarak en yakın eşleşmeyi seç.
                var bestMatch = null; // {layer, combined, dist}
                fg.eachLayer(function(existing) {
                    if (existing === newLayer) return;
                    if (!(existing instanceof L.Polyline) || existing instanceof L.Polygon) return;
                    var exLatLngs = existing.getLatLngs();
                    if (!Array.isArray(exLatLngs) || exLatLngs.length < 2 || Array.isArray(exLatLngs[0])) return;
                    var exStart = exLatLngs[0], exEnd = exLatLngs[exLatLngs.length - 1];

                    var candidates = [
                        {a: exEnd, b: newStart, combined: exLatLngs.concat(newLatLngs.slice(1))},
                        {a: exStart, b: newEnd, combined: newLatLngs.concat(exLatLngs.slice(1))},
                        {a: exEnd, b: newEnd, combined: exLatLngs.concat(newLatLngs.slice().reverse().slice(1))},
                        {a: exStart, b: newStart, combined: exLatLngs.slice().reverse().concat(newLatLngs.slice(1))},
                    ];
                    candidates.forEach(function(c) {
                        var dist = toPx(c.a).distanceTo(toPx(c.b));
                        if (dist < SNAP_PX && (!bestMatch || dist < bestMatch.dist)) {
                            bestMatch = {layer: existing, combined: c.combined, dist: dist};
                        }
                    });
                });

                if (!bestMatch) return;

                // NOT (düzeltme): Önceden burada, birleşme sonucu ortaya
                // çıkacak hattın İKİ UCU da birbirine çok yakınsa (kroki
                // kendi üzerine kapanıyorsa) birleştirme SESSİZCE iptal
                // ediliyordu. Bu, bir bina adasının/apartman bloğunun
                // çevresini dolaşan (başlangıca yakın biten) GERÇEK ve
                // yaygın bir güzergah türünü (döngü/loop besleme hattı)
                // kırıyordu: parçalar ayrı kalıyor, Python tarafı da yalnızca
                // "en uzun" parçayı kullanıp diğer parçalardaki tüm noktaları
                // sessizce atıyordu. Kapalı döngüler artık normal şekilde
                // birleştiriliyor; kullanıcı yine de istemeden döngü
                // oluşturursa "✂️ Çizgiyi Kırp" aracıyla elle düzeltebilir.
                var combined = bestMatch.combined;

                bestMatch.layer.setLatLngs(combined);
                // Senkron silme: streamlit-folium'un aynı olay turunda
                // çalışan 'draw:created' dinleyicisi, bu satır çalıştıktan
                // SONRA devreye girer (script kayıt sırası nedeniyle) — bu
                // yüzden setTimeout'a gerek yok, gecikme yalnızca hayalet
                // (silinmeyi bekleyen) katmanın Python tarafına sızmasına
                // neden oluyordu.
                fg.removeLayer(newLayer);
            });
        })();
        {% endmacro %}
        """
    )

    def __init__(self, feature_group_var, snap_px=20, trim_px=18):
        super().__init__()
        self._name = "SketchLineTools"
        # Raw JS variable name (string), e.g. "drawnItems_<draw_ctrl.get_name()>" —
        # NOT a folium object. The pinned folium/Draw version creates its own
        # internal FeatureGroup and doesn't expose a `feature_group` constructor
        # argument, so we reference that internal variable by name instead.
        self.feature_group_var = feature_group_var
        self.snap_px = snap_px
        self.trim_px = trim_px


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

    voltage_class = st.selectbox(
        "Gerilim sınıfı",
        options=list(CorridorDataCollector.VOLTAGE_SPAN_LIMITS_M.keys()),
        format_func=lambda v: "AG (0.4 kV)" if v == "AG" else "OG (34.5 kV)",
        help=(
            "Seçilen sınıfa göre azami açıklık (span) sınırlanır. Bu "
            "değerler tipik saha pratiğine dayanan YAKLAŞIK üst "
            "sınırlardır; TEDAŞ'ın güncel projelendirme şartnamesi ve "
            "onaylı statik hesap ile teyit edilmelidir."
        ),
    )
    max_span_for_class = CorridorDataCollector.VOLTAGE_SPAN_LIMITS_M[voltage_class]
    spacing_m = st.slider(
        "Direk aralığı / span (m)",
        20, int(max_span_for_class), min(40, int(max_span_for_class)), step=5,
        help=f"{voltage_class} sınıfı için yaklaşık azami açıklık: {max_span_for_class:.0f} m.",
    )

    st.subheader("Direk Konumlandırma")
    offset_m = st.slider(
        "Yol kenarı offseti (m)", 0, 15, 5, step=1,
        help="Direklerin yol orta hattından kaldırım kenarına doğru ne kadar kaydırılacağı.",
    )
    offset_side = st.radio(
        "Offset yönü", options=["auto", "right", "left"],
        format_func=lambda v: {
            "auto": "🧭 Otomatik (çizdiğim tarafa göre)",
            "right": "Sağ (A→B'ye bakarken)",
            "left": "Sol (A→B'ye bakarken)",
        }[v],
        horizontal=True,
        help=(
            "Otomatik: direkler, sizin haritada ÇİZDİĞİNİZ krokinin yolun "
            "hangi tarafında olduğunu tespit edip o tarafa yerleştirilir "
            "(yalnızca ✏️ kroki modunda anlamlıdır). Elle koordinat "
            "modunda ya da kroki yoksa 'Sağ' varsayılanına düşülür."
        ),
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
        "düzenleme/silme araçlarıyla çizimi düzeltebilirsiniz.\n\n"
        "**Yeni araçlar:** ✂️ (sol üst) çizgi kırpma aracını açar — aktifken "
        "haritaya tıkladığınızda en yakın çizginin, tıkladığınız noktaya en "
        "yakın ucu o noktaya kadar kısaltılır. Yeni bir çizgiyi mevcut bir "
        "çizginin ucuna yakın bitirirseniz (imleç yeşil 🔗 halkayla bunu "
        "gösterir), iki çizgi otomatik olarak tek bir hatta birleştirilir "
        "(birden fazla aday uç varsa EN YAKIN olan seçilir). Birleştirme, "
        "hattı kapalı bir döngüye (başlangıcın kendi ucuna kapanmasına) "
        "çevirecekse GÜVENLİK NEDENİYLE yapılmaz — çizgiler ayrı kalır, "
        "gerekirse elle düzenleyin. 🎯 (sağ üst) ise çizdiğiniz tüm hatta "
        "otomatik olarak odaklanır/zum yapar."
    )

    use_direct_line = st.checkbox(
        "📍 Çizdiğim hattı olduğu gibi kullan (yol ağına oturtma yapma)",
        value=False,
        help=(
            "İşaretlenirse map-matching TAMAMEN devre dışı kalır ve "
            "direkler doğrudan sizin çizdiğiniz hat üzerine yerleştirilir. "
            "OSM'de yol/patika verisi OLMAYAN orman, arazi, mesire alanı "
            "gibi güzergahlarda kullanın — bu tür bölgelerde uygulama "
            "krokiyi en yakın (uzaktaki) dış yola oturtmaya çalışıp hattı "
            "anlamsızlaştırabiliyor. Yol/patika verisi olan güzergahlarda "
            "bu kutuyu işaretsiz bırakmanız (varsayılan) daha gerçekçi "
            "sonuç verir."
        ),
    )
    if use_direct_line:
        st.info(
            "ℹ️ Doğrudan hat modu aktif: direkler, yol ağına bakılmaksızın "
            "tam olarak çizdiğiniz hat üzerine (offset ayarınız varsa ona "
            "göre kaydırılarak) yerleştirilecek."
        )
        if offset_side == "auto":
            st.warning(
                "⚠️ Bu modda rota = çizdiğiniz hattın kendisi olduğundan, "
                "'Otomatik' offset yönü hangi tarafı kastettiğinizi tespit "
                "edemez ve sessizce 'Sağ (A→B'ye bakarken)' varsayılanına "
                "düşer. Direkler yanlış tarafa yerleşirse, soldaki panelden "
                "'Offset yönü' seçeneğini elle 'Sol' ya da 'Sağ' olarak "
                "değiştirip tekrar üretin."
            )

    draw_map = folium.Map(
        location=[center_lat, center_lon], zoom_start=zoom_level, tiles="OpenStreetMap"
    )
    draw_ctrl = Draw(
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
    )
    draw_ctrl.add_to(draw_map)
    # Draw plugin, doğrudan bir FeatureGroup nesnesi almıyor (bu folium/Draw
    # sürümünde `feature_group` parametresi yok); kendi içinde
    # `drawnItems_<draw_ctrl_adı>` isimli bir JS değişkeni oluşturuyor.
    # SketchLineTools'a da bu aynı JS değişkeninin adını veriyoruz ki aynı
    # katman üzerinde çalışsın.
    SketchLineTools(
        feature_group_var=f"drawnItems_{draw_ctrl.get_name()}"
    ).add_to(draw_map)

    draw_result = st_folium(
        draw_map,
        key="sketch_map",
        width=None,
        height=500,
        returned_objects=["all_drawings"],
    )

    def _line_length_m(coords_lonlat):
        total = 0.0
        for (lon1, lat1), (lon2, lat2) in zip(coords_lonlat, coords_lonlat[1:]):
            total += haversine_distance_m(
                GeoPoint(lat=lat1, lon=lon1), GeoPoint(lat=lat2, lon=lon2)
            )
        return total

    # Uç noktaya olan mesafe piksel yerine metre bazlı ölçülüyor (harita
    # zoom seviyesinden bağımsız, tarayıcı JS tarafındaki SNAP_PX'ten ayrı
    # bir güvence katmanı).
    CHAIN_MERGE_SNAP_M = 30.0

    def _split_into_chains(line_drawings):
        """Haritada ayrı ayrı çizilmiş çizgi parçalarını, birbirine en
        yakın uçlarından zincirleyerek BAĞLANTILI GRUPLARA (bileşenlere)
        ayırır. Uçları birbirine yakın olan parçalar tek bir zincirde
        birleşir; birbirinden uzak duran (ör. aynı yolun İKİ AYRI
        TARAFI/PARALEL iki sokak gibi kasıtlı olarak ayrık çizilmiş)
        parçalar KENDİ ayrı zincirlerinde kalır — hiçbiri atılmaz.

        ESKİ DAVRANIŞ (bug): tüm parçalar TEK bir zincire zorlanıyordu;
        birbirine bağlanamayan (paralel/ayrık) parça(lar) "leftover"
        sayılıp sessizce hesap dışı bırakılıyordu. Bu yüzden "aynı yolun
        iki tarafını çizince bir tarafı gitmiyor" sorunu oluşuyordu.
        Artık her bağlı grup kendi ayrı güzergahı olarak korunuyor ve
        çağıran kod (aşağıda) her biri için AYRI AYRI direk hesabı
        çalıştırıyor.

        Returns:
            List[List[[lon, lat], ...]] — her biri bağımsız bir hattı
            temsil eden koordinat listelerinden oluşan liste.
        """
        if not line_drawings:
            return []

        all_segments = [list(d["geometry"]["coordinates"]) for d in line_drawings]
        if len(all_segments) == 1:
            return [all_segments[0]]

        def _dist(p1, p2):
            lon1, lat1 = p1
            lon2, lat2 = p2
            return haversine_distance_m(
                GeoPoint(lat=lat1, lon=lon1), GeoPoint(lat=lat2, lon=lon2)
            )

        def _grow_chain(seed, pool):
            chain = seed
            merged_any = True
            while merged_any and pool:
                merged_any = False
                chain_start, chain_end = chain[0], chain[-1]
                best = None  # (index, mode, dist)
                for i, seg in enumerate(pool):
                    seg_start, seg_end = seg[0], seg[-1]
                    for mode, dist in (
                        ("end_to_start", _dist(chain_end, seg_start)),
                        ("end_to_end", _dist(chain_end, seg_end)),
                        ("start_to_start", _dist(chain_start, seg_start)),
                        ("start_to_end", _dist(chain_start, seg_end)),
                    ):
                        if dist <= CHAIN_MERGE_SNAP_M and (best is None or dist < best[2]):
                            best = (i, mode, dist)
                if best is not None:
                    i, mode, _dist_val = best
                    seg = pool.pop(i)
                    if mode == "end_to_start":
                        chain = chain + seg[1:]
                    elif mode == "end_to_end":
                        chain = chain + list(reversed(seg))[1:]
                    elif mode == "start_to_start":
                        chain = list(reversed(chain)) + seg[1:]
                    elif mode == "start_to_end":
                        chain = seg[:-1] + chain
                    merged_any = True
            return chain

        remaining = list(all_segments)
        chains = []
        while remaining:
            seed = remaining.pop(0)
            chain = _grow_chain(seed, remaining)
            chains.append(chain)
        # En uzun hat en başta gösterilsin (ör. "A tarafı" olarak).
        chains.sort(key=_line_length_m, reverse=True)
        return chains

    sketch_chains_lonlat = []
    if draw_result:
        drawings = draw_result.get("all_drawings") or []
        line_drawings = [
            d for d in drawings if d.get("geometry", {}).get("type") == "LineString"
        ]
        if line_drawings:
            sketch_chains_lonlat = _split_into_chains(line_drawings)

    sketch_chains = [
        [GeoPoint(lat=lat, lon=lon) for lon, lat in chain]
        for chain in sketch_chains_lonlat
        if len(chain) >= 2
    ]

    status_col, action_col = st.columns([3, 1], vertical_alignment="center")
    with status_col:
        if sketch_chains:
            total_points = sum(len(c) for c in sketch_chains)
            if len(sketch_chains) == 1:
                st.success(f"✅ Çizim algılandı: {total_points} nokta. Hesaplamaya hazır.")
            else:
                st.success(
                    f"✅ {len(sketch_chains)} AYRI hat algılandı (toplam "
                    f"{total_points} nokta). Uçları birbirine "
                    f"{CHAIN_MERGE_SNAP_M:.0f} m'den yakın olmayan çizgiler "
                    "artık atılmıyor — her biri kendi bağımsız güzergahı "
                    "olarak (ör. aynı yolun iki tarafı için) AYRI AYRI "
                    "hesaplanıp haritada farklı renkte gösterilecek."
                )
            sketch_points = sketch_chains[0]
        else:
            sketch_points = None
            st.info("Henüz bir çizgi çizilmedi. Haritadan bir hat çizin.")
    with action_col:
        run_button = st.button(
            "🚀 Direkleri Hesapla",
            type="primary",
            disabled=not sketch_chains,
            use_container_width=True,
        )
else:
    run_button = st.button("🚀 Koridor Verisini Getir", type="primary")

# ------------------------------------------------------------------ #
# Hesaplama
# ------------------------------------------------------------------ #
if run_button:
    with st.spinner("OSM verisi çekiliyor ve sanal düğümler üretiliyor... (biraz sürebilir)"):
        try:
            runs = []
            if input_mode == "sketch":
                # Her AYRI çizilmiş hat (ör. aynı yolun iki tarafı) kendi
                # A/B'siyle bağımsız bir koleksiyon/rota olarak çalıştırılır
                # — böylece hiçbiri diğerinin "leftover"ı olarak atılmaz.
                for chain_points in sketch_chains:
                    start_pt = chain_points[0]
                    end_pt = chain_points[-1]
                    collector = CorridorDataCollector(
                        start=start_pt,
                        end=end_pt,
                        corridor_buffer_m=float(buffer_m),
                        node_spacing_m=float(spacing_m),
                        pole_offset_m=float(offset_m),
                        offset_side=offset_side,
                        voltage_class=voltage_class,
                    )
                    chain_data = collector.run(
                        sketch_points=chain_points,
                        direct_line_mode=use_direct_line,
                    )
                    runs.append((chain_data, collector, chain_points))
            else:
                collector = CorridorDataCollector(
                    start=GeoPoint(lat=start_lat, lon=start_lon),
                    end=GeoPoint(lat=end_lat, lon=end_lon),
                    corridor_buffer_m=float(buffer_m),
                    node_spacing_m=float(spacing_m),
                    pole_offset_m=float(offset_m),
                    offset_side=offset_side,
                    voltage_class=voltage_class,
                )
                data = collector.run()
                runs.append((data, collector, None))

            st.session_state["corridor_runs"] = runs
            # Geriye dönük uyumluluk (diğer kodun tekil `corridor_data`
            # okuduğu yerler için): ilk/en uzun hattı ana veri olarak tut.
            st.session_state["corridor_data"] = runs[0][0]
            st.session_state["max_span_m"] = runs[0][1].max_span_m
            st.session_state["voltage_class_used"] = runs[0][1].voltage_class
        except Exception as exc:  # noqa: BLE001
            st.error(f"Veri toplama sırasında bir hata oluştu: {exc}")
            st.session_state.pop("corridor_data", None)
            st.session_state.pop("corridor_runs", None)

data = st.session_state.get("corridor_data")
corridor_runs = st.session_state.get("corridor_runs") or []

# ------------------------------------------------------------------ #
# Sonuçlar
# ------------------------------------------------------------------ #
if data is None:
    if input_mode == "manual":
        st.info("Sol panelden A/B koordinatlarını girip **'Koridor Verisini Getir'** butonuna basın.")
else:
    _run_labels = ["A tarafı", "B tarafı", "C tarafı", "D tarafı"]

    def _route_source_msg(route_source, label=None):
        prefix = f"**[{label}]** " if label else ""
        if route_source == "sketch_matched":
            st.success(
                f"✅ {prefix}Çiziminiz gerçek yol ağına oturtuldu (map-matching) "
                "ve direkler bu rotanın kenarına offsetli olarak yerleştirildi."
            )
        elif route_source == "sketch_direct":
            st.success(
                f"✅ {prefix}Doğrudan hat modu: direkler yol ağına oturtulmadan, "
                "tam olarak çizdiğiniz kroki üzerine (offset ayarınıza göre "
                "kaydırılarak) yerleştirildi."
            )
        elif route_source == "road_snapped":
            st.success(f"✅ {prefix}Direkler yol ağı üzerinden hesaplanan rotaya oturtuldu (kenar offsetli).")
        else:
            st.warning(
                f"⚠️ {prefix}Yol ağı üzerinden geçerli bir rota bulunamadı — düz "
                "hatta geri dönüldü. Bu durumda düğümler bina/arazi üzerinden "
                "geçebilir; sonuç yalnızca kaba bir ön izlemedir."
            )

    if len(corridor_runs) > 1:
        st.info(
            f"ℹ️ {len(corridor_runs)} bağımsız hat birlikte hesaplandı "
            "(örn. bir yolun iki tarafı). Her biri aşağıda ayrı renkte "
            "gösteriliyor ve aralarında **hiçbir kablo bağlantısı "
            "çizilmiyor**."
        )
        for i, (run_data, _collector, _pts) in enumerate(corridor_runs):
            label = _run_labels[i] if i < len(_run_labels) else f"{i + 1}. hat"
            _route_source_msg(run_data.route_source, label)
    else:
        _route_source_msg(data.route_source)

    all_nodes = [n for run_data, _c, _p in corridor_runs for n in run_data.virtual_nodes]
    num_angle = sum(1 for n in all_nodes if n.pole_type == "açı")
    num_end = sum(1 for n in all_nodes if n.pole_type == "nihayet")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Bina Sayısı", sum(len(r[0].buildings_gdf) for r in corridor_runs))
    col2.metric("Engel Poligonu", sum(len(r[0].obstacles_gdf) for r in corridor_runs))
    col3.metric("Yol Segmenti", sum(len(r[0].roads_gdf) for r in corridor_runs))
    col4.metric("Sanal Düğüm (toplam)", len(all_nodes))
    col5.metric("Açı / Nihayet Direği", f"{num_angle} / {num_end}")
    if len(corridor_runs) > 1:
        st.caption(
            "ℹ️ Yukarıdaki bina/engel/yol sayıları hatlar arasında "
            "çakışan koridor alanları varsa aynı öğeyi birden fazla kez "
            "sayabilir; bu yalnızca bir ön izleme özetidir."
        )

    _max_span = st.session_state.get("max_span_m")
    _voltage_used = st.session_state.get("voltage_class_used")
    if _max_span is not None:
        _over_limit = [n for n in all_nodes if n.span_length_m > _max_span + 1e-6]
        if _over_limit:
            st.warning(
                f"⚠️ {len(_over_limit)} direğin açıklığı, seçilen gerilim "
                f"sınıfının ({_voltage_used}) yaklaşık azami açıklığını "
                f"({_max_span:.0f} m) aşıyor — genelde köşe/vertex "
                "noktalarında kaçınılmazdır, projelendirme aşamasında "
                "ayrıca değerlendirilmelidir."
            )

    st.caption(
        "💡 Haritadaki herhangi bir direk simgesine **tıklayarak** "
        "koordinatlarını görebilirsiniz. Direk tipi (nihayet / açı / ara) "
        "kaba bir ön çalışma sınıflamasıdır, TEDAŞ onaylı projelendirmenin "
        "yerine geçmez."
    )

    show_span_distances = st.checkbox(
        "📏 Direkler arası mesafeyi haritada etiket olarak göster",
        value=False,
        help=(
            "Direkler arası kablo geçişi (kesikli yeşil çizgi) haritada "
            "HER ZAMAN gösterilir. Bu kutu yalnızca, işaretlenirse her "
            "segmentin üzerine uzunluğunu (metre) yazan ek bir etiket "
            "ekler. Her direğin popup'ında da bir önceki direğe olan "
            "mesafe zaten yer alır."
        ),
    )
    show_corridor_polygon = st.checkbox(
        "🟧 Koridor tampon alanını (OSM veri sınırını) haritada göster",
        value=False,
        help=(
            "İşaretlenirse, OSM verisinin çekildiği tampon poligonunun "
            "sınırı turuncu bir çizgiyle gösterilir. Varsayılan olarak "
            "kapalıdır — bu sınır direklerin ya da hattın kendisi DEĞİLDİR, "
            "sadece hangi alanda bina/yol verisi arandığını gösterir ve "
            "haritayı karmaşıklaştırabilir."
        ),
    )

    with st.expander("🔧 Teşhis Bilgisi (sorun bildirirken bu bölümü paylaşın)"):
        diag_runs = []
        for i, (run_data, _c, run_points) in enumerate(corridor_runs):
            raw_len_m = None
            if run_points and len(run_points) >= 2:
                raw_len_m = sum(
                    haversine_distance_m(run_points[j], run_points[j + 1])
                    for j in range(len(run_points) - 1)
                )
            computed_len_m = (
                run_data.virtual_nodes[-1].cumulative_distance_m
                if run_data.virtual_nodes else 0.0
            )
            label = _run_labels[i] if i < len(_run_labels) else f"{i + 1}. hat"
            diag_runs.append({
                "hat": label,
                "route_source": run_data.route_source,
                "cizilen_kroki_nokta_sayisi": len(run_points) if run_points else 0,
                "cizilen_kroki_toplam_uzunluk_m": (
                    round(raw_len_m, 1) if raw_len_m is not None else None
                ),
                "hesaplanan_rota_toplam_uzunluk_m": round(computed_len_m, 1),
                "uretilen_direk_sayisi": len(run_data.virtual_nodes),
                "A_baslangic": [round(run_data.start.lat, 6), round(run_data.start.lon, 6)],
                "B_bitis": [round(run_data.end.lat, 6), round(run_data.end.lon, 6)],
            })
            if (
                raw_len_m is not None
                and computed_len_m < 0.5 * raw_len_m
            ):
                st.error(
                    f"⚠️ [{label}] Hesaplanan rota, çizdiğiniz krokinin "
                    "uzunluğunun yarısından kısa. Bu, direklerin krokinin "
                    "büyük kısmı boyunca değil sadece küçük bir parçasında "
                    "üretildiği anlamına gelir."
                )
        st.json({
            "input_mode": input_mode,
            "direct_line_mode_secildi_mi": (
                use_direct_line if input_mode == "sketch" else "n/a (manuel mod)"
            ),
            "hat_sayisi": len(corridor_runs),
            "hatlar": diag_runs,
        })

    RUN_CABLE_COLORS = ["#2ca02c", "#ff7f0e", "#9467bd", "#17becf"]
    RUN_SKETCH_COLORS = ["#e91e63", "#3f51b5", "#009688", "#795548"]

    all_lats = [n.point.lat for r in corridor_runs for n in r[0].virtual_nodes] or [
        r[0].start.lat for r in corridor_runs
    ]
    all_lons = [n.point.lon for r in corridor_runs for n in r[0].virtual_nodes] or [
        r[0].start.lon for r in corridor_runs
    ]
    mid_lat = sum(all_lats) / len(all_lats)
    mid_lon = sum(all_lons) / len(all_lons)

    fmap = folium.Map(location=[mid_lat, mid_lon], zoom_start=15, tiles="OpenStreetMap")

    pole_type_style = {
        "nihayet": ("#9467bd", "🟣 Nihayet Direği", 8),
        "açı": ("#d62728", "🔴 Açı Direği", 7),
        "ara": ("#1f77b4", "🔵 Ara Direk", 4),
    }
    max_span_for_run = st.session_state.get("max_span_m")

    for run_idx, (run_data, _collector, run_points) in enumerate(corridor_runs):
        run_label = (
            _run_labels[run_idx] if run_idx < len(_run_labels) else f"{run_idx + 1}. hat"
        )
        side_tag = f" ({run_label})" if len(corridor_runs) > 1 else ""
        cable_color = RUN_CABLE_COLORS[run_idx % len(RUN_CABLE_COLORS)]
        sketch_color = RUN_SKETCH_COLORS[run_idx % len(RUN_SKETCH_COLORS)]

        folium.Marker(
            [run_data.start.lat, run_data.start.lon],
            tooltip=f"A - Başlangıç{side_tag}",
            popup=folium.Popup(
                f"<b>A - Başlangıç{side_tag}</b><br>"
                f"Lat: {run_data.start.lat:.6f}<br>Lon: {run_data.start.lon:.6f}",
                max_width=250,
            ),
            icon=folium.Icon(color="green"),
        ).add_to(fmap)
        folium.Marker(
            [run_data.end.lat, run_data.end.lon],
            tooltip=f"B - Bitiş{side_tag}",
            popup=folium.Popup(
                f"<b>B - Bitiş{side_tag}</b><br>"
                f"Lat: {run_data.end.lat:.6f}<br>Lon: {run_data.end.lon:.6f}",
                max_width=250,
            ),
            icon=folium.Icon(color="red"),
        ).add_to(fmap)

        # Kullanıcının çizdiği ham krokiyi de referans olarak (soluk,
        # kesikli çizgi) haritaya ekleyelim ki hesaplanan rotayla
        # karşılaştırılabilsin. Her hat kendi rengiyle gösterilir.
        if run_points:
            folium.PolyLine(
                [(p.lat, p.lon) for p in run_points],
                color=sketch_color,
                weight=3,
                opacity=0.5,
                dash_array="6,8",
                tooltip=f"Sizin çiziminiz (ham kroki){side_tag}",
            ).add_to(fmap)

        # ÖNEMLİ: prev_node her hat başında sıfırlanır — böylece bir
        # hattın son direği ile bir SONRAKİ (bağımsız) hattın ilk direği
        # arasında YANLIŞLIKLA kablo çizilmez.
        prev_node = None
        for node in run_data.virtual_nodes:
            color, label, radius = pole_type_style.get(
                node.pole_type, pole_type_style["ara"]
            )
            span_exceeds = (
                max_span_for_run is not None
                and node.span_length_m > max_span_for_run + 1e-6
            )
            popup_html = (
                f"<b>{label} #{node.node_id}{side_tag}</b><br>"
                f"Enlem (lat): {node.point.lat:.6f}<br>"
                f"Boylam (lon): {node.point.lon:.6f}<br>"
                f"A'dan mesafe: {node.cumulative_distance_m:.0f} m<br>"
                + (
                    f"Sapma açısı: {node.deflection_angle_deg:.0f}°<br>"
                    if node.is_corner and node.node_id > 0
                    else ""
                )
                + (
                    f"Önceki direğe mesafe: {node.span_length_m:.0f} m"
                    + (" ⚠️ azami açıklığı aşıyor" if span_exceeds else "")
                    if node.node_id > 0
                    else "Önceki direk yok (başlangıç)"
                )
            )
            folium.CircleMarker(
                [node.point.lat, node.point.lon],
                radius=radius,
                color="#ff9800" if span_exceeds else color,
                fill=True,
                fill_opacity=0.9 if node.is_corner else 0.8,
                tooltip=(
                    f"{label} #{node.node_id}{side_tag} "
                    f"({node.cumulative_distance_m:.0f}m) — koordinat için tıklayın"
                ),
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(fmap)

            # Direkler arası kablo geçişini (fiziksel hat güzergahını) her
            # zaman kesikli bir çizgiyle gösteriyoruz — bu, "Direkler
            # arası mesafeyi göster" seçeneğinden bağımsızdır; o seçenek
            # yalnızca segment üzerindeki metre etiketini açıp kapatır.
            if prev_node is not None:
                folium.PolyLine(
                    [
                        (prev_node.point.lat, prev_node.point.lon),
                        (node.point.lat, node.point.lon),
                    ],
                    color=cable_color,
                    weight=3,
                    opacity=0.85,
                    dash_array="10,6",
                    tooltip=f"Kablo geçişi{side_tag}: {node.span_length_m:.0f} m",
                ).add_to(fmap)

            # İsteğe bağlı: bu direkle bir önceki direk arasındaki
            # mesafeyi segment üzerinde bir etiket olarak da göster.
            if show_span_distances and prev_node is not None:
                mid_seg_lat = (prev_node.point.lat + node.point.lat) / 2
                mid_seg_lon = (prev_node.point.lon + node.point.lon) / 2
                folium.Marker(
                    [mid_seg_lat, mid_seg_lon],
                    icon=folium.DivIcon(
                        html=(
                            "<div style='font-size:11px;font-weight:600;"
                            "color:#1a1a1a;background:rgba(255,255,255,0.85);"
                            "padding:1px 4px;border-radius:3px;white-space:nowrap;"
                            "transform:translate(-50%,-50%);'>"
                            f"{node.span_length_m:.0f} m</div>"
                        )
                    ),
                ).add_to(fmap)

            prev_node = node

        if show_corridor_polygon and run_data.corridor_polygon is not None:
            folium.GeoJson(
                run_data.corridor_polygon.__geo_interface__,
                style_function=lambda _: {"color": "#ff7f0e", "fillOpacity": 0.05},
            ).add_to(fmap)

    st_folium(fmap, width=None, height=600, returned_objects=[], key="result_map")
