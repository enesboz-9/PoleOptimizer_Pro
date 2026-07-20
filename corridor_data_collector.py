"""
corridor_data_collector.py
============================================================
PoleOptimizer Pro - Modül 1: Koridor Veri Toplama & Sanal Düğüm Üretimi
============================================================

Bu modül, kullanıcının haritadan seçtiği Başlangıç (A) ve Bitiş (B)
koordinatları arasındaki hat koridorunu tanımlar; bu koridor içindeki
OpenStreetMap (yol, bina, su, doğal/sit alanı) verilerini OSMnx/OSM
Overpass API üzerinden çeker ve A-B doğrusu üzerinde her X metrede bir
"aday direk noktası" (candidate pole / virtual node) üretir.

Üretilen düğümler, sonraki modüllerde (DEM tabanlı eğim kontrolü,
sag/clearance hesaplayıcı, optimizasyon motoru) kısıt kontrolüne tabi
tutulacak ham girdi (raw candidate set) olarak kullanılır.

Bağımlılıklar:
    pip install osmnx geopandas shapely networkx pyproj numpy

Yazar: PoleOptimizer Pro Engineering Team
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np

try:
    import osmnx as ox
    import geopandas as gpd
    import networkx as nx
    from shapely.geometry import Point, LineString, Polygon, MultiLineString
    from shapely.ops import transform as shapely_transform
    from shapely.strtree import STRtree
    import pyproj
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Bu modül osmnx, geopandas, networkx, shapely ve pyproj paketlerine "
        "ihtiyaç duyar. Kurulum için: pip install osmnx geopandas networkx "
        "shapely pyproj"
    ) from exc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PoleOptimizerPro.CorridorDataCollector")


# ------------------------------------------------------------------ #
# Veri Sınıfları (Data Classes)
# ------------------------------------------------------------------ #

@dataclass
class GeoPoint:
    """WGS84 (EPSG:4326) formatında enlem/boylam noktası.

    Attributes:
        lat: Enlem (Latitude), derece cinsinden.
        lon: Boylam (Longitude), derece cinsinden.
    """
    lat: float
    lon: float

    def as_tuple(self) -> Tuple[float, float]:
        """(lat, lon) tuple olarak döner."""
        return (self.lat, self.lon)

    def as_xy(self) -> Tuple[float, float]:
        """(lon, lat) sırasıyla döner (Shapely/GIS Point sırası için)."""
        return (self.lon, self.lat)


@dataclass
class VirtualNode:
    """Aday direk lokasyonu olarak değerlendirilecek sanal düğüm.

    Bu sınıf henüz kısıt kontrolünden (eğim, sag/clearance, yasak alan
    maskesi) geçmemiş HAM bir adaydır. `is_feasible` alanı sonraki
    modüller tarafından güncellenecektir.

    Attributes:
        node_id: Düğümün sıra numarası (0 = başlangıç noktası A).
        point: Düğümün coğrafi konumu (WGS84).
        cumulative_distance_m: A noktasından bu düğüme kadar olan
            kümülatif mesafe (metre, Haversine bazlı).
        span_length_m: Bir önceki direkten (node_id - 1) bu direğe kadar
            olan mesafe (metre) — yani bu direkle bir önceki direk
            arasındaki "açıklık/span" uzunluğu. İlk düğüm (A, node_id=0)
            için 0.0'dır (önceki düğüm yoktur).
        is_feasible: Kısıt kontrolünden geçip geçmediği (varsayılan None
            = henüz kontrol edilmedi).
        elevation_m: DEM'den okunacak yükseklik (varsayılan None,
            sonraki modülde doldurulur).
        blocking_reason: Eğer is_feasible=False ise, sebebi (örn.
            "slope_exceeds_25pct", "inside_building_polygon" vb.).
        is_corner: True ise, bu düğüm rotanın bir dönüş/kavşak noktasında
            (yol ağının bir vertex'inde) bulunuyor demektir. Kavşak
            direkleri genellikle açısal çekme kuvvetine maruz kaldığından
            (guy-wire / gerilme direği) mühendislik açısından düz hat
            üzerindeki ara direklerden farklı muamele görür; bu yüzden
            zorunlu (atlanamaz) düğüm olarak işaretlenir.
        deflection_angle_deg: Bu düğümde, gelen ve giden hat segmentleri
            arasındaki sapma açısı (derece, 0 = tam düz devam). Yalnızca
            `is_corner=True` olan düğümler için anlamlıdır; düz ara
            direklerde 0.0'dır.
        pole_type: TEDAŞ şartname pratiğine yakın kaba direk sınıflaması:
            "nihayet" -> hattın başlangıç (A) ya da bitiş (B) ucu; hat bu
                direkte sonlanır, tam gerilim (nihayet) direğidir.
            "açı" -> güzergahın `ANGLE_DEFLECTION_THRESHOLD_DEG` değerini
                aşan bir yönde döndüğü kavşak/köşe noktası; açısal çekme
                kuvveti nedeniyle payanda/gerilme direği gerektirir.
            "ara" -> düz hat üzerindeki (ya da açısı eşiğin altında kalan)
                taşıyıcı/ara direk.
            NOT: Bu sınıflama, gerçek TEDAŞ projelendirme şartnamesinin
            (statik/mekanik hesap, zemin etüdü, rüzgar/buz yükü vb.)
            YERİNE GEÇMEZ — sadece saha ön çalışması için kaba bir
            yönlendirmedir; nihai direk tipi TEDAŞ onaylı projelendirme
            mühendisi tarafından belirlenmelidir.
    """
    node_id: int
    point: GeoPoint
    cumulative_distance_m: float
    span_length_m: float = 0.0
    is_feasible: Optional[bool] = None
    elevation_m: Optional[float] = None
    blocking_reason: Optional[str] = None
    is_corner: bool = False
    deflection_angle_deg: float = 0.0
    pole_type: str = "ara"


@dataclass
class CorridorData:
    """A-B hattı koridoruna ait tüm ham GIS verilerini tutan konteyner.

    Attributes:
        start: Başlangıç noktası (A).
        end: Bitiş noktası (B).
        corridor_buffer_m: Koridor genişliği (A-B hattının her iki
            yanına uygulanan tampon/buffer mesafesi, metre).
        buildings_gdf: OSM'den çekilen bina poligonları (GeoDataFrame).
        obstacles_gdf: OSM'den çekilen su/nehir/sit/doğal koruma alanı
            poligonları (GeoDataFrame).
        roads_gdf: OSM'den çekilen yol/karayolu çizgileri (GeoDataFrame).
        corridor_polygon: A-B hattı etrafında oluşturulan tampon
            poligonu (WGS84, EPSG:4326).
        virtual_nodes: A'dan B'ye doğru üretilen sanal direk aday
            düğümleri listesi.
        route_source: Düğümlerin nasıl üretildiğini belirtir:
            "road_snapped" -> A/B noktaları arasında yol ağı üzerinden en
                kısa rota bulundu ve direkler yol kenarına offsetlendi.
            "sketch_matched" -> kullanıcının haritada kalemle çizdiği
                (eğik/yamuk olabilen) kroki hattı, gerçek OSM yol ağına
                harita-eşleştirme (map-matching) ile oturtuldu ve
                direkler bu eşleştirilmiş rotanın kenarına offsetlendi
                (tercih edilen, en gerçekçi durum — yol/patika verisi
                olan güzergahlarda).
            "sketch_direct" -> kullanıcı yol ağına oturtmayı bilinçli
                olarak devre dışı bıraktı (`direct_line_mode=True`);
                direkler kullanıcının çizdiği krokinin KENDİSİ üzerine
                (varsa offsetlenerek) yerleştirildi. OSM'de yol/patika
                verisi olmayan orman, arazi, mesire alanı gibi
                güzergahlarda kullanılır — bu bölgelerde map-matching
                tüm ara noktaları en yakın dış yola "çökertip" hattı
                anlamsızlaştırabildiği için tercih edilir.
            "straight_line_fallback" -> yol ağı bulunamadı/rota
                hesaplanamadı, A-B (ya da kroki uç noktaları) düz hattına
                geri dönüldü (bu durumda sonuçlar binaların/engellerin
                üzerinden geçebilir ve sadece kaba bir ön izleme olarak
                değerlendirilmelidir).
    """
    start: GeoPoint
    end: GeoPoint
    corridor_buffer_m: float
    buildings_gdf: gpd.GeoDataFrame = field(default_factory=gpd.GeoDataFrame)
    obstacles_gdf: gpd.GeoDataFrame = field(default_factory=gpd.GeoDataFrame)
    roads_gdf: gpd.GeoDataFrame = field(default_factory=gpd.GeoDataFrame)
    corridor_polygon: Optional[Polygon] = None
    virtual_nodes: List[VirtualNode] = field(default_factory=list)
    route_source: str = "straight_line_fallback"


# ------------------------------------------------------------------ #
# Yardımcı Fonksiyonlar (Utility Functions)
# ------------------------------------------------------------------ #

def haversine_distance_m(p1: GeoPoint, p2: GeoPoint) -> float:
    """İki WGS84 nokta arasındaki kuş uçuşu (Haversine) mesafeyi hesaplar.

    Args:
        p1: İlk nokta.
        p2: İkinci nokta.

    Returns:
        Metre cinsinden mesafe.
    """
    R = 6_371_000.0  # Dünya yarıçapı (metre)
    lat1, lon1 = math.radians(p1.lat), math.radians(p1.lon)
    lat2, lon2 = math.radians(p2.lat), math.radians(p2.lon)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def _get_utm_epsg(lat: float, lon: float) -> str:
    """Verilen enlem/boylama en uygun UTM projeksiyon kodunu (EPSG) döner.

    Metre bazlı hassas mesafe/buffer hesapları için WGS84 (EPSG:4326)
    yerine yerel UTM projeksiyonuna geçmek gerekir; aksi halde
    interpolasyon ve buffer hataları oluşur.

    Args:
        lat: Enlem.
        lon: Boylam.

    Returns:
        "EPSG:XXXXX" formatında UTM zon kodu.
    """
    zone_number = int((lon + 180) / 6) + 1
    hemisphere_code = 326 if lat >= 0 else 327  # Kuzey / Güney yarımküre
    return f"EPSG:{hemisphere_code}{zone_number:02d}"


def _build_transformers(utm_epsg: str) -> Tuple[pyproj.Transformer, pyproj.Transformer]:
    """WGS84 <-> UTM dönüşümü için ileri ve geri transformer'ları üretir.

    Args:
        utm_epsg: Hedef UTM EPSG kodu (örn. "EPSG:32636").

    Returns:
        (to_utm, to_wgs84) transformer çifti.
    """
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)
    to_wgs84 = pyproj.Transformer.from_crs(utm_epsg, "EPSG:4326", always_xy=True)
    return to_utm, to_wgs84


# ------------------------------------------------------------------ #
# Ana Sınıf: CorridorDataCollector
# ------------------------------------------------------------------ #

class CorridorDataCollector:
    """A-B hattı için OSM verisini toplayan ve sanal düğüm üreten servis.

    Kullanım:
        collector = CorridorDataCollector(
            start=GeoPoint(lat=40.9500, lon=36.3500),
            end=GeoPoint(lat=40.9700, lon=36.3900),
            corridor_buffer_m=150.0,
            node_spacing_m=40.0,
        )
        corridor_data = collector.run()

    Attributes:
        start: Hattın başlangıç noktası (A).
        end: Hattın bitiş noktası (B).
        corridor_buffer_m: A-B hattı etrafında OSM verisinin çekileceği
            tampon genişliği (metre). Kısıt matrisi bu alan içinde
            oluşturulur.
        node_spacing_m: Sanal düğümler arası mesafe (metre). Bu, iletken
            tipine göre belirlenen maksimum açıklığın (span length)
            altında seçilmelidir (örn. max açıklık 60m ise 20-30m
            aralıklarla düğüm üretmek, optimizasyon motoruna daha ince
            arama uzayı sağlar).
    """

    # OSM Overpass sorgusunda "engel" (obstacle) sayılacak etiketler.
    # Sit alanı, su kütlesi, doğal koruma alanı vb. -> "Direk Dikilemez Alan"
    OBSTACLE_TAGS = {
        "natural": ["water", "wetland", "wood"],
        "waterway": True,
        "leisure": ["nature_reserve"],
        "boundary": ["protected_area", "national_park"],
        "landuse": ["forest", "military"],
    }

    BUILDING_TAGS = {"building": True}
    ROAD_TAGS = {"highway": True}

    # ---------------------------------------------------------------- #
    # TEDAŞ şartnamesine yaklaşım: gerilim sınıfına göre azami açıklık
    # (max span) değerleri.
    #
    # ÖNEMLİ KAYNAK NOTU: Bu değerler, Türkiye dağıtım şebekesinde
    # (TEDAŞ/bağlı dağıtım şirketleri) YAYGIN OLARAK UYGULANAN tipik
    # havai hat projelendirme pratiklerine dayanan YAKLAŞIK üst
    # sınırlardır (iletken kesiti, direk boyu/sınıfı, güzergahın
    # rüzgar/buz yükü bölgesi, zemin cinsi ve arazi eğimine göre saha
    # bazında değişebilir). Belirli bir TEDAŞ-MYD/TEDAŞ-MLZ şartname
    # maddesinden BİREBİR alınmamıştır; nihai açıklık değeri her zaman
    # ilgili bölge dağıtım şirketinin güncel projelendirme şartnamesi ve
    # onaylı statik/mekanik hesap ile TEYİT EDİLMELİDİR. Bu araç yalnızca
    # saha ön çalışması (fizibilite) amaçlıdır, uygulama projesi yerine
    # geçmez.
    VOLTAGE_SPAN_LIMITS_M = {
        "AG": 40.0,      # Alçak Gerilim (0.4 kV) dağıtım hattı
        "OG": 100.0,     # Orta Gerilim (34.5 kV) dağıtım hattı
    }

    # Bir düğümdeki yön değişimi (sapma açısı) bu eşiği aşarsa, düğüm
    # "açı direği" (angle pole / açısal çekme kuvvetine maruz, payandalı)
    # olarak sınıflandırılır; aksi halde düz hat ara direği sayılır. 15°
    # değeri de yukarıdaki gibi tipik saha pratiğine dayanan yaklaşık bir
    # eşiktir, kesin şartname maddesi değildir.
    ANGLE_DEFLECTION_THRESHOLD_DEG = 15.0

    def __init__(
        self,
        start: GeoPoint,
        end: GeoPoint,
        corridor_buffer_m: float = 150.0,
        node_spacing_m: float = 30.0,
        pole_offset_m: float = 5.0,
        offset_side: str = "right",
        voltage_class: str = "OG",
    ) -> None:
        if corridor_buffer_m <= 0:
            raise ValueError("corridor_buffer_m sıfırdan büyük olmalıdır.")
        if node_spacing_m <= 0:
            raise ValueError("node_spacing_m sıfırdan büyük olmalıdır.")
        if pole_offset_m < 0:
            raise ValueError("pole_offset_m negatif olamaz.")
        if offset_side not in ("left", "right"):
            raise ValueError("offset_side yalnızca 'left' veya 'right' olabilir.")
        if voltage_class not in self.VOLTAGE_SPAN_LIMITS_M:
            raise ValueError(
                f"voltage_class yalnızca {list(self.VOLTAGE_SPAN_LIMITS_M)} "
                f"olabilir, verilen: {voltage_class!r}"
            )

        self.start = start
        self.end = end
        self.corridor_buffer_m = corridor_buffer_m
        self.voltage_class = voltage_class
        self.max_span_m = self.VOLTAGE_SPAN_LIMITS_M[voltage_class]
        if node_spacing_m > self.max_span_m:
            logger.warning(
                "node_spacing_m (%.1fm) seçilen gerilim sınıfının (%s) "
                "yaklaşık azami açıklığını (%.1fm) aşıyor; düğüm üretimi "
                "sırasında %.1fm ile sınırlandırılacak.",
                node_spacing_m, voltage_class, self.max_span_m, self.max_span_m,
            )
        self.node_spacing_m = min(node_spacing_m, self.max_span_m)
        # pole_offset_m: Direklerin yol ORTA HATTINDAN ne kadar uzağa,
        # kaldırım/yol kenarına doğru kaydırılacağı (metre). 0 verilirse
        # direkler yolun tam ortasında kalır (gerçekçi değildir, trafiği
        # keser); tipik olarak 3-6m arası bir değer kaldırım kenarına
        # karşılık gelir. Kesin değer, gerçek kaldırım genişliği bilinene
        # kadar bir yaklaşıklıktır.
        self.pole_offset_m = pole_offset_m
        # offset_side: Yürüyüş yönüne (A'dan B'ye) göre hangi tarafa
        # offsetleneceği. Gerçek projede bu, YEDAŞ'ın mülkiyet/irtifak
        # hakkı bulunan taraf gibi saha bilgisine göre seçilmelidir;
        # şu an için sabit bir varsayılan sunuyoruz.
        self.offset_side = offset_side

        # A-B hattının orta noktasına göre en uygun UTM projeksiyonunu seç.
        mid_lat = (start.lat + end.lat) / 2
        mid_lon = (start.lon + end.lon) / 2
        self.utm_epsg = _get_utm_epsg(mid_lat, mid_lon)
        self.to_utm, self.to_wgs84 = _build_transformers(self.utm_epsg)

        logger.info(
            "CorridorDataCollector başlatıldı | A=%s B=%s | UTM=%s | "
            "buffer=%.1fm | spacing=%.1fm (azami %.1fm, %s) | "
            "pole_offset=%.1fm (%s)",
            start.as_tuple(), end.as_tuple(), self.utm_epsg,
            corridor_buffer_m, self.node_spacing_m, self.max_span_m,
            voltage_class, pole_offset_m, offset_side,
        )

    # -------------------------------------------------------------- #
    # Public API
    # -------------------------------------------------------------- #

    def run(
        self,
        sketch_points: Optional[List[GeoPoint]] = None,
        direct_line_mode: bool = False,
    ) -> CorridorData:
        """Tüm veri toplama ve sanal düğüm üretim akışını çalıştırır.

        Args:
            sketch_points: Kullanıcının haritada kalemle serbest çizdiği
                kroki hattının (WGS84) noktaları, A'dan B'ye sırayla.
                Verilirse (en az 2 nokta), A-B arası salt en kısa yol
                yerine, bu kroki gerçek OSM yol ağına harita-eşleştirme
                (map-matching) ile oturtulur — böylece çizim eğik/yamuk
                olsa bile kullanıcının "anlatmak istediği" güzergah
                (örn. belirli bir sokaktan geçme niyeti) korunur.
                None ise (varsayılan), eski davranış korunur: A ve B
                arasında yol ağı üzerinden en kısa rota aranır.
            direct_line_mode: True verilirse (ve `sketch_points` en az 2
                nokta içeriyorsa), yol ağına oturtma (map-matching) HİÇ
                denenmez; direkler doğrudan kullanıcının çizdiği kroki
                üzerine (yalnızca `pole_offset_m` kadar yana kaydırılmış
                olarak) yerleştirilir. OSM'de yol/patika verisi olmayan
                orman, arazi, mesire alanı gibi güzergahlarda kullanılır
                — map-matching bu tür alanlarda tüm ara noktaları en
                yakın dış yola "çökertip" hattı anlamsızlaştırabilir.
                `sketch_points` verilmemişse bu parametrenin bir etkisi
                yoktur (manuel A/B modunda zaten oturtulacak bir kroki
                yoktur).

        Returns:
            Bina, engel, yol katmanlarını, koridor poligonunu ve
            sanal düğüm listesini içeren CorridorData nesnesi.
        """
        corridor_polygon_wgs84 = self._build_corridor_polygon(sketch_points)

        buildings_gdf = self._fetch_osm_layer(
            corridor_polygon_wgs84, self.BUILDING_TAGS, layer_name="buildings"
        )
        obstacles_gdf = self._fetch_osm_layer(
            corridor_polygon_wgs84, self.OBSTACLE_TAGS, layer_name="obstacles"
        )
        roads_gdf = self._fetch_osm_layer(
            corridor_polygon_wgs84, self.ROAD_TAGS, layer_name="roads"
        )

        # --- Rota hesaplama: kroki + doğrudan mod ise hattın kendisi,
        # kroki + eşleştirme modu ise map-matching, kroki yoksa en kısa
        # yol, hiçbiri olmazsa düz hat ---
        virtual_nodes, route_source = self._compute_virtual_nodes(
            corridor_polygon_wgs84, roads_gdf, sketch_points, direct_line_mode
        )

        corridor_data = CorridorData(
            start=self.start,
            end=self.end,
            corridor_buffer_m=self.corridor_buffer_m,
            buildings_gdf=buildings_gdf,
            obstacles_gdf=obstacles_gdf,
            roads_gdf=roads_gdf,
            corridor_polygon=corridor_polygon_wgs84,
            virtual_nodes=virtual_nodes,
            route_source=route_source,
        )

        logger.info(
            "Veri toplama tamamlandı | %d bina | %d engel poligonu | "
            "%d yol segmenti | %d sanal düğüm | rota kaynağı=%s",
            len(buildings_gdf), len(obstacles_gdf), len(roads_gdf),
            len(virtual_nodes), route_source,
        )

        return corridor_data

    # -------------------------------------------------------------- #
    # İç Yardımcı Metotlar (Private Helpers)
    # -------------------------------------------------------------- #

    def _build_corridor_polygon(
        self, sketch_points: Optional[List[GeoPoint]] = None
    ) -> Polygon:
        """Referans hat etrafında `corridor_buffer_m` genişliğinde tampon
        poligonu üretir. Bu poligon, OSM sorgularının sınırlarını
        (bounding region) belirler.

        `sketch_points` verilmişse (kullanıcının kalemle çizdiği kroki),
        referans hat A-B düz çizgisi DEĞİL, krokinin tüm noktalarından
        geçen kırık çizgidir. Bu sayede OSM verisi (bina/engel/yol),
        krokinin düz A-B hattından saptığı bölgeleri de kapsar; aksi
        halde kullanıcı kavisli bir güzergah çizse bile koridor dışında
        kalan sokaklar hiç çekilmemiş olurdu.

        İşlem hassasiyeti için UTM projeksiyonunda buffer uygulanır,
        ardından sonuç tekrar WGS84'e (EPSG:4326) dönüştürülür.

        Args:
            sketch_points: Varsa, kroki hattının WGS84 noktaları.

        Returns:
            WGS84 koordinat sisteminde koridor poligonu (Shapely Polygon).
        """
        if sketch_points and len(sketch_points) >= 2:
            coords_utm = [self.to_utm.transform(*p.as_xy()) for p in sketch_points]
            line_utm = LineString(coords_utm)
        else:
            start_xy_utm = self.to_utm.transform(*self.start.as_xy())
            end_xy_utm = self.to_utm.transform(*self.end.as_xy())
            line_utm = LineString([start_xy_utm, end_xy_utm])

        buffered_utm = line_utm.buffer(self.corridor_buffer_m, cap_style=2)  # flat cap

        buffered_wgs84 = shapely_transform(
            lambda x, y: self.to_wgs84.transform(x, y), buffered_utm
        )
        return buffered_wgs84

    def _fetch_osm_layer(
        self,
        corridor_polygon: Polygon,
        tags: dict,
        layer_name: str,
    ) -> gpd.GeoDataFrame:
        """Verilen koridor poligonu içinde, belirtilen OSM etiketlerine
        (tags) uyan geometrileri OSMnx üzerinden çeker.

        Args:
            corridor_polygon: OSM verisinin çekileceği alan (WGS84).
            tags: OSMnx `features_from_polygon` için OSM etiket sözlüğü.
            layer_name: Loglama amaçlı katman adı.

        Returns:
            Çekilen geometrileri içeren GeoDataFrame. Veri bulunamazsa
            veya bir hata oluşursa boş bir GeoDataFrame döner (pipeline
            bu durumda kesintiye uğramamalıdır).
        """
        try:
            gdf = ox.features_from_polygon(corridor_polygon, tags=tags)
            if gdf is None or gdf.empty:
                logger.warning("'%s' katmanı için veri bulunamadı.", layer_name)
                return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            return gdf
        except Exception as exc:  # noqa: BLE001
            # OSM/Overpass tarafında "veri bulunamadı" (örn. seçilen koridorda
            # hiç bina/su/sit alanı olmaması) normal bir senaryodur ve HATA
            # DEĞİLDİR; pipeline'ı durdurmak yerine boş katman ile devam
            # edilir. Overpass kütüphanesi bu durumu bir exception olarak
            # fırlattığı için burada yakalanıp bilgi (INFO) seviyesinde
            # loglanıyor.
            logger.info(
                "'%s' katmanı için OSM'de veri bulunamadı (bu normal olabilir): %s",
                layer_name, exc,
            )
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    def _compute_virtual_nodes(
        self,
        corridor_polygon: Polygon,
        roads_gdf: gpd.GeoDataFrame,
        sketch_points: Optional[List[GeoPoint]] = None,
        direct_line_mode: bool = False,
    ) -> Tuple[List[VirtualNode], str]:
        """Sanal direk düğümlerini üretmenin ana orkestrasyon metodu.

        `direct_line_mode=True` ve bir kroki verilmişse, yol ağına
        oturtma (map-matching) HİÇ denenmez; direkler doğrudan kullanıcının
        çizdiği hat üzerine yerleştirilir ("sketch_direct"). Bu, OSM'de
        yol/patika verisi olmayan orman/arazi güzergahları için
        tasarlanmıştır.

        Aksi halde (varsayılan davranış): `sketch_points` verilmişse önce
        kroki hattını gerçek yol ağına eşleştirmeyi (map-matching) dener.
        Verilmemişse (ya da eşleştirme başarısız olursa) A ve B arasında
        yol ağı üzerinden en kısa rotayı bulmayı dener (yollardan geçen,
        köşelerde dönen direk dizisi). Bu da başarısız olursa (yol ağı
        bulunamadı, uç noktalar bir yola yakın değil, bağlantısız graf
        vb.) düz hatta geri döner ve bunu log'da açıkça belirtir.

        ÖNEMLİ: Hangi rota kaynağı seçilirse seçilsin, rotanın KENDİSİ
        (çizilen/eşleştirilen hat) hiçbir zaman değiştirilmez ya da
        kısaltılmaz — direkler bu rotanın üzerinde üretilir, ardından her
        biri ayrı ayrı en yakın gerçek yolun KENARINA yapıştırılır (bkz.
        `_place_poles_along_route`). Eskiden olduğu gibi tüm rota tek
        seferde sabit bir mesafeyle kaydırılmaz.

        Args:
            corridor_polygon: OSM yol ağının çekileceği alan (WGS84).
            roads_gdf: `_fetch_osm_layer` ile çekilen ham yol katmanı
                (WGS84) — her direğin en yakın yol kenarına
                yapıştırılması için kullanılır.
            sketch_points: Varsa, kullanıcının kalemle çizdiği kroki
                hattının WGS84 noktaları (A'dan B'ye sırayla).
            direct_line_mode: True ise ve `sketch_points` verilmişse,
                map-matching atlanır ve direkler doğrudan kroki üzerine
                yerleştirilir.

        Returns:
            (virtual_nodes, route_source) tuple'ı. route_source,
            "sketch_matched", "sketch_direct", "road_snapped" veya
            "straight_line_fallback" değerlerinden birini alır.
        """
        if direct_line_mode and sketch_points and len(sketch_points) >= 2:
            direct_line_utm = self._sketch_points_to_utm_line(sketch_points)
            if direct_line_utm is not None:
                nodes = self._place_poles_along_route(direct_line_utm, roads_gdf)
                logger.info(
                    "Doğrudan çizim modu: direkler kroki üzerinde üretildi "
                    "ve (varsa) en yakın yol kenarına yapıştırıldı | %d düğüm.",
                    len(nodes),
                )
                return nodes, "sketch_direct"
            logger.warning(
                "Doğrudan çizim modu için kroki noktaları anlamlı bir "
                "hat oluşturamadı; düz hatta geri dönülüyor."
            )

        road_graph = self._fetch_road_graph(corridor_polygon)

        if road_graph is not None:
            if sketch_points and len(sketch_points) >= 2:
                route_line_utm = self._match_sketch_to_road_graph(
                    road_graph, sketch_points
                )
                route_source = "sketch_matched"
            else:
                route_line_utm = self._compute_road_route_utm(road_graph)
                route_source = "road_snapped"

            if route_line_utm is not None:
                nodes = self._place_poles_along_route(route_line_utm, roads_gdf)
                return nodes, route_source

        logger.warning(
            "Geçerli bir rota (kroki eşleştirmesi ya da en kısa yol) "
            "bulunamadı; düz hatta geri dönülüyor. UYARI: Bu durumda "
            "direk noktaları bina/engel üzerinden geçebilir, sonuç "
            "yalnızca kaba bir ön izlemedir."
        )
        start_xy = np.array(self.to_utm.transform(*self.start.as_xy()))
        end_xy = np.array(self.to_utm.transform(*self.end.as_xy()))
        straight_line_utm = LineString([tuple(start_xy), tuple(end_xy)])
        nodes = self._place_poles_along_route(straight_line_utm, roads_gdf)
        return nodes, "straight_line_fallback"

    def _resample_line_utm(
        self, line_utm: LineString, interval_m: float
    ) -> List[Tuple[float, float]]:
        """UTM çizgisini eşit `interval_m` aralıklarla örnekleyip nokta
        listesi üretir (çizginin kendi köşeleri örnekleme dışında da
        işin doğası gereği aradaki noktalarla örtük olarak yakalanır).

        Bu, kullanıcının serbest elle çizdiği (dolayısıyla köşe sayısı
        ve şekli belirsiz/gürültülü olabilen) bir hattı, yol ağı ile
        eşleştirme için yeterince sık "GPS iz noktası" gibi örneklenmiş
        bir noktalar dizisine çevirmek için kullanılır.

        Args:
            line_utm: Örneklenecek UTM koordinatlı çizgi.
            interval_m: Örnekler arası hedef mesafe (metre).

        Returns:
            (x, y) UTM koordinat tuple'larından oluşan liste.
        """
        length = line_utm.length
        if length == 0:
            return list(line_utm.coords)

        distances = list(np.arange(0.0, length, interval_m))
        if not distances or distances[-1] < length:
            distances.append(length)

        points = [line_utm.interpolate(d) for d in distances]
        return [(p.x, p.y) for p in points]

    def _sketch_points_to_utm_line(
        self, sketch_points: List[GeoPoint]
    ) -> Optional[LineString]:
        """Kullanıcının kalemle çizdiği ham kroki noktalarını, bitişik/
        tekrarlı ardışık noktaları sadeleştirerek bir UTM LineString'e
        çevirir.

        Bu, hem map-matching (`_match_sketch_to_road_graph`) hem de
        doğrudan-çizim modu (`_compute_virtual_nodes` içindeki
        `direct_line_mode` dalı) tarafından ortak olarak kullanılır —
        çizim aracının çift-tıklama ile bitirilen son noktada bıraktığı
        neredeyse-çakışık fazladan vertex'i (< 0.5m) her iki yolda da
        aynı şekilde temizlemek için.

        Args:
            sketch_points: Kroki hattının WGS84 noktaları, A'dan B'ye.

        Returns:
            Sadeleştirilmiş UTM LineString, ya da anlamlı bir hat
            oluşmazsa (örn. tüm noktalar birbirine çok yakınsa) None.
        """
        if len(sketch_points) < 2:
            return None

        sketch_coords_utm = [self.to_utm.transform(*p.as_xy()) for p in sketch_points]

        deduped_coords_utm: List[Tuple[float, float]] = [sketch_coords_utm[0]]
        for x, y in sketch_coords_utm[1:]:
            last_x, last_y = deduped_coords_utm[-1]
            if math.hypot(x - last_x, y - last_y) >= 0.5:
                deduped_coords_utm.append((x, y))

        if len(deduped_coords_utm) < 2:
            logger.warning("Kroki noktaları birbirine çok yakın; anlamlı bir çizgi oluşmadı.")
            return None

        return LineString(deduped_coords_utm)

    def _match_sketch_to_road_graph(
        self, road_graph: nx.MultiDiGraph, sketch_points: List[GeoPoint]
    ) -> Optional[LineString]:
        """Kullanıcının haritada kalemle serbest çizdiği (eğik/yamuk
        olabilen) kroki hattını, gerçek OSM yol ağına "harita eşleştirme"
        (map-matching) mantığıyla oturtur.

        Yöntem: elle çizilen çizgi genelde düzensiz (fazla/eksik köşe,
        titrek el hareketi vb.) olduğundan, önce çizgi eşit aralıklarla
        yoğun biçimde örneklenir (`_resample_line_utm`). Her örnek nokta
        yol ağının en yakın düğümüne "yapıştırılır" (snap); ardışık
        aynı düğümler sadeleştirilir. Sonra bu düğüm dizisi bir dizi
        "ara hedef" (waypoint) gibi ele alınır ve her ardışık waypoint
        çifti arasında en kısa yol hesaplanıp uç uca eklenir.

        Bu sayede, çizim tam olarak yol üzerinde olmasa (eğik/kaymış
        olsa) bile, nihai rota kullanıcının "anlatmak istediği" gerçek
        sokak dizisini takip eder — çünkü yoğun örnekleme, çizginin
        genel şeklini/yönünü (hangi sokaktan geçildiğini) yol ağına
        aktarmaya yeter.

        Args:
            road_graph: `_fetch_road_graph` ile elde edilen graf.
            sketch_points: Kroki hattının WGS84 noktaları.

        Returns:
            Eşleştirilmiş rotayı temsil eden UTM koordinatlı LineString,
            ya da eşleştirme hiçbir şekilde başarılamazsa None (bu
            durumda çağıran taraf düz hatta geri döner).
        """
        # Çizim aracı, son noktayı bitirmek için çift tıklama gerektirir;
        # bu genelde aynı (ya da bir-iki piksel kaymış) koordinatta fazladan
        # bir vertex bırakır. Böyle bitişik/tekrarlı ardışık noktaları
        # (< 0.5m) sadeleştiriyoruz, aksi halde sıfır uzunluklu son
        # segment örneklemeyi ve eşleştirmeyi bozabilir.
        sketch_line_utm = self._sketch_points_to_utm_line(sketch_points)
        if sketch_line_utm is None:
            return None

        # Kullanıcının çizdiği köşeleri kaçırmamak için node_spacing_m'den
        # bağımsız, sabitçe sık bir örnekleme aralığı kullanılır (10-25m).
        match_interval = max(10.0, min(self.node_spacing_m, 25.0))
        waypoints_utm = self._resample_line_utm(sketch_line_utm, match_interval)

        waypoints_lonlat = [self.to_wgs84.transform(x, y) for x, y in waypoints_utm]
        xs = [lon for lon, _lat in waypoints_lonlat]
        ys = [lat for _lon, lat in waypoints_lonlat]

        try:
            nearest = ox.distance.nearest_nodes(road_graph, X=xs, Y=ys)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Kroki noktaları yol ağına yapıştırılamadı (snap): %s", exc)
            return None

        # Ardışık aynı düğümleri sadeleştir (yoğun örnekleme aynı yol
        # segmentine düşen birçok noktayı aynı düğüme yapıştırabilir).
        waypoint_nodes: List[int] = []
        for node_id in nearest:
            if not waypoint_nodes or waypoint_nodes[-1] != node_id:
                waypoint_nodes.append(node_id)

        if len(waypoint_nodes) < 2:
            logger.warning(
                "Kroki, yol ağında tek bir düğüme yapıştı; anlamlı bir "
                "rota oluşturulamadı."
            )
            return None

        # Rota, yönsüz (undirected) bir graf üzerinde hesaplanır: tek
        # yönlü sokaklar directed kenar olarak modellendiği için, kroki
        # bir noktada tek yönlü akışın tersine gitmeyi gerektirirse
        # (örn. bir sokakta gidip aynı/paralel sokaktan geri gelen bir
        # çizim) yönlü grafikte rota bulunamaz. Direk yerleşimi için
        # araç yönü zaten anlamsızdır.
        route_graph = road_graph.to_undirected()

        # ÖNEMLİ: rota her zaman "şu ana kadar ulaşılan gerçek son düğüm"
        # (current_node) üzerinden bir sonraki waypoint'e aranır — orijinal
        # waypoint çiftleri (i, i+1) üzerinden DEĞİL. Aksi halde bir ara
        # segment rotasız kalırsa (örn. bağlantısız bir yol parçası), bir
        # sonraki deneme rotanın gerçek ucundan değil "atlanan" waypoint'ten
        # başlardı; bu da iki bağlantısız düğüm dizisinin uç uca eklenip
        # aralarında görünmez bir "ışınlanma" (kopuk/hayalet) segmenti
        # oluşmasına ve nihai hattın büyük ölçüde kısalmasına yol açardı.
        current_node = waypoint_nodes[0]
        full_route_nodes: List[int] = [current_node]
        skipped_waypoints = 0

        for target in waypoint_nodes[1:]:
            if target == current_node:
                continue
            try:
                sub_path = nx.shortest_path(route_graph, current_node, target, weight="length")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # Bu waypoint şu anki uçtan ulaşılamaz durumda (örn.
                # krokinin bir kısmı yol ağı dışına taştı); bu waypoint'i
                # atla ama current_node'u DEĞİŞTİRME — bir sonraki
                # waypoint'e yine rotanın gerçek son noktasından
                # ulaşmayı dene. Böylece süreklilik hiç bozulmaz.
                skipped_waypoints += 1
                continue
            # sub_path[0] zaten current_node'a eşit olduğundan tekrar
            # eklenmiyor.
            full_route_nodes.extend(sub_path[1:])
            current_node = target

        if len(full_route_nodes) < 2:
            logger.warning("Kroki hattı yol ağı üzerinde bir rotaya dönüştürülemedi.")
            return None

        if skipped_waypoints:
            logger.warning(
                "Kroki eşleştirmesinde %d waypoint, rotanın o anki ucundan "
                "ulaşılamadığı için atlandı; süreklilik korunarak devam "
                "edildi.",
                skipped_waypoints,
            )

        coords_utm: List[Tuple[float, float]] = []
        for node_id in full_route_nodes:
            node_data = road_graph.nodes[node_id]
            x, y = self.to_utm.transform(node_data["x"], node_data["y"])
            coords_utm.append((x, y))

        logger.info(
            "Kroki -> yol ağı eşleştirmesi tamamlandı | %d kroki örnek "
            "noktası -> %d benzersiz waypoint -> %d rota düğümü",
            len(waypoints_utm), len(waypoint_nodes), len(coords_utm),
        )
        return LineString(coords_utm)

    def _fetch_road_graph(self, corridor_polygon: Polygon) -> Optional[nx.MultiDiGraph]:
        """Koridor poligonu içindeki OSM yol ağını, rota hesaplamaya
        uygun bir networkx grafı (düğüm+kenar bağlantı bilgisiyle)
        olarak çeker.

        Not: Bu, `_fetch_osm_layer` ile çekilen `roads_gdf`'den farklıdır
        — `roads_gdf` sadece çizim/maskeleme amaçlı düz bir geometri
        listesidir ve yol bağlantı topolojisini (hangi yol hangisine
        bağlanıyor) içermez. Rota (en kısa yol) hesaplamak için gerçek
        bir graf yapısı (`ox.graph_from_polygon`) gerekir.

        `network_type="all"` kullanılır (sadece "drive" değil): kroki,
        araç yolu olmayan alanlardan (yaya yolu, avlu, park içi patika,
        servis yolu vb.) geçebilir. Sadece "drive" kullanılsaydı, bu tür
        bölgelerdeki kroki noktalarının çoğu en yakın (uzaktaki) tek bir
        sokağa yapışıp aynı düğümde toplanır; bu da rotanın pratikte
        sadece başlangıç-bitiş arası kısa bir sokak parçasına inmesine
        (ara noktaların "yutulmasına") yol açar.

        Args:
            corridor_polygon: Yol ağının çekileceği alan (WGS84).

        Returns:
            networkx.MultiDiGraph, ya da veri/bağlantı hatası durumunda
            None (bu durumda çağıran taraf düz hatta geri döner).
        """
        try:
            graph = ox.graph_from_polygon(
                corridor_polygon,
                network_type="all",
                simplify=True,
                retain_all=True,
            )
            if graph.number_of_nodes() == 0:
                logger.warning("Koridor içinde OSM yol ağı bulunamadı.")
                return None
            return graph
        except Exception as exc:  # noqa: BLE001
            logger.warning("Yol ağı (graph) çekilirken sorun oluştu: %s", exc)
            return None

    def _compute_road_route_utm(self, road_graph: nx.MultiDiGraph) -> Optional[LineString]:
        """A ve B'yi yol ağının en yakın düğümlerine "yapıştırıp" (snap)
        aralarındaki en kısa mesafeli rotayı hesaplar.

        Not: Rota, yönsüz (undirected) bir graf üzerinde hesaplanır.
        OSM sürüş (drive) grafiği yönlüdür (tek yönlü sokaklar directed
        kenar olarak modellenir); ama direk yerleşimi için yön önemsizdir
        — bir direk, tek yönlü bir sokakta hangi yönde "araç akışı" olursa
        olsun aynı fiziksel yerde durur. Yönlü grafik kullanılsaydı, A-B
        arası rota tek yönlü bir sokağın tersine gitmeyi gerektirdiğinde
        (örn. bir döngü/geri dönüş şekli) rota bulunamayabilirdi.

        Args:
            road_graph: `_fetch_road_graph` ile elde edilen graf.

        Returns:
            Rotayı temsil eden UTM koordinatlı LineString, ya da rota
            bulunamazsa None.
        """
        try:
            orig_node = ox.distance.nearest_nodes(
                road_graph, X=self.start.lon, Y=self.start.lat
            )
            dest_node = ox.distance.nearest_nodes(
                road_graph, X=self.end.lon, Y=self.end.lat
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("A/B noktaları yol ağına yapıştırılamadı (snap): %s", exc)
            return None

        route_graph = road_graph.to_undirected()

        try:
            route_node_ids = nx.shortest_path(
                route_graph, orig_node, dest_node, weight="length"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
            logger.warning(
                "A ve B arasında yol ağı üzerinden bir rota bulunamadı "
                "(muhtemelen bağlantısız yol parçaları): %s", exc
            )
            return None

        if len(route_node_ids) < 2:
            return None

        coords_utm: List[Tuple[float, float]] = []
        for node_id in route_node_ids:
            node_data = road_graph.nodes[node_id]
            x, y = self.to_utm.transform(node_data["x"], node_data["y"])
            coords_utm.append((x, y))

        logger.info(
            "Yol ağı üzerinden rota bulundu | %d graf düğümü kullanıldı",
            len(coords_utm),
        )
        return LineString(coords_utm)

    # Bir düğüm noktasının "yol kenarına yapıştırılmış" sayılabilmesi için
    # en yakın OSM yoluna olan mesafesinin üst sınırı (metre). Koridor
    # tamponu geniş olduğunda, krokinin bir tarlanın/ormanın içinden
    # geçtiği kısa bir kesitte bile `roads_gdf` içinde uzakta bir yol
    # bulunabilir; bu sınırın ötesindeki yollar o düğüm için "yakında yol
    # yok" kabul edilir — aksi halde direk, çizilen hattan onlarca/
    # yüzlerce metre uzağa "ışınlanabilir".
    MAX_ROAD_SNAP_DISTANCE_M = 25.0

    # Art arda iki düğüm yola bağımsız olarak yapıştırıldığında, aralarındaki
    # gerçek mesafenin rotadaki beklenen açıklığa (`span_length_m`) oranı bu
    # aralığın dışına çıkarsa, yapıştırma "tutarsız" kabul edilir (bkz.
    # `_place_poles_along_route` adım 4) — çünkü bu, iki düğümün aynı yol
    # noktasına yakınsayıp kümelendiğini (çok küçük oran) ya da rotanın
    # izlediği basitleştirilmiş hattın gerçek yoldan önemli ölçüde farklı
    # kıvrıldığını (çok büyük oran) gösterir; her iki durumda da direk
    # dizilimi düzensiz/aralıklı görünür.
    SNAP_CONSISTENCY_RATIO_MIN = 0.3
    SNAP_CONSISTENCY_RATIO_MAX = 3.0

    def _prepare_road_lines_utm(
        self, roads_gdf: gpd.GeoDataFrame
    ) -> List[LineString]:
        """`roads_gdf` (WGS84) içindeki geometrileri UTM'e çevirip yalnızca
        çizgisel (LineString) yol geometrilerinin düz bir listesini üretir.

        OSMnx `features_from_polygon`, bazen alansal (örn. meydan
        poligonu) ya da nokta (örn. bariyer düğümü) geometrileri de
        döndürebilir; bunlar kenar-yapıştırma (edge snapping) için
        anlamsız olduğundan filtrelenir. `MultiLineString`'ler tekil
        `LineString` parçalarına ayrıştırılır (explode) — böylece her
        parça için ayrı ayrı en yakın nokta/yön hesabı yapılabilir.

        Args:
            roads_gdf: `_fetch_osm_layer` ile çekilen ham yol katmanı.

        Returns:
            UTM koordinatlarında LineString listesi (yol yoksa boş liste).
        """
        if roads_gdf is None or roads_gdf.empty:
            return []

        try:
            roads_utm = roads_gdf.to_crs(self.utm_epsg)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Yol katmanı UTM'e çevrilirken sorun oluştu, kenar "
                "yapıştırma bu düğümler için atlanacak: %s", exc
            )
            return []

        lines: List[LineString] = []
        for geom in roads_utm.geometry:
            if geom is None or geom.is_empty:
                continue
            if isinstance(geom, LineString):
                lines.append(geom)
            elif isinstance(geom, MultiLineString):
                lines.extend(list(geom.geoms))
            # Polygon/Point vb. diğer (nadir) geometri tipleri atlanır.

        return lines

    def _route_tangent_at(
        self, route_line_utm: LineString, point_utm: Point
    ) -> Tuple[float, float]:
        """Rotanın, verilen noktaya en yakın konumundaki yerel yön
        (tanjant) birim vektörünü sayısal türevle hesaplar.

        Args:
            route_line_utm: Orijinal (offsetlenmemiş) rota, UTM.
            point_utm: Rota üzerinde (ya da tam üzerinde kabul edilecek
                kadar yakınında) bir nokta.

        Returns:
            (dx, dy) birim tanjant vektörü. Dejenere durumlarda (sıfır
            uzunluklu rota vb.) varsayılan olarak (1.0, 0.0) döner.
        """
        s = route_line_utm.project(point_utm)
        delta = 0.5
        s_back = max(0.0, s - delta)
        s_fwd = min(route_line_utm.length, s + delta)
        if s_fwd - s_back < 1e-6:
            return (1.0, 0.0)

        p_back = route_line_utm.interpolate(s_back)
        p_fwd = route_line_utm.interpolate(s_fwd)
        dx, dy = p_fwd.x - p_back.x, p_fwd.y - p_back.y
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return (1.0, 0.0)
        return (dx / length, dy / length)

    def _snap_point_to_road_edge(
        self,
        point_utm: Point,
        road_lines: List[LineString],
        road_tree: "STRtree",
        route_perp_unit: Tuple[float, float],
    ) -> Tuple[Point, bool]:
        """Tek bir aday direk noktasını en yakın gerçek yolun KENARINA
        yapıştırır.

        Yolun OSM'de hangi yönde çizildiği (digitization direction)
        bilinemediğinden, yolun iki olası dikey (normal) yönünden hangisi
        seçilecek diye `route_perp_unit` (rotanın o noktadaki yerel
        yönüne dik, `offset_side` tarafını gösteren birim vektör) ile
        karşılaştırılır — böylece direkler, yolun rastgele bir yönde
        sıçramak yerine, rotanın tutarlı bir tarafında (örn. hep aynı
        kaldırım kenarında) kalır.

        Args:
            point_utm: Rotanın orijinal (kaydırılmamış) hattı üzerindeki
                aday direk noktası (UTM).
            road_lines: `_prepare_road_lines_utm` çıktısı.
            road_tree: `road_lines` üzerine kurulmuş STRtree (hızlı en
                yakın yol araması için).
            route_perp_unit: Rotanın bu noktadaki yerel dikeyinin,
                `offset_side` tarafını gösteren birim vektörü (x, y).

        Returns:
            (yeni_nokta, yapıştırıldı_mı) tuple'ı. Yakında
            (`MAX_ROAD_SNAP_DISTANCE_M` içinde) bir yol bulunamazsa,
            `yapıştırıldı_mı=False` ile birlikte rotanın kendi yerel
            dikeyine göre sabit offsetlenmiş nokta döner (güvenli
            fallback — eski davranışla aynı).
        """
        fallback_point = Point(
            point_utm.x + route_perp_unit[0] * self.pole_offset_m,
            point_utm.y + route_perp_unit[1] * self.pole_offset_m,
        )

        if not road_lines or road_tree is None:
            return fallback_point, False

        nearest_idx = road_tree.nearest(point_utm)
        nearest_road = road_lines[int(nearest_idx)]

        if nearest_road.distance(point_utm) > self.MAX_ROAD_SNAP_DISTANCE_M:
            return fallback_point, False

        s = nearest_road.project(point_utm)
        nearest_on_road = nearest_road.interpolate(s)

        # Yol segmentinin bu noktadaki yerel yönünü (tanjant), küçük bir
        # ileri/geri adımla (delta) sayısal türev alarak buluyoruz.
        delta = 0.5
        s_back = max(0.0, s - delta)
        s_fwd = min(nearest_road.length, s + delta)
        if s_fwd - s_back < 1e-6:
            return fallback_point, False

        p_back = nearest_road.interpolate(s_back)
        p_fwd = nearest_road.interpolate(s_fwd)
        road_dx, road_dy = p_fwd.x - p_back.x, p_fwd.y - p_back.y
        road_len = math.hypot(road_dx, road_dy)
        if road_len < 1e-9:
            return fallback_point, False
        road_dx, road_dy = road_dx / road_len, road_dy / road_len

        # Yolun iki olası dikey (normal) yönünden, rotanın istenen
        # tarafına (route_perp_unit) daha yakın olanı seçilir.
        perp_a = (-road_dy, road_dx)
        perp_b = (road_dy, -road_dx)
        dot_a = perp_a[0] * route_perp_unit[0] + perp_a[1] * route_perp_unit[1]
        dot_b = perp_b[0] * route_perp_unit[0] + perp_b[1] * route_perp_unit[1]
        chosen_perp = perp_a if dot_a >= dot_b else perp_b

        snapped_point = Point(
            nearest_on_road.x + chosen_perp[0] * self.pole_offset_m,
            nearest_on_road.y + chosen_perp[1] * self.pole_offset_m,
        )
        return snapped_point, True

    def _place_poles_along_route(
        self,
        route_line_utm: LineString,
        roads_gdf: gpd.GeoDataFrame,
    ) -> List[VirtualNode]:
        """Direk noktalarını, rotayı HİÇ değiştirmeden/kısaltmadan üretir
        ve her birini ayrı ayrı en yakın gerçek yolun kenarına yapıştırır.

        Önceki yaklaşım (bkz. eski `_offset_route_line`), tüm rota
        çizgisini tek seferde sabit bir mesafeyle (`pole_offset_m`)
        kaydırıyordu. Bu, özellikle kullanıcının serbest çizdiği krokinin
        gerçek yoldan saptığı kısımlarında, direği yolun gerçek kenarına
        değil "krokinin X metre yanına" koyuyordu.

        Bu metot bunun yerine:
          1. Direk noktalarını DOĞRUDAN orijinal (kaydırılmamış) rota
             üzerinde, `node_spacing_m` aralıklarla üretir — rota hiçbir
             zaman kısaltılmaz/değiştirilmez; köşe noktaları korunur.
          2. Her nokta için `roads_gdf` içindeki en yakın gerçek yol
             segmentini bulur, noktayı o segmentin üzerine izdüşürür
             (nearest point) ve segmentin yerel yönüne dik olarak
             `pole_offset_m` kadar kaydırarak yolun KENARINA yerleştirir.
          3. Yakınında (bkz. `MAX_ROAD_SNAP_DISTANCE_M`) hiçbir yol
             bulunamazsa, o nokta rotanın kendi yerel dikeyine göre sabit
             offsetlenir (güvenli fallback) — direk asla çizilen hattan
             anlamsız derecede uzağa sıçramaz.
          4. Her düğüm BAĞIMSIZ olarak en yakın yola yapıştırıldığı için,
             gerçek yol rotanın izlediği hayali çizgiden (basitleştirilmiş
             graf kirişinden) farklı bükülüyorsa, art arda iki düğüm
             yola izdüşürüldüğünde birbirine anormal derecede yakınlaşıp
             ("kümelenme") ya da anormal derecede uzaklaşabilir
             ("boşluk/eksik nokta görüntüsü") — çünkü en yakın nokta
             projeksiyonu, orijinal rota üzerindeki düzenli aralığı
             (arc-length) yol üzerinde otomatik olarak korumaz. Bunu
             önlemek için, art arda iki yapıştırılmış nokta arasındaki
             gerçek mesafe, rotadaki beklenen açıklıktan
             (`span_length_m`) çok sapıyorsa (bkz.
             `SNAP_CONSISTENCY_RATIO_MIN/MAX`), o düğümün yol yapıştırması
             güvenilmez sayılıp adım 3'teki güvenli fallback'e dönülür.

        Args:
            route_line_utm: Orijinal (offsetlenmemiş) rota, UTM
                koordinatlarında (kroki/eşleştirme/en kısa yol/düz hat).
            roads_gdf: `_fetch_osm_layer` ile çekilen ham yol katmanı
                (WGS84).

        Returns:
            node_id sırasına göre sıralı, her biri en yakın yol kenarına
            yapıştırılmış (ya da yakında yol yoksa/yapıştırma tutarsızsa
            güvenli fallback ile offsetlenmiş) VirtualNode listesi.
        """
        raw_nodes = self._generate_nodes_along_line(route_line_utm)

        if self.pole_offset_m == 0:
            return raw_nodes

        road_lines = self._prepare_road_lines_utm(roads_gdf)
        road_tree = STRtree(road_lines) if road_lines else None

        # Shapely offset_curve sözleşmesiyle tutarlı olacak şekilde:
        # "left" -> tanjantın soluna (+90°), "right" -> sağına (-90°).
        signed_side = 1.0 if self.offset_side == "left" else -1.0

        snapped_count = 0
        rejected_count = 0
        placed_nodes: List[VirtualNode] = []
        prev_placed_point: Optional[Point] = None

        for node in raw_nodes:
            x, y = self.to_utm.transform(node.point.lon, node.point.lat)
            point_utm = Point(x, y)

            tangent = self._route_tangent_at(route_line_utm, point_utm)
            route_perp = (-tangent[1] * signed_side, tangent[0] * signed_side)
            fallback_point = Point(
                point_utm.x + route_perp[0] * self.pole_offset_m,
                point_utm.y + route_perp[1] * self.pole_offset_m,
            )

            new_point, did_snap = self._snap_point_to_road_edge(
                point_utm, road_lines, road_tree, route_perp
            )

            if did_snap and prev_placed_point is not None and node.span_length_m > 1e-6:
                actual_gap = new_point.distance(prev_placed_point)
                gap_ratio = actual_gap / node.span_length_m
                if (
                    gap_ratio < self.SNAP_CONSISTENCY_RATIO_MIN
                    or gap_ratio > self.SNAP_CONSISTENCY_RATIO_MAX
                ):
                    # Bu düğümün yol yapıştırması, rotadaki beklenen
                    # açıklıkla tutarsız (kümelenme ya da anormal
                    # sıçrama) — güvenilmez kabul edilip rota-dikeyine
                    # sabit offsetlenmiş güvenli noktaya dönülüyor.
                    new_point = fallback_point
                    did_snap = False
                    rejected_count += 1

            if did_snap:
                snapped_count += 1
            prev_placed_point = new_point

            lon, lat = self.to_wgs84.transform(new_point.x, new_point.y)
            placed_nodes.append(
                VirtualNode(
                    node_id=node.node_id,
                    point=GeoPoint(lat=lat, lon=lon),
                    cumulative_distance_m=node.cumulative_distance_m,
                    span_length_m=node.span_length_m,
                    is_corner=node.is_corner,
                    deflection_angle_deg=node.deflection_angle_deg,
                    pole_type=node.pole_type,
                )

            )

        logger.info(
            "Yol kenarına yapıştırma tamamlandı | %d/%d düğüm gerçek yol "
            "kenarına yapıştırıldı | %d düğüm tutarsız yapıştırma (kümelenme/"
            "sıçrama) tespit edilip güvenli fallback'e döndürüldü.",
            snapped_count, len(placed_nodes), rejected_count,
        )

        return placed_nodes

    def _generate_nodes_along_line(self, line_utm: LineString) -> List[VirtualNode]:
        """Verilen (offsetlenmiş) rota çizgisi üzerinde sanal direk
        düğümleri üretir.

        Stratejı: çizginin kendi vertex'leri (yol ağının köşe/dönüş
        noktaları) HER ZAMAN birer düğüm olarak korunur ve `is_corner`
        olarak işaretlenir — bunlar mühendislik açısından atlanamaz
        (kavşak/açı direği gerektiren) noktalardır. İki vertex arasındaki
        mesafe `node_spacing_m`'i aşıyorsa, aradaki boşluk maksimum
        açıklık kısıtını ihlal etmeyecek şekilde eşit aralıklı ara
        düğümlerle bölünür.

        Args:
            line_utm: UTM koordinatlarında rota çizgisi (offsetlenmiş
                ya da düz hat fallback'i olabilir).

        Returns:
            node_id sırasına göre (0 = A ucu) sıralanmış VirtualNode
            listesi.
        """
        coords = list(line_utm.coords)
        if len(coords) < 2:
            raise ValueError("Rota en az 2 noktadan oluşmalıdır.")

        nodes: List[VirtualNode] = []
        node_id = 0
        cumulative_distance = 0.0

        first_lon, first_lat = self.to_wgs84.transform(*coords[0])
        nodes.append(
            VirtualNode(
                node_id=node_id,
                point=GeoPoint(lat=first_lat, lon=first_lon),
                cumulative_distance_m=0.0,
                span_length_m=0.0,  # A ucu: önceki direk yok.
                is_corner=True,
                pole_type="nihayet",
            )
        )
        node_id += 1
        prev_cumulative_distance = 0.0

        for i in range(len(coords) - 1):
            p1 = np.array(coords[i])
            p2 = np.array(coords[i + 1])
            segment_length = float(np.linalg.norm(p2 - p1))

            if segment_length == 0:
                continue

            num_subsegments = max(1, math.ceil(segment_length / self.node_spacing_m))

            for k in range(1, num_subsegments + 1):
                t = k / num_subsegments
                point_xy = p1 + t * (p2 - p1)
                lon, lat = self.to_wgs84.transform(point_xy[0], point_xy[1])
                cumulative_distance_at_point = cumulative_distance + t * segment_length

                # Bir alt-segmentin son noktası (k == num_subsegments),
                # orijinal rotanın gerçek bir vertex'ine (köşesine)
                # denk gelir; bu yüzden köşe olarak işaretlenir.
                is_corner_point = (k == num_subsegments)
                is_last_vertex = is_corner_point and (i == len(coords) - 2)

                deflection_deg = 0.0
                pole_type = "ara"
                if is_last_vertex:
                    # Rotanın B ucu: hattın nihayet (son) direği.
                    pole_type = "nihayet"
                elif is_corner_point and (i + 2) < len(coords):
                    # Gelen segment (p1->p2) ile giden segment
                    # (coords[i+1]->coords[i+2]) arasındaki sapma açısı.
                    p3 = np.array(coords[i + 2])
                    deflection_deg = self._deflection_angle_deg(p2 - p1, p3 - p2)
                    pole_type = (
                        "açı"
                        if deflection_deg >= self.ANGLE_DEFLECTION_THRESHOLD_DEG
                        else "ara"
                    )

                nodes.append(
                    VirtualNode(
                        node_id=node_id,
                        point=GeoPoint(lat=lat, lon=lon),
                        cumulative_distance_m=round(cumulative_distance_at_point, 3),
                        span_length_m=round(
                            cumulative_distance_at_point - prev_cumulative_distance, 3
                        ),
                        is_corner=is_corner_point,
                        deflection_angle_deg=round(deflection_deg, 2),
                        pole_type=pole_type,
                    )
                )
                node_id += 1
                prev_cumulative_distance = cumulative_distance_at_point

            cumulative_distance += segment_length

        num_corners = sum(1 for n in nodes if n.is_corner)
        num_angle_poles = sum(1 for n in nodes if n.pole_type == "açı")
        num_over_limit = sum(
            1 for n in nodes if n.span_length_m > self.max_span_m + 1e-6
        )
        logger.info(
            "Toplam rota mesafesi: %.2f m | Üretilen düğüm sayısı: %d "
            "(köşe/kavşak: %d, açı direği: %d) | hedef aralık: %.1fm | "
            "gerilim sınıfı: %s (azami açıklık: %.1fm) | açıklığı aşan "
            "düğüm: %d",
            cumulative_distance, len(nodes), num_corners, num_angle_poles,
            self.node_spacing_m, self.voltage_class, self.max_span_m,
            num_over_limit,
        )
        if num_over_limit:
            logger.warning(
                "%d düğümün açıklığı (span) %s sınıfı için yaklaşık azami "
                "değeri (%.1fm) aşıyor — bu genellikle rota köşelerinde "
                "(vertex'ler arası mesafe zorunlu olarak korunduğu için) "
                "oluşur; ilgili direkler saha/projelendirme aşamasında "
                "ayrıca değerlendirilmelidir.",
                num_over_limit, self.voltage_class, self.max_span_m,
            )

        return nodes

    @staticmethod
    def _deflection_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
        """İki ardışık segment vektörü arasındaki sapma (deflection)
        açısını derece cinsinden hesaplar.

        0° = giden segment, gelen segmentin tam devamı (düz hat).
        180° = giden segment, gelen segmentin tam tersi yöne dönüyor.

        Args:
            v1: Gelen segment vektörü (p1 -> p2), UTM düzleminde (x, y).
            v2: Giden segment vektörü (p2 -> p3), UTM düzleminde (x, y).

        Returns:
            0-180 derece arası sapma açısı. Vektörlerden biri sıfır
            uzunluktaysa (çakışık noktalar) 0.0 döner.
        """
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            return 0.0
        cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return float(math.degrees(math.acos(cos_theta)))


# NOT: Bu dosya bilinçli olarak saf bir Python modülüdür (backend/servis
# katmanı) ve içinde Streamlit arayüz kodu YOKTUR. Bu yüzden Streamlit'in
# "main module" (ana dosya) olarak DOĞRUDAN deploy edilmemelidir — deploy
# edilirse ekranda hiçbir arayüz elemanı görünmez (boş sayfa) ve olası bir
# ağ gecikmesinde uygulama asılı kalabilir.
#
# Bu modülü kullanan gerçek Streamlit uygulaması için bkz. app.py.
