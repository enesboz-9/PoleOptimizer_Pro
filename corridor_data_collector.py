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
    """
    node_id: int
    point: GeoPoint
    cumulative_distance_m: float
    is_feasible: Optional[bool] = None
    elevation_m: Optional[float] = None
    blocking_reason: Optional[str] = None
    is_corner: bool = False


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
                (tercih edilen, en gerçekçi durum).
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

    def __init__(
        self,
        start: GeoPoint,
        end: GeoPoint,
        corridor_buffer_m: float = 150.0,
        node_spacing_m: float = 30.0,
        pole_offset_m: float = 5.0,
        offset_side: str = "right",
    ) -> None:
        if corridor_buffer_m <= 0:
            raise ValueError("corridor_buffer_m sıfırdan büyük olmalıdır.")
        if node_spacing_m <= 0:
            raise ValueError("node_spacing_m sıfırdan büyük olmalıdır.")
        if pole_offset_m < 0:
            raise ValueError("pole_offset_m negatif olamaz.")
        if offset_side not in ("left", "right"):
            raise ValueError("offset_side yalnızca 'left' veya 'right' olabilir.")

        self.start = start
        self.end = end
        self.corridor_buffer_m = corridor_buffer_m
        self.node_spacing_m = node_spacing_m
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
            "buffer=%.1fm | spacing=%.1fm | pole_offset=%.1fm (%s)",
            start.as_tuple(), end.as_tuple(), self.utm_epsg,
            corridor_buffer_m, node_spacing_m, pole_offset_m, offset_side,
        )

    # -------------------------------------------------------------- #
    # Public API
    # -------------------------------------------------------------- #

    def run(self, sketch_points: Optional[List[GeoPoint]] = None) -> CorridorData:
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

        # --- Rota hesaplama: kroki varsa eşleştir, yoksa en kısa yol,
        # ikisi de olmazsa düz hat ---
        virtual_nodes, route_source = self._compute_virtual_nodes(
            corridor_polygon_wgs84, sketch_points
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
        sketch_points: Optional[List[GeoPoint]] = None,
    ) -> Tuple[List[VirtualNode], str]:
        """Sanal direk düğümlerini üretmenin ana orkestrasyon metodu.

        `sketch_points` verilmişse önce kroki hattını gerçek yol ağına
        eşleştirmeyi (map-matching) dener. Verilmemişse (ya da eşleştirme
        başarısız olursa) A ve B arasında yol ağı üzerinden en kısa rotayı
        bulmayı dener (yollardan geçen, köşelerde dönen, kaldırım kenarına
        offsetlenmiş direk dizisi). Bu da başarısız olursa (yol ağı
        bulunamadı, uç noktalar bir yola yakın değil, bağlantısız graf
        vb.) düz hatta geri döner ve bunu log'da açıkça belirtir.

        Args:
            corridor_polygon: OSM yol ağının çekileceği alan (WGS84).
            sketch_points: Varsa, kullanıcının kalemle çizdiği kroki
                hattının WGS84 noktaları (A'dan B'ye sırayla).

        Returns:
            (virtual_nodes, route_source) tuple'ı. route_source,
            "sketch_matched", "road_snapped" veya
            "straight_line_fallback" değerlerinden birini alır.
        """
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
                offset_line_utm = self._offset_route_line(route_line_utm)
                nodes = self._generate_nodes_along_line(offset_line_utm)
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
        nodes = self._generate_nodes_along_line(straight_line_utm)
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
        if len(sketch_points) < 2:
            return None

        sketch_coords_utm = [self.to_utm.transform(*p.as_xy()) for p in sketch_points]

        # Çizim aracı, son noktayı bitirmek için çift tıklama gerektirir;
        # bu genelde aynı (ya da bir-iki piksel kaymış) koordinatta fazladan
        # bir vertex bırakır. Böyle bitişik/tekrarlı ardışık noktaları
        # (< 0.5m) sadeleştiriyoruz, aksi halde sıfır uzunluklu son
        # segment örneklemeyi ve eşleştirmeyi bozabilir.
        deduped_coords_utm: List[Tuple[float, float]] = [sketch_coords_utm[0]]
        for x, y in sketch_coords_utm[1:]:
            last_x, last_y = deduped_coords_utm[-1]
            if math.hypot(x - last_x, y - last_y) >= 0.5:
                deduped_coords_utm.append((x, y))
        if len(deduped_coords_utm) < 2:
            logger.warning("Kroki noktaları birbirine çok yakın; anlamlı bir çizgi oluşmadı.")
            return None

        sketch_line_utm = LineString(deduped_coords_utm)

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

        full_route_nodes: List[int] = [waypoint_nodes[0]]
        skipped_segments = 0

        for i in range(len(waypoint_nodes) - 1):
            origin, dest = waypoint_nodes[i], waypoint_nodes[i + 1]
            try:
                sub_path = nx.shortest_path(route_graph, origin, dest, weight="length")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # Bu ara segment bağlantısız kalmış olabilir (örn. krokinin
                # bir kısmı yol ağı dışına taştı); atla ve bir sonraki
                # yakalanan noktadan devam et. Rotanın tamamını iptal
                # etmiyoruz çünkü kullanıcının çizdiği hattın büyük kısmı
                # hâlâ geçerli olabilir.
                skipped_segments += 1
                continue
            # sub_path[0] zaten full_route_nodes'un son elemanına eşit
            # (origin) olduğundan tekrar eklenmiyor.
            full_route_nodes.extend(sub_path[1:])

        if len(full_route_nodes) < 2:
            logger.warning("Kroki hattı yol ağı üzerinde bir rotaya dönüştürülemedi.")
            return None

        if skipped_segments:
            logger.warning(
                "Kroki eşleştirmesinde %d ara segment bağlantısız yol ağı "
                "nedeniyle atlandı; rota bir sonraki yakalanan noktadan "
                "devam etti.",
                skipped_segments,
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

    def _offset_route_line(self, route_line_utm: LineString) -> LineString:
        """Rota çizgisini, direklerin yolun ORTASINDA değil KENARINDA
        durması için `pole_offset_m` kadar yana kaydırır (offset).

        Kavşak/köşe noktalarında keskin offset artefaktları oluşmaması
        için mitre (gönye) birleşim stili kullanılır; bu, orijinal
        rotanın köşe noktalarını byüyük ölçüde korur.

        Args:
            route_line_utm: Yol ağından elde edilen orijinal (orta hat)
                rota, UTM koordinatlarında.

        Returns:
            Offsetlenmiş LineString. Offset başarısız olursa (örn.
            çok kısa/dejenere geometri), güvenli fallback olarak
            offsetsiz orijinal rota döner.
        """
        if self.pole_offset_m == 0:
            return route_line_utm

        # Shapely offset_curve sözleşmesi: pozitif mesafe, çizginin
        # A->B yönüne göre SOLUNA offsetler; negatif mesafe SAĞINA.
        signed_offset = self.pole_offset_m if self.offset_side == "left" else -self.pole_offset_m

        try:
            offset_geom = route_line_utm.offset_curve(signed_offset, join_style="mitre")
            if offset_geom.is_empty:
                raise ValueError("Offset sonucu boş geometri döndü.")
            if isinstance(offset_geom, MultiLineString):
                # Karmaşık/kendisiyle kesişen rotalarda offset birden
                # fazla parçaya bölünebilir; en uzun parçayı ana rota
                # olarak kabul ediyoruz.
                offset_geom = max(offset_geom.geoms, key=lambda g: g.length)
            return offset_geom
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Rota offsetlenirken sorun oluştu, offsetsiz orta hat "
                "kullanılacak: %s", exc
            )
            return route_line_utm

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
                is_corner=True,
            )
        )
        node_id += 1

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

                nodes.append(
                    VirtualNode(
                        node_id=node_id,
                        point=GeoPoint(lat=lat, lon=lon),
                        cumulative_distance_m=round(cumulative_distance_at_point, 3),
                        is_corner=is_corner_point,
                    )
                )
                node_id += 1

            cumulative_distance += segment_length

        num_corners = sum(1 for n in nodes if n.is_corner)
        logger.info(
            "Toplam rota mesafesi: %.2f m | Üretilen düğüm sayısı: %d "
            "(köşe/kavşak düğümü: %d) | hedef aralık: %.1fm",
            cumulative_distance, len(nodes), num_corners, self.node_spacing_m,
        )

        return nodes


# NOT: Bu dosya bilinçli olarak saf bir Python modülüdür (backend/servis
# katmanı) ve içinde Streamlit arayüz kodu YOKTUR. Bu yüzden Streamlit'in
# "main module" (ana dosya) olarak DOĞRUDAN deploy edilmemelidir — deploy
# edilirse ekranda hiçbir arayüz elemanı görünmez (boş sayfa) ve olası bir
# ağ gecikmesinde uygulama asılı kalabilir.
#
# Bu modülü kullanan gerçek Streamlit uygulaması için bkz. app.py.
