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
            "road_snapped" -> yol ağı üzerinden rota bulundu ve direkler
                yol kenarına offsetlendi (tercih edilen, gerçekçi durum).
            "straight_line_fallback" -> yol ağı bulunamadı/rota
                hesaplanamadı, A-B düz hattına geri dönüldü (bu durumda
                sonuçlar binaların/engellerin üzerinden geçebilir ve
                sadece kaba bir ön izleme olarak değerlendirilmelidir).
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

    def run(self) -> CorridorData:
        """Tüm veri toplama ve sanal düğüm üretim akışını çalıştırır.

        Returns:
            Bina, engel, yol katmanlarını, koridor poligonunu ve
            sanal düğüm listesini içeren CorridorData nesnesi.
        """
        corridor_polygon_wgs84 = self._build_corridor_polygon()

        buildings_gdf = self._fetch_osm_layer(
            corridor_polygon_wgs84, self.BUILDING_TAGS, layer_name="buildings"
        )
        obstacles_gdf = self._fetch_osm_layer(
            corridor_polygon_wgs84, self.OBSTACLE_TAGS, layer_name="obstacles"
        )
        roads_gdf = self._fetch_osm_layer(
            corridor_polygon_wgs84, self.ROAD_TAGS, layer_name="roads"
        )

        # --- Rota hesaplama: önce yol ağı üzerinden dene, olmazsa düz hat ---
        virtual_nodes, route_source = self._compute_virtual_nodes(corridor_polygon_wgs84)

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

    def _build_corridor_polygon(self) -> Polygon:
        """A-B hattı etrafında `corridor_buffer_m` genişliğinde tampon
        poligonu üretir. Bu poligon, OSM sorgularının sınırlarını
        (bounding region) belirler.

        İşlem hassasiyeti için UTM projeksiyonunda buffer uygulanır,
        ardından sonuç tekrar WGS84'e (EPSG:4326) dönüştürülür.

        Returns:
            WGS84 koordinat sisteminde koridor poligonu (Shapely Polygon).
        """
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
        self, corridor_polygon: Polygon
    ) -> Tuple[List[VirtualNode], str]:
        """Sanal direk düğümlerini üretmenin ana orkestrasyon metodu.

        Önce yol ağı üzerinden gerçek bir rota bulmayı dener (yollardan
        geçen, köşelerde dönen, kaldırım kenarına offsetlenmiş direk
        dizisi). Bu başarısız olursa (yol ağı bulunamadı, A/B bir yola
        yakın değil, bağlantısız graf vb.) A-B düz hattına geri döner
        ve bunu log'da açıkça belirtir.

        Args:
            corridor_polygon: OSM yol ağının çekileceği alan (WGS84).

        Returns:
            (virtual_nodes, route_source) tuple'ı. route_source,
            "road_snapped" veya "straight_line_fallback" değerini alır.
        """
        road_graph = self._fetch_road_graph(corridor_polygon)

        if road_graph is not None:
            route_line_utm = self._compute_road_route_utm(road_graph)
            if route_line_utm is not None:
                offset_line_utm = self._offset_route_line(route_line_utm)
                nodes = self._generate_nodes_along_line(offset_line_utm)
                return nodes, "road_snapped"

        logger.warning(
            "Yol ağı üzerinden geçerli bir rota bulunamadı; A-B düz hattına "
            "geri dönülüyor. UYARI: Bu durumda direk noktaları bina/engel "
            "üzerinden geçebilir, sonuç yalnızca kaba bir ön izlemedir."
        )
        start_xy = np.array(self.to_utm.transform(*self.start.as_xy()))
        end_xy = np.array(self.to_utm.transform(*self.end.as_xy()))
        straight_line_utm = LineString([tuple(start_xy), tuple(end_xy)])
        nodes = self._generate_nodes_along_line(straight_line_utm)
        return nodes, "straight_line_fallback"

    def _fetch_road_graph(self, corridor_polygon: Polygon) -> Optional[nx.MultiDiGraph]:
        """Koridor poligonu içindeki OSM yol ağını, rota hesaplamaya
        uygun bir networkx grafı (düğüm+kenar bağlantı bilgisiyle)
        olarak çeker.

        Not: Bu, `_fetch_osm_layer` ile çekilen `roads_gdf`'den farklıdır
        — `roads_gdf` sadece çizim/maskeleme amaçlı düz bir geometri
        listesidir ve yol bağlantı topolojisini (hangi yol hangisine
        bağlanıyor) içermez. Rota (en kısa yol) hesaplamak için gerçek
        bir graf yapısı (`ox.graph_from_polygon`) gerekir.

        Args:
            corridor_polygon: Yol ağının çekileceği alan (WGS84).

        Returns:
            networkx.MultiDiGraph, ya da veri/bağlantı hatası durumunda
            None (bu durumda çağıran taraf düz hatta geri döner).
        """
        try:
            graph = ox.graph_from_polygon(
                corridor_polygon,
                network_type="drive",
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

        try:
            route_node_ids = nx.shortest_path(
                road_graph, orig_node, dest_node, weight="length"
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
