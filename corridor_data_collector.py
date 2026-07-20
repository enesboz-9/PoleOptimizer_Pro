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
    from shapely.geometry import Point, LineString, Polygon
    from shapely.ops import transform as shapely_transform
    import pyproj
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Bu modül osmnx, geopandas, shapely ve pyproj paketlerine ihtiyaç duyar. "
        "Kurulum için: pip install osmnx geopandas shapely pyproj"
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
    """
    node_id: int
    point: GeoPoint
    cumulative_distance_m: float
    is_feasible: Optional[bool] = None
    elevation_m: Optional[float] = None
    blocking_reason: Optional[str] = None


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
        virtual_nodes: A'dan B'ye doğru X metre aralıklarla üretilen
            sanal direk aday düğümleri listesi.
    """
    start: GeoPoint
    end: GeoPoint
    corridor_buffer_m: float
    buildings_gdf: gpd.GeoDataFrame = field(default_factory=gpd.GeoDataFrame)
    obstacles_gdf: gpd.GeoDataFrame = field(default_factory=gpd.GeoDataFrame)
    roads_gdf: gpd.GeoDataFrame = field(default_factory=gpd.GeoDataFrame)
    corridor_polygon: Optional[Polygon] = None
    virtual_nodes: List[VirtualNode] = field(default_factory=list)


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
    ) -> None:
        if corridor_buffer_m <= 0:
            raise ValueError("corridor_buffer_m sıfırdan büyük olmalıdır.")
        if node_spacing_m <= 0:
            raise ValueError("node_spacing_m sıfırdan büyük olmalıdır.")

        self.start = start
        self.end = end
        self.corridor_buffer_m = corridor_buffer_m
        self.node_spacing_m = node_spacing_m

        # A-B hattının orta noktasına göre en uygun UTM projeksiyonunu seç.
        mid_lat = (start.lat + end.lat) / 2
        mid_lon = (start.lon + end.lon) / 2
        self.utm_epsg = _get_utm_epsg(mid_lat, mid_lon)
        self.to_utm, self.to_wgs84 = _build_transformers(self.utm_epsg)

        logger.info(
            "CorridorDataCollector başlatıldı | A=%s B=%s | UTM=%s | "
            "buffer=%.1fm | spacing=%.1fm",
            start.as_tuple(), end.as_tuple(), self.utm_epsg,
            corridor_buffer_m, node_spacing_m,
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

        virtual_nodes = self._generate_virtual_nodes()

        corridor_data = CorridorData(
            start=self.start,
            end=self.end,
            corridor_buffer_m=self.corridor_buffer_m,
            buildings_gdf=buildings_gdf,
            obstacles_gdf=obstacles_gdf,
            roads_gdf=roads_gdf,
            corridor_polygon=corridor_polygon_wgs84,
            virtual_nodes=virtual_nodes,
        )

        logger.info(
            "Veri toplama tamamlandı | %d bina | %d engel poligonu | "
            "%d yol segmenti | %d sanal düğüm",
            len(buildings_gdf), len(obstacles_gdf), len(roads_gdf),
            len(virtual_nodes),
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

    def _generate_virtual_nodes(self) -> List[VirtualNode]:
        """A noktasından B noktasına doğru, `node_spacing_m` aralıklarla
        sanal direk aday düğümleri (virtual nodes) üretir.

        Enterpolasyon, hassasiyet için UTM projeksiyonunda yapılır
        (WGS84 derece biriminde doğrusal enterpolasyon coğrafi olarak
        hatalı sonuç verir), ardından düğümler tekrar WGS84'e çevrilir.

        Not: Bu fonksiyon SADECE A-B doğrusu üzerinde ham (kısıtsız)
        aday noktalar üretir. Gerçek rota optimizasyonu (yamaçtan
        kaçınma, bina etrafından dolanma vb.) sonraki "Optimizasyon
        Motoru" modülünde bu düğümler bir başlangıç arama uzayı
        (search space) olarak kullanılarak yapılır.

        Returns:
            node_id sırasına göre (0 = A, son eleman = B) sıralanmış
            VirtualNode listesi.
        """
        start_xy = np.array(self.to_utm.transform(*self.start.as_xy()))
        end_xy = np.array(self.to_utm.transform(*self.end.as_xy()))

        total_distance_m = float(np.linalg.norm(end_xy - start_xy))

        if total_distance_m == 0:
            raise ValueError("Başlangıç ve bitiş noktaları aynı olamaz.")

        num_segments = max(1, math.ceil(total_distance_m / self.node_spacing_m))
        nodes: List[VirtualNode] = []

        for i in range(num_segments + 1):
            t = i / num_segments  # 0.0 -> A, 1.0 -> B
            interpolated_xy = start_xy + t * (end_xy - start_xy)

            lon, lat = self.to_wgs84.transform(interpolated_xy[0], interpolated_xy[1])
            cumulative_distance = t * total_distance_m

            nodes.append(
                VirtualNode(
                    node_id=i,
                    point=GeoPoint(lat=lat, lon=lon),
                    cumulative_distance_m=round(cumulative_distance, 3),
                )
            )

        logger.info(
            "Toplam hat mesafesi: %.2f m | Üretilen düğüm sayısı: %d "
            "(hedef aralık: %.1fm)",
            total_distance_m, len(nodes), self.node_spacing_m,
        )

        return nodes


# ------------------------------------------------------------------ #
# Örnek Kullanım (Bağımsız çalıştırma testi için)
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    # Örnek: Çorum bölgesinde iki nokta arasında test amaçlı bir koridor.
    collector = CorridorDataCollector(
        start=GeoPoint(lat=40.5506, lon=34.9556),
        end=GeoPoint(lat=40.5650, lon=34.9700),
        corridor_buffer_m=150.0,   # Koridor genişliği: A-B hattının her iki yanına 150m
        node_spacing_m=40.0,       # Her 40 metrede bir aday direk düğümü
    )

    data = collector.run()

    print(f"\nÜretilen sanal düğüm sayısı: {len(data.virtual_nodes)}")
    for node in data.virtual_nodes[:5]:
        print(
            f"  Node #{node.node_id}: "
            f"lat={node.point.lat:.6f}, lon={node.point.lon:.6f}, "
            f"mesafe={node.cumulative_distance_m:.1f}m"
        )
