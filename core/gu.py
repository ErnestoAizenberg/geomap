import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Geometry(ABC):
    @abstractmethod
    def centroid(self) -> np.ndarray:
        pass
    
    @property
    @abstractmethod
    def type(self) -> str:
        pass
    
    @abstractmethod
    def plot(self, ax, **kwargs):
        pass
    
    @abstractmethod
    def plot_3d(self, ax, height: float = 0, **kwargs):
        pass
    
    @property
    def area(self) -> float:
        """Вычисляет площадь геометрии (0 для точек и линий)"""
        return 0.0

class Point(Geometry):
    def __init__(self, coords: np.ndarray):
        self.coords = np.array(coords, dtype=np.float64)
    
    @property
    def type(self) -> str:
        return "Point"
    
    def centroid(self) -> np.ndarray:
        return self.coords
    
    def plot(self, ax, **kwargs):
        ax.scatter(*self.coords, **kwargs)
    
    def plot_3d(self, ax, height: float = 0, **kwargs):
        ax.scatter(*self.coords, height, **kwargs)

class Polygon(Geometry):
    def __init__(self, exterior: np.ndarray, interiors: Optional[List[np.ndarray]] = None):
        self.exterior = np.array(exterior, dtype=np.float64)
        self.interiors = [np.array(i, dtype=np.float64) for i in interiors] if interiors else []
    
    @property
    def type(self) -> str:
        return "Polygon"
    
    @property
    def area(self) -> float:
        """Вычисляет площадь полигона с учетом отверстий"""
        main_area = 0.5 * np.abs(np.sum(
            self.exterior[:-1, 0] * self.exterior[1:, 1] - 
            self.exterior[1:, 0] * self.exterior[:-1, 1]
        ))
        hole_area = sum(
            0.5 * np.abs(np.sum(
                hole[:-1, 0] * hole[1:, 1] - 
                hole[1:, 0] * hole[:-1, 1]
            )) for hole in self.interiors
        )
        return main_area - hole_area
    
    def centroid(self) -> np.ndarray:
        x = self.exterior[:, 0]
        y = self.exterior[:, 1]
        signed_area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
        if signed_area == 0:
            return np.mean(self.exterior, axis=0)
        cx = np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / (6 * signed_area)
        cy = np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / (6 * signed_area)
        return np.array([cx, cy])
    
    def plot(self, ax, **kwargs):
        patch = plt.Polygon(self.exterior, **kwargs)
        ax.add_patch(patch)
        for interior in self.interiors:
            patch = plt.Polygon(interior, **kwargs)
            ax.add_patch(patch)
    
    def plot_3d(self, ax, height: float = 0, **kwargs):
        verts = [np.column_stack([self.exterior[:, 0], self.exterior[:, 1], np.full(len(self.exterior), height)])]
        poly = Poly3DCollection(verts, **kwargs)
        ax.add_collection3d(poly)
        
        for interior in self.interiors:
            verts = [np.column_stack([interior[:, 0], interior[:, 1], np.full(len(interior), height)])]
            poly = Poly3DCollection(verts, **kwargs)
            ax.add_collection3d(poly)

class MultiPolygon(Geometry):
    def __init__(self, polygons: List[Polygon]):
        self.polygons = polygons
    
    @property
    def type(self) -> str:
        return "MultiPolygon"
    
    @property
    def area(self) -> float:
        return sum(poly.area for poly in self.polygons)
    
    def centroid(self) -> np.ndarray:
        centroids = []
        areas = []
        for poly in self.polygons:
            area = poly.area
            centroids.append(poly.centroid())
            areas.append(area)
        
        total_area = np.sum(areas)
        if total_area == 0:
            return np.mean(centroids, axis=0)
        return np.average(centroids, weights=areas, axis=0)
    
    def plot(self, ax, **kwargs):
        for poly in self.polygons:
            poly.plot(ax, **kwargs)
    
    def plot_3d(self, ax, height: float = 0, **kwargs):
        for poly in self.polygons:
            poly.plot_3d(ax, height, **kwargs)

class LineString(Geometry):
    def __init__(self, coords: np.ndarray):
        self.coords = np.array(coords, dtype=np.float64)
    
    @property
    def type(self) -> str:
        return "LineString"
    
    def centroid(self) -> np.ndarray:
        return np.mean(self.coords, axis=0)
    
    def plot(self, ax, **kwargs):
        ax.plot(*self.coords.T, **kwargs)
    
    def plot_3d(self, ax, height: float = 0, **kwargs):
        ax.plot(*self.coords.T, height, **kwargs)

class GeoDataFrame:
    def __init__(self, data: Dict[str, List], geometry: List[Geometry]):
        self.data = pd.DataFrame(data)
        self.geometry = geometry
        self.crs = None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, key: str):
        return self.data[key]
    
    def __setitem__(self, key: str, value):
        self.data[key] = value



    def copy(self) -> 'GeoDataFrame':
        """Создает глубокую копию GeoDataFrame"""
        return GeoDataFrame( 
           self.data.copy(deep=True).to_dict('list'),
           [geom for geom in self.geometry]
        )
    
    def plot(self, ax, **kwargs):
        for geom in self.geometry:
            geom.plot(ax, **kwargs)
    
    def plot_3d(self, ax, height_column: Optional[str] = None, **kwargs):
        for idx, geom in enumerate(self.geometry):
            height = self.data[height_column].iloc[idx] if height_column else 0
            geom.plot_3d(ax, height, **kwargs)
    
    def boundary_plot(self, ax, **kwargs):
        for geom in self.geometry:
            if geom.type in ["Polygon", "MultiPolygon"]:
                if geom.type == "Polygon":
                    ax.plot(*geom.exterior.T, **kwargs)
                    for interior in geom.interiors:
                        ax.plot(*interior.T, **kwargs)
                else:
                    for poly in geom.polygons:
                        ax.plot(*poly.exterior.T, **kwargs)
                        for interior in poly.interiors:
                            ax.plot(*interior.T, **kwargs)
            elif geom.type == "LineString":
                geom.plot(ax, **kwargs)
    
    def nlargest(self, n: int, column: str) -> 'GeoDataFrame':
        idx = self.data[column].nlargest(n).index
        return self._subset_by_index(idx)
    
    def nsmallest(self, n: int, column: str) -> 'GeoDataFrame':
        idx = self.data[column].nsmallest(n).index
        return self._subset_by_index(idx)
    
    def _subset_by_index(self, idx) -> 'GeoDataFrame':
        return GeoDataFrame(
            self.data.loc[idx].to_dict('list'),
            [self.geometry[i] for i in idx]
        )

class GeoProcessor:
    @staticmethod
    def from_geojson(geojson: Dict[str, Any]) -> GeoDataFrame:
        features = geojson['features']
        properties = []
        geometries = []
        
        for feature in features:
            props = feature['properties']
            geom = GeoProcessor._parse_geometry(feature['geometry'])
            properties.append(props)
            geometries.append(geom)
        
        all_keys = set().union(*(p.keys() for p in properties))
        data = {k: [p.get(k) for p in properties] for k in all_keys}
        return GeoDataFrame(data, geometries)
    
    @staticmethod
    def _parse_geometry(geom_dict: Dict[str, Any]) -> Geometry:
        geom_type = geom_dict['type']
        coordinates = geom_dict['coordinates']
        
        if geom_type == "Point":
            return Point(coordinates)
        elif geom_type == "LineString":
            return LineString(coordinates)
        elif geom_type == "Polygon":
            return Polygon(coordinates[0], coordinates[1:])
        elif geom_type == "MultiPolygon":
            polygons = [Polygon(poly[0], poly[1:]) for poly in coordinates]
            return MultiPolygon(polygons)
        else:
            raise ValueError(f"Unsupported geometry type: {geom_type}")

class MapVisualizer:
    def __init__(self, figsize=(16, 10), dpi=100, dark_mode=True):
        self.figsize = figsize
        self.dpi = dpi
        self.dark_mode = dark_mode
        self._setup_colormaps()
    
    def _setup_colormaps(self):
        if self.dark_mode:
            self.facecolor = 'black'
            self.textcolor = 'white'
            self.cmap = LinearSegmentedColormap.from_list(
                'dark_rdylgn', ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
            )
            self.boundary_color = 'gray'
        else:
            self.facecolor = 'white'
            self.textcolor = 'black'
            self.cmap = 'RdYlGn'
            self.boundary_color = 'black'
    
    def visualize(self, 
                 gdf: GeoDataFrame, 
                 color_map: Dict[str, float],
                 title: str,
                 output_path: Optional[str] = None,
                 show_values: bool = True,
                 top_n: int = 10,
                 bottom_n: int = 10,
                 show_legend: bool = True,
                 show_colorbar: bool = True,
                 boundary_width: float = 0.2,
                 **plot_kwargs):
        """
        Визуализация с внешним словарем значений для регионов
        
        Параметры:
        ----------
        gdf : GeoDataFrame
            Геодатафрейм для визуализации
        color_map : Dict[str, float]
            Словарь с значениями для регионов (ключ - имя региона)
        title : str
            Заголовок карты
        output_path : str, optional
            Путь для сохранения изображения
        show_values : bool, default True
            Показывать ли значения на карте
        top_n : int, default 10
            Количество топовых регионов для подписей
        bottom_n : int, default 10
            Количество аутсайдеров для подписей
        show_legend : bool, default True
            Показывать ли легенду с топами и аутсайдерами
        show_colorbar : bool, default True
            Показывать ли шкалу цветов
        boundary_width : float, default 0.2
            Толщина границ регионов
        **plot_kwargs
            Дополнительные параметры для matplotlib
        """
        start_time = time.time()
        
        # Добавляем значения из color_map в GeoDataFrame
        gdf['value'] = gdf['name'].map(color_map).fillna(0)
        values = gdf['value']
        
        # Создаем фигуру
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        fig.patch.set_facecolor(self.facecolor)
        ax.set_facecolor(self.facecolor)
        
        # Нормализация значений
        norm = Normalize(vmin=values.min(), vmax=values.max())
        sm = ScalarMappable(norm=norm, cmap=self.cmap)
        
        # Сначала рисуем границы
        gdf.boundary_plot(ax, color=self.boundary_color, linewidth=boundary_width)
        
        # Рисуем полигоны с цветами
        for idx, geom in enumerate(gdf.geometry):
            color_val = gdf['value'].iloc[idx]
            color = sm.to_rgba(color_val)
            
            if geom.type in ["Polygon", "MultiPolygon"]:
                geom.plot(ax, alpha=0.7, facecolor=color, edgecolor=self.boundary_color, **plot_kwargs)
        
        # Добавляем подписи
        if show_values:
            self._add_labels(gdf, ax, 'value', top_n, bottom_n)
        
        # Добавляем легенду
        if show_legend:
            self._add_annotations(gdf, ax, 'value', top_n, bottom_n)
        
        # Добавляем шкалу цветов
        if show_colorbar:
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.01)
            cbar.ax.yaxis.set_tick_params(color=self.textcolor)
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=self.textcolor)
            cbar.set_label('Value', color=self.textcolor)
        
        # Настройки заголовка
        plt.title(title, fontsize=24, color=self.textcolor, pad=20)
        ax.set_axis_off()
        
        # Сохраняем или показываем
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight', facecolor=self.facecolor, dpi=self.dpi)
        
        plt.close()
        print(f"Визуализация завершена за {time.time() - start_time:.2f} секунд")
    
    def visualize_3d(self,
                    gdf: GeoDataFrame,
                    color_map: Dict[str, float],
                    height_map: Optional[Dict[str, float]] = None,
                    title: str = "",
                    output_path: Optional[str] = None,
                    elev: float = 30,
                    azim: float = -60,
                    scale_height: float = 1.0,
                    **plot_kwargs):
        """
        3D визуализация с внешними словарями значений и высот
        
        Параметры:
        ----------
        gdf : GeoDataFrame
            Геодатафрейм для визуализации
        color_map : Dict[str, float]
            Словарь с значениями для цвета регионов
        height_map : Dict[str, float], optional
            Словарь с высотами регионов (если None, используется color_map)
        title : str
            Заголовок карты
        output_path : str, optional
            Путь для сохранения изображения
        elev : float, default 30
            Угол возвышения камеры
        azim : float, default -60
            Азимутальный угол камеры
        scale_height : float, default 1.0
            Масштабный коэффициент для высот
        **plot_kwargs
            Дополнительные параметры для matplotlib
        """
        start_time = time.time()
        
        # Добавляем значения из словарей в GeoDataFrame
        gdf['color_value'] = gdf['name'].map(color_map).fillna(0)
        if height_map is not None:
            gdf['height_value'] = gdf['name'].map(height_map).fillna(0)
        else:
            gdf['height_value'] = gdf['color_value']
        
        # Преобразуем высоты в numpy array для надежности
        height_data = gdf['height_value'].values if hasattr(gdf['height_value'], 'values') else np.array(gdf['height_value'])
        max_height = height_data.max()
        if max_height > 0:
            heights = height_data / max_height * scale_height
        else:
            heights = np.zeros(len(gdf))
        
        # Создаем 3D фигуру
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        fig.patch.set_facecolor(self.facecolor)
        ax.set_facecolor(self.facecolor)
        
        # Настройка вида
        ax.view_init(elev=elev, azim=azim)
        ax.grid(False)
        ax.set_axis_off()
        
        # Нормализация значений для цвета
        values = gdf['color_value']
        norm = Normalize(vmin=values.min(), vmax=values.max())
        sm = ScalarMappable(norm=norm, cmap=self.cmap)
        
        # Рисуем 3D полигоны
        for idx, geom in enumerate(gdf.geometry):
            color_val = gdf['color_value'].iloc[idx] if hasattr(gdf['color_value'], 'iloc') else gdf['color_value'][idx]
            color = sm.to_rgba(color_val)
            height = heights[idx]  # Теперь heights точно numpy array
            
            if geom.type in ["Polygon", "MultiPolygon"]:
                geom.plot_3d(ax, height=height, 
                           color=color, edgecolor=self.boundary_color, 
                           alpha=0.7, **plot_kwargs)
        
        # Добавляем шкалу цветов
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.01)
        cbar.ax.yaxis.set_tick_params(color=self.textcolor)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=self.textcolor)
        cbar.set_label('Value', color=self.textcolor)
        
        # Настройки заголовка
        plt.title(title, fontsize=24, color=self.textcolor, pad=20)
        
        # Сохраняем или показываем
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight', facecolor=self.facecolor, dpi=self.dpi)
        
        plt.close()
        print(f"3D визуализация завершена за {time.time() - start_time:.2f} секунд")




   
    def _add_labels(self, gdf, ax, value_column: str, top_n: int, bottom_n: int):
        top = gdf.nlargest(top_n, value_column)
        bottom = gdf.nsmallest(bottom_n, value_column)
        
        for df, color in [(top, 'lightgreen'), (bottom, 'red')]:
            for idx in range(len(df)):
                centroid = df.geometry[idx].centroid()
                label = df[value_column].iloc[idx]
                ax.text(centroid[0], centroid[1], f'{label:.2f}', 
                       fontsize=8, ha='center', color=color,
                       bbox=dict(facecolor=self.facecolor, alpha=0.7, edgecolor='none'))
    
    def _add_annotations(self, gdf, ax, value_column: str, top_n: int, bottom_n: int):
        top = gdf.nlargest(top_n, value_column)
        bottom = gdf.nsmallest(bottom_n, value_column)
        
        for i, (name, val) in enumerate(zip(top['name'], top[value_column])):
            ax.text(1.05, 0.95 - i * 0.05, f'■ {name}: {val:.2f}',
                   fontsize=10, ha='left', color='lightgreen', transform=ax.transAxes,
                   bbox=dict(facecolor=self.facecolor, alpha=0.7, edgecolor='none'))
        
        for i, (name, val) in enumerate(zip(bottom['name'], bottom[value_column])):
            ax.text(1.05, 0.45 - i * 0.05, f'■ {name}: {val:.2f}',
                   fontsize=10, ha='left', color='red', transform=ax.transAxes,
                   bbox=dict(facecolor=self.facecolor, alpha=0.7, edgecolor='none'))

if __name__ == '__main__':
    # Пример данных
    sample_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Region1"},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [
                        [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                        [[[2, 2], [3, 2], [3, 3], [2, 3], [2, 2]]]
                    ]
                }
            },
            {
                "type": "Feature",
                "properties": {"name": "Region2"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[4, 4], [5, 4], [5, 5], [4, 5], [4, 4]]
                    ]
                }
            }
        ]
    }
    
    with open('src/russia_regions_map/ru.json') as f:
        sample_geojson = json.load(f)


    # Создаем GeoDataFrame
    gdf = GeoProcessor.from_geojson(sample_geojson)
    
    # Внешние данные для визуализации
    color_values = {"Pskov": 100, "Moscow": 50}
    height_values = {"Pskov": 80, "Moskow": 30}
    
    # Тестируем 2D визуализацию
    visualizer = MapVisualizer(dark_mode=True)
    visualizer.visualize(
        gdf=gdf,
        color_map=color_values,
        title='Пример 2D карты с внешними значениями',
        output_path='output/test_map_2d.png',
        top_n=2,
        bottom_n=1
    )
    
    # Тестируем 3D визуализацию
    visualizer.visualize_3d(
        gdf=gdf,
        color_map=color_values,
        height_map=height_values,
        title='Пример 3D карты с внешними значениями и высотами',
        output_path='output/test_map_3d.png',
        scale_height=2.0
    )