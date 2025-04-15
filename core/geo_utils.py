from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

class Geometry(ABC):
    @abstractmethod
    def centroid(self) -> np.ndarray:
        pass
    
    @property
    @abstractmethod
    def type(self) -> str:
        pass
    
    @abstractmethod
    def plot_boundary(self, ax, **kwargs):
        pass
    
    @abstractmethod
    def plot_fill(self, ax, **kwargs):
        pass

class Polygon(Geometry):
    def __init__(self, exterior: np.ndarray, interiors: Optional[List[np.ndarray]] = None):
        self.exterior = np.array(exterior)
        self.interiors = [np.array(i) for i in interiors] if interiors else []
    
    @property
    def type(self) -> str:
        return "Polygon"
    
    def centroid(self) -> np.ndarray:
        x = self.exterior[:, 0]
        y = self.exterior[:, 1]
        signed_area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
        cx = np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / (6 * signed_area)
        cy = np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / (6 * signed_area)
        return np.array([cx, cy])
    
    def plot_boundary(self, ax, **kwargs):
        ax.plot(*self.exterior.T, **kwargs)
        for interior in self.interiors:
            ax.plot(*interior.T, **kwargs)
    
    def plot_fill(self, ax, **kwargs):
        patch = MplPolygon(self.exterior, **kwargs)
        ax.add_patch(patch)
        for interior in self.interiors:
            patch = MplPolygon(interior, **kwargs)
            ax.add_patch(patch)

class GeoDataFrame:
    def __init__(self, data: Dict[str, List], geometry: List[Geometry]):
        self.data = pd.DataFrame(data)
        self.geometry = geometry
    
    def boundary_plot(self, ax, **kwargs):
        """Отрисовка границ всех геометрий"""
        for geom in self.geometry:
            geom.plot_boundary(ax, **kwargs)
    
    def plot_fill(self, ax, **kwargs):
        """Заливка всех геометрий"""
        for geom in self.geometry:
            geom.plot_fill(ax, **kwargs)
    
    def __getitem__(self, key: str):
        return self.data[key]
    
    def __setitem__(self, key: str, value):
        self.data[key] = value
    
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
    
    def get_geometry_by_name(self, name: str) -> Optional[Geometry]:
        """Получение геометрии по имени региона"""
        if name not in self.data['name'].values:
            return None
        idx = self.data[self.data['name'] == name].index[0]
        return self.geometry[idx]

class MapVisualizer:
    def __init__(self, figsize=(32, 20)):
        self.figsize = figsize
    
    def visualize(self,
                 gdf: GeoDataFrame,
                 visual_params: Dict[str, Any],
                 title: str,
                 output_path: Optional[str] = None):
        """
        Визуализация карты
        
        Args:
            gdf: GeoDataFrame с геоданными
            visual_params: {
                'color_map': Dict[str, float],  # Значения для раскраски
                'top_regions': List[str],      # Регионы для верхних меток
                'bottom_regions': List[str],   # Регионы для нижних меток
                '3d': bool,                   # Использовать 3D визуализацию
                'z_values': Dict[str, float]   # Высоты для 3D
            }
            title: Заголовок карты
            output_path: Путь для сохранения
        """
        # Подготовка данных
        gdf['color_value'] = gdf['name'].map(visual_params.get('color_map', {}))
        
        # Создание фигуры
        fig = plt.figure(figsize=self.figsize)
        
        if visual_params.get('3d', False):
            ax = fig.add_subplot(111, projection='3d')
            self._plot_3d(ax, gdf, visual_params)
        else:
            ax = plt.subplot(111)
            self._plot_2d(ax, gdf, visual_params)
        
        # Настройка внешнего вида
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        plt.title(title, fontsize=36, color='white')
        ax.set_axis_off()
        
        # Сохранение или отображение
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight', facecolor='black')
        plt.close()
    
    def _plot_2d(self, ax, gdf, params):
        """2D визуализация"""
        # Отрисовка границ
        gdf.boundary_plot(ax, color='black', linewidth=0.2)
        
        # Отрисовка заливки
        if not gdf['color_value'].empty:
            norm = plt.Normalize(
                gdf['color_value'].min(),
                gdf['color_value'].max()
            )
            cmap = plt.get_cmap('RdYlGn')
            
            for idx, geom in enumerate(gdf.geometry):
                color_val = gdf['color_value'].iloc[idx]
                if pd.isna(color_val):
                    continue
                    
                color = cmap(norm(color_val))
                geom.plot_fill(ax, alpha=0.5, facecolor=color, edgecolor='k')
        
        # Добавление меток
        self._add_labels(ax, gdf, params.get('top_regions', []), 'lightgreen')
        self._add_labels(ax, gdf, params.get('bottom_regions', []), 'red')
    
    def _plot_3d(self, ax, gdf, params):
        """3D визуализация"""
        if 'z_values' not in params:
            raise ValueError("Для 3D визуализации необходимо указать z_values")
            
        gdf['z_value'] = gdf['name'].map(params['z_values'])
        norm = plt.Normalize(gdf['z_value'].min(), gdf['z_value'].max())
        cmap = plt.get_cmap('RdYlGn')
        
        for idx, geom in enumerate(gdf.geometry):
            z_val = gdf['z_value'].iloc[idx]
            if pd.isna(z_val):
                continue
                
            color = cmap(norm(z_val))
            if geom.type == "Polygon":
                x, y = geom.exterior[:, 0], geom.exterior[:, 1]
                ax.plot_trisurf(x, y, [z_val]*len(x), color=color, alpha=0.5)
    
    def _add_labels(self, ax, gdf, regions, color):
        """Добавление меток для указанных регионов"""
        for region in regions:
            geom = gdf.get_geometry_by_name(region)
            if geom is None:
                continue
                
            value = gdf[gdf['name'] == region]['color_value'].iloc[0]
            centroid = geom.centroid()
            ax.text(
                *centroid, f'{value:.2f}',
                fontsize=10, ha='center', color=color
            )

# Пример использования
if __name__ == '__main__':
    # Создаем тестовые данные
    sample_data = {
        'name': ['Region1', 'Region2'],
        'value': [10, 5]
    }
    
    sample_geometries = [
        Polygon(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])),
        Polygon(np.array([[2, 2], [3, 2], [3, 3], [2, 3], [2, 2]]))
    ]
    
    gdf = GeoDataFrame(sample_data, sample_geometries)
    
    # Визуализация
    visualizer = MapVisualizer()
    
    # 2D визуализация
    visualizer.visualize(
        gdf=gdf,
        visual_params={
            'color_map': {'Region1': 10, 'Region2': 5},
            'top_regions': ['Region1'],
            'bottom_regions': ['Region2']
        },
        title='Sample 2D Map',
        output_path='output/sample_2d.png'
    )
    
    # 3D визуализация
    visualizer.visualize(
        gdf=gdf,
        visual_params={
            'color_map': {'Region1': 10, 'Region2': 5},
            '3d': True,
            'z_values': {'Region1': 100, 'Region2': 50},
            'top_regions': ['Region1']
        },
        title='Sample 3D Map',
        output_path='output/sample_3d.png'
    )
