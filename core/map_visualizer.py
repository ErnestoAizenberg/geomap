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
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class MapVisualizer3D:
    def __init__(self, figsize=(16, 10), dpi=100, dark_mode=True):
        self.figsize = figsize
        self.dpi = dpi
        self.dark_mode = dark_mode
        self._setup_colormaps()
    
    def _setup_colormaps(self):
        """Настройка цветовых карт и параметров освещения"""
        if self.dark_mode:
            self.facecolor = 'black'
            self.textcolor = 'white'
            self.cmap = cm.viridis
            self.boundary_color = (0.7, 0.7, 0.7, 0.5)
            self.light_source = LightSource(azdeg=315, altdeg=45)
        else:
            self.facecolor = 'white'
            self.textcolor = 'black'
            self.cmap = cm.plasma
            self.boundary_color = (0.3, 0.3, 0.3, 0.5)
            self.light_source = LightSource(azdeg=225, altdeg=45)
    


    def visualize_3d(self,
                    gdf,
                    value_dict: Dict[str, float],
                    height_dict: Optional[Dict[str, float]] = None,
                    title: str = "",
                    output_path: Optional[str] = None,
                    elev: float = 45,
                    azim: float = -60,
                    z_scale: float = 0.1,
                    edge_alpha: float = 0.3,
                    face_alpha: float = 0.8,
                    scale_height: bool = True):
        """3D визуализация с исправленной обработкой цветов"""
        # Подготовка данных
        gdf['value'] = gdf['name'].map(value_dict).fillna(0)
        height_data = gdf['name'].map(height_dict if height_dict else value_dict).fillna(0)
        
        # Нормализация высот
        if scale_height and height_data.max() > 0:
            heights = (height_data / height_data.max()).values * z_scale
        else:
            heights = height_data.values * z_scale
        
        # Создание фигуры
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi, facecolor=self.facecolor)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor(self.facecolor)
        
        # Настройка вида
        ax.view_init(elev=elev, azim=azim)
        ax.grid(False)
        ax.set_axis_off()
        
        # Нормализация значений
        norm = plt.Normalize(vmin=gdf['value'].min(), vmax=gdf['value'].max())
        
        # Визуализация полигонов
        for idx, geom in enumerate(gdf.geometry):
            if geom.type not in ["Polygon", "MultiPolygon"]:
                continue
                
            height = heights[idx]
            color_val = norm(gdf['value'].iloc[idx])
            
            # Получаем базовый цвет из colormap
            base_color = self.cmap(color_val)
            
            # Преобразуем в массив RGB и применяем освещение
            rgb_array = np.array([base_color[:3]])  # Только RGB компоненты
            shaded_rgb = self.light_source.shade_normals(rgb_array, fraction=0.7)[0]
            
            # Создаем итоговый цвет в правильном формате
            facecolor = [
                float(100),  # R
                float(100),  # G
                float(100),  # B
                face_alpha            # Alpha
            ]
            
            edgecolor = [
                float(self.boundary_color[0]),
                float(self.boundary_color[1]),
                float(self.boundary_color[2]),
                edge_alpha
            ]
            
            # Обработка полигонов
            polygons = geom.polygons if geom.type == "MultiPolygon" else [geom]
            for poly in polygons:
                verts = [np.column_stack([
                    poly.exterior[:,0], 
                    poly.exterior[:,1], 
                    np.full(len(poly.exterior), height)
                ])]
                
                ax.add_collection3d(Poly3DCollection(
                    verts,
                    facecolors=[facecolor],
                    edgecolors=[edgecolor],
                    linewidths=0.5
                ))
                
                # Добавляем боковые грани
                if height > 0:
                    self._add_side_faces(ax, poly.exterior, height, facecolor)
        
        # Настройка осей
        self._set_3d_axes_limits(ax, gdf)
        
        # Цветовая шкала
        mappable = cm.ScalarMappable(norm=norm, cmap=self.cmap)
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, pad=0.01)
        cbar.ax.yaxis.set_tick_params(color=self.textcolor)
        plt.setp(cbar.ax.get_yticklabels(), color=self.textcolor)
        
        if title:
            ax.set_title(title, color=self.textcolor, pad=20)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight', facecolor=self.facecolor, dpi=self.dpi)
        
        plt.close()
    
    def _add_side_faces(self, ax, coords, height, facecolor):
        """Добавление боковых граней с затемненным цветом"""
        side_color = [
            facecolor[0] * 0.8,  # R
            facecolor[1] * 0.8,  # G
            facecolor[2] * 0.8,  # B
            facecolor[3] * 0.7   # Alpha
        ]
        
        for i in range(len(coords)-1):
            x = [coords[i][0], coords[i+1][0], coords[i+1][0], coords[i][0]]
            y = [coords[i][1], coords[i+1][1], coords[i+1][1], coords[i][1]]
            z = [0, 0, height, height]
            
            ax.add_collection3d(Poly3DCollection(
                [list(zip(x, y, z))],
                facecolors=[side_color],
                linewidths=0.2
            ))


    def _set_3d_axes_limits(self, ax, gdf):
        """Устанавливает границы осей"""
        all_coords = []
        for geom in gdf.geometry:
            if geom.type == "Polygon":
                all_coords.append(geom.exterior)
            elif geom.type == "MultiPolygon":
                for poly in geom.polygons:
                    all_coords.append(poly.exterior)
        
        if not all_coords:
            return
        
        all_coords = np.vstack(all_coords)
        x_min, y_min = all_coords.min(axis=0)
        x_max, y_max = all_coords.max(axis=0)
        
        padding = max(x_max - x_min, y_max - y_min) * 0.05
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.set_zlim(0, (y_max - y_min) * 0.5)
        ax.set_box_aspect([1, 1, 0.2])

