import json
from core.geo_utils import GeoDataFrame, MapVisualizer


if __name__ == '__main__':
    sample_data = {
        'name': ['Pskov', 'Region2'],
        'value': [10, 5]
    }
    with open('src/russia_regions_map/ru.json') as f:
        sample_geometries = json.load(f)

    gdf = GeoDataFrame(sample_data, sample_geometries)
    visualizer = MapVisualizer()
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
