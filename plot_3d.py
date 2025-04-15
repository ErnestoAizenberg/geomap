import json
from core.map_visualizer import MapVisualizer3D
from core.gu import GeoProcessor

def main():
    # Initialize the 3D map visualizer with dark mode and specific figure size.
    visualizer_3d = MapVisualizer3D(dark_mode=True, figsize=(14, 10))

    # Data for visualization: region names with corresponding values and heights.
    values = {
        "Pskow": 25, 
        "Moskva": 18, 
        "Chuvash": 100, 
        "Bashkortostan": 78, 
        "Udmurt": 26
    }
    heights = {
        "Pskow": 30, 
        "Moskva": 22, 
        "Chuvash": 100, 
        "Bashkortostan": 78, 
        "Udmurt": 26
    }

    # Load GeoJSON data for the regions of Russia from a file.
    with open('src/russia_regions_map/ru.json') as f:
        sample_geojson = json.load(f)

    # Convert the GeoJSON data into a GeoDataFrame.
    regions_gdf = GeoProcessor.from_geojson(sample_geojson)

    # Create a 3D map using the visualizer with specified titles, paths, and view parameters.
    visualizer_3d.visualize_3d(
        gdf=regions_gdf,
        value_dict=values,
        height_dict=heights,
        title="3D map of Russian Regions",
        output_path="russia_3d_map.png",
        elev=40,
        azim=-70,
        z_scale=0.2,
        face_alpha=0.9
    )

if __name__ == '__main__':
    main()