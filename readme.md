# GeoMap - Geometric Data Visualization Project

![Map example](https://github.com/ErnestoAizenberg/geomap/blob/main/output%2russia_regions_predict.jpg)

## Description

GeoMap is project for working with geometric objects and visualizing them on 2D and 3D maps. It implements an abstract `Geometry` class along with concrete classes like `Polygon` that allow creating geometric shapes, calculating their centroids, and visualizing them using `matplotlib`.

## Features

- Abstract `Geometry` base class with common geometric operations
- `Polygon` class implementation with centroid calculation
- GeoDataFrame for storing geometric data and attributes
- MapVisualizer for 2D/3D visualization
- Support for custom color mapping and region highlighting

## Installation

Install required dependencies using either method:

### Via requirements.txt
```bash
pip install -r requirements.txt
```

### Manual installation
```bash
pip install numpy>=1.21.0 pandas>=1.3.0 matplotlib>=3.4.0
```

## Usage

### Basic Example

```python
import json
from core.geo_utils import GeoDataFrame, MapVisualizer

if __name__ == '__main__':
    sample_data = {
        'name': ['Pskov', 'Region2'],
        'value': [10, 5]
    }

    # Load geometric data from JSON file
    with open('src/russia_regions_map/ru.json') as f:
        sample_geometries = json.load(f)

    # Create GeoDataFrame and visualizer
    gdf = GeoDataFrame(sample_data, sample_geometries)
    visualizer = MapVisualizer()

    # Visualize 2D map
    visualizer.visualize(
        gdf=gdf,
        visual_params={
            'color_map': {'Pskov': 10, 'Region2': 5},
            'top_regions': ['Pskov'],
            'bottom_regions': ['Region2']
        },
        title='Sample 2D Map',
        output_path='output/sample_2d.png'
    )
```

## API Reference

### `Geometry` (Abstract Class)
- `centroid()`: Returns the centroid of the geometry
- `type`: Returns geometry type
- `plot_boundary(ax, **kwargs)`: Plots geometry boundary
- `plot_fill(ax, **kwargs)`: Fills geometry

### `Polygon` (Inherits from `Geometry`)
- `__init__(exterior, interiors)`: Initializes polygon with exterior and interior boundaries
- `centroid()`: Computes polygon centroid
- `plot_boundary(ax, **kwargs)`: Plots polygon boundary
- `plot_fill(ax, **kwargs)`: Fills polygon

### `GeoDataFrame`
- `boundary_plot(ax, **kwargs)`: Plots all geometry boundaries
- `plot_fill(ax, **kwargs)`: Fills all geometries
- `get_geometry_by_name(name)`: Gets geometry by region name

### `MapVisualizer`
- `visualize(gdf, visual_params, title, output_path)`: Visualizes map with geometric data

## Examples

See the `examples/` directory for:
- Basic 2D visualization
- Custom color mapping
- Region highlighting

## Contributing

Contributions are welcome! Please open an issue or pull request for any bugs or feature requests.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).