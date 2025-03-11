#!/usr/bin/env python3
import sys
import os
import numpy as np
import pyvista as pv
import argparse

def visualize_vtk(filename, scalar_field=None, screenshot=None):
    """
    Visualize a VTK file with PyVista
    
    Parameters:
    -----------
    filename : str
        Path to the VTK file
    scalar_field : str, optional
        Name of the scalar field to visualize. If None, the first available scalar field is used.
    screenshot : str, optional
        Path to save a screenshot of the visualization
    """
    print(f"Reading VTK file: {filename}")
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File {filename} does not exist")
        return
    
    # Read the VTK file
    try:
        grid = pv.read(filename)
    except Exception as e:
        print(f"Error reading VTK file: {e}")
        return
    
    print(f"VTK file loaded successfully")
    print(f"Number of points: {grid.n_points}")
    print(f"Number of cells: {grid.n_cells}")
    
    # Get available scalar fields
    point_data_keys = list(grid.point_data.keys())
    if not point_data_keys:
        print("No scalar fields found in the VTK file")
        return
    
    print(f"Available scalar fields: {', '.join(point_data_keys)}")
    
    # Select scalar field
    if scalar_field is None:
        scalar_field = point_data_keys[0]
        print(f"No scalar field specified, using: {scalar_field}")
    elif scalar_field not in point_data_keys:
        print(f"Scalar field '{scalar_field}' not found. Available fields: {', '.join(point_data_keys)}")
        print(f"Using {point_data_keys[0]} instead")
        scalar_field = point_data_keys[0]
    
    # Get scalar range
    scalar_range = grid.get_data_range(scalar_field)
    print(f"Scalar range for {scalar_field}: [{scalar_range[0]}, {scalar_range[1]}]")
    
    # Create plotter with off_screen option if screenshot is requested
    off_screen = screenshot is not None
    plotter = pv.Plotter(off_screen=off_screen)
    
    # Add data with a colormap
    plotter.add_mesh(grid, scalars=scalar_field, cmap="viridis", 
                    show_edges=True, edge_color='black',
                    scalar_bar_args={'title': scalar_field})
    
    # Add axes
    plotter.add_axes()
    plotter.add_bounding_box()
    
    # Add title
    plotter.add_title(f"VTK Visualization - {os.path.basename(filename)}")
    
    # Show the visualization and save screenshot if requested
    if screenshot:
        print(f"Saving screenshot to {screenshot}")
        plotter.show(auto_close=False)  # Show but don't close
        plotter.screenshot(screenshot)
        plotter.close()
        print(f"Screenshot saved to {screenshot}")
    else:
        # Just show the visualization
        print("Displaying visualization. Close the window to exit.")
        plotter.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize VTK files')
    parser.add_argument('filename', help='Path to the VTK file')
    parser.add_argument('-f', '--field', help='Scalar field to visualize')
    parser.add_argument('-s', '--screenshot', help='Save screenshot to file')
    args = parser.parse_args()
    
    # Visualize the VTK file
    visualize_vtk(args.filename, args.field, args.screenshot)

if __name__ == '__main__':
    main()