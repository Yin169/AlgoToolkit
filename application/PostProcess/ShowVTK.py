#!/usr/bin/env python3
import sys
import os
import numpy as np
import pyvista as pv
import argparse

def visualize_vtk(filename, scalar_field=None, vector_field=None, screenshot=None, 
                  display_mode='surface', colormap='viridis', clip_plane=None,
                  scale_factor=1.0, slice_plane=None, slice_normal=None, slice_origin=None):
    """
    Visualize a VTK file with PyVista
    
    Parameters:
    -----------
    filename : str
        Path to the VTK file
    scalar_field : str, optional
        Name of the scalar field to visualize. If None, the first available scalar field is used.
    vector_field : str, optional
        Name of the vector field to visualize (e.g., velocity)
    screenshot : str, optional
        Path to save a screenshot of the visualization
    display_mode : str, optional
        Display mode for the mesh: 'surface', 'wireframe', 'points', or 'surface_with_edges'
    colormap : str, optional
        Colormap to use for scalar data visualization
    clip_plane : tuple, optional
        (origin, normal) for a clipping plane to see inside the volume
    scale_factor : float, optional
        Scale factor for vector field visualization
    slice_plane : str, optional
        Plane to slice the data: 'x', 'y', 'z', or None
    slice_normal : tuple, optional
        Normal vector for custom slice plane (x, y, z)
    slice_origin : tuple, optional
        Origin point for custom slice plane (x, y, z)
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
    
    # Get available data fields
    point_data_keys = list(grid.point_data.keys())
    cell_data_keys = list(grid.cell_data.keys())
    
    print("Available point data fields:", ", ".join(point_data_keys) if point_data_keys else "None")
    print("Available cell data fields:", ", ".join(cell_data_keys) if cell_data_keys else "None")
    
    # Create plotter with off_screen option if screenshot is requested
    off_screen = screenshot is not None
    plotter = pv.Plotter(off_screen=off_screen)
    
    # Set up display mode
    show_edges = False
    style = 'surface'
    
    if display_mode == 'wireframe':
        style = 'wireframe'
    elif display_mode == 'points':
        style = 'points'
    elif display_mode == 'surface_with_edges':
        style = 'surface'
        show_edges = True
    
    # Apply clipping plane if specified
    if clip_plane:
        origin, normal = clip_plane
        grid = grid.clip(normal, origin)
    
    # Handle scalar field visualization
    if scalar_field is None and point_data_keys:
        scalar_field = point_data_keys[0]
        print(f"No scalar field specified, using: {scalar_field}")
    elif scalar_field and scalar_field not in point_data_keys and scalar_field not in cell_data_keys:
        available_fields = point_data_keys + cell_data_keys
        print(f"Scalar field '{scalar_field}' not found. Available fields: {', '.join(available_fields)}")
        if point_data_keys:
            print(f"Using {point_data_keys[0]} instead")
            scalar_field = point_data_keys[0]
        else:
            scalar_field = None
    
    # Create a slice if specified
    if slice_plane or (slice_normal and slice_origin):
        # Create a copy of the original grid for slicing
        slice_grid = grid.copy()
        
        if slice_plane:
            # Get the center of the dataset for default slice origin
            center = slice_grid.center
            
            # Create a slice along the specified axis
            if slice_plane.lower() == 'x':
                slice_normal = (1, 0, 0)
                slice_origin = (center[0], 0, 0)
                print(f"Creating slice at x={center[0]}")
            elif slice_plane.lower() == 'y':
                slice_normal = (0, 1, 0)
                slice_origin = (0, center[1], 0)
                print(f"Creating slice at y={center[1]}")
            elif slice_plane.lower() == 'z':
                slice_normal = (0, 0, 1)
                slice_origin = (0, 0, center[2])
                print(f"Creating slice at z={center[2]}")
            else:
                print(f"Invalid slice plane: {slice_plane}. Using default visualization.")
                slice_grid = None
        else:
            # Use the custom slice normal and origin
            print(f"Creating custom slice with normal={slice_normal}, origin={slice_origin}")
        
        if slice_grid is not None:
            # Create the slice
            slice_result = slice_grid.slice(normal=slice_normal, origin=slice_origin)
            
            # Add the slice to the visualization
            if scalar_field:
                plotter.add_mesh(slice_result, scalars=scalar_field, cmap=colormap,
                                style=style, show_edges=show_edges, edge_color='black',
                                scalar_bar_args={'title': scalar_field})
            else:
                plotter.add_mesh(slice_result, style=style, show_edges=show_edges, edge_color='black')
            
            # Also add the outline of the original dataset
            plotter.add_mesh(grid.outline(), color='black', line_width=1.0)
    else:
        # Regular visualization without slicing
        if scalar_field:
            # Get scalar range
            scalar_range = grid.get_data_range(scalar_field)
            print(f"Scalar range for {scalar_field}: [{scalar_range[0]}, {scalar_range[1]}]")
            
            # Add data with a colormap
            plotter.add_mesh(grid, scalars=scalar_field, cmap=colormap, 
                            style=style, show_edges=show_edges, edge_color='black',
                            scalar_bar_args={'title': scalar_field})
        else:
            # Just show the mesh without scalar data
            plotter.add_mesh(grid, style=style, show_edges=show_edges, edge_color='black')
    
    # Handle vector field visualization if specified
    if vector_field and vector_field in point_data_keys:
        vectors = grid.point_data[vector_field]
        if vectors.shape[1] == 3:  # Make sure it's a 3D vector field
            # Add arrows to represent the vector field
            plotter.add_arrows(grid.points, vectors, mag=scale_factor, name='vectors')
            print(f"Added vector field visualization for: {vector_field}")
        else:
            print(f"Vector field {vector_field} is not a 3D vector field")
    
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
    parser.add_argument('-s', '--scalar', dest='scalar_field', help='Scalar field to visualize')
    parser.add_argument('-v', '--vector', dest='vector_field', help='Vector field to visualize')
    parser.add_argument('-o', '--output', dest='screenshot', help='Save screenshot to file')
    parser.add_argument('-d', '--display', dest='display_mode', 
                        choices=['surface', 'wireframe', 'points', 'surface_with_edges'],
                        default='surface', help='Display mode for the mesh')
    parser.add_argument('-c', '--colormap', dest='colormap', default='viridis',
                        help='Colormap for scalar data (e.g., viridis, jet, rainbow)')
    parser.add_argument('-x', '--clip-x', dest='clip_x', type=float,
                        help='X coordinate for clipping plane origin')
    parser.add_argument('-y', '--clip-y', dest='clip_y', type=float,
                        help='Y coordinate for clipping plane origin')
    parser.add_argument('-z', '--clip-z', dest='clip_z', type=float,
                        help='Z coordinate for clipping plane origin')
    parser.add_argument('-n', '--clip-normal', dest='clip_normal', type=str,
                        help='Normal vector for clipping plane (format: x,y,z)')
    parser.add_argument('-f', '--scale-factor', dest='scale_factor', type=float, default=1.0,
                        help='Scale factor for vector field visualization')
    parser.add_argument('--slice', dest='slice_plane', choices=['x', 'y', 'z'],
                        help='Create a slice along the specified axis')
    parser.add_argument('--slice-normal', dest='slice_normal', type=str,
                        help='Normal vector for custom slice plane (format: x,y,z)')
    parser.add_argument('--slice-origin', dest='slice_origin', type=str,
                        help='Origin point for custom slice plane (format: x,y,z)')
    
    args = parser.parse_args()
    
    # Process clipping plane arguments
    clip_plane = None
    if all(v is not None for v in [args.clip_x, args.clip_y, args.clip_z, args.clip_normal]):
        origin = (args.clip_x, args.clip_y, args.clip_z)
        normal = tuple(map(float, args.clip_normal.split(',')))
        if len(normal) != 3:
            print("Error: Clip normal must be in format x,y,z")
            return
        clip_plane = (origin, normal)
    
    # Process custom slice arguments
    slice_normal = None
    slice_origin = None
    if args.slice_normal and args.slice_origin:
        try:
            slice_normal = tuple(map(float, args.slice_normal.split(',')))
            slice_origin = tuple(map(float, args.slice_origin.split(',')))
            if len(slice_normal) != 3 or len(slice_origin) != 3:
                print("Error: Slice normal and origin must be in format x,y,z")
                return
        except ValueError:
            print("Error: Invalid format for slice normal or origin")
            return
    
    # Visualize the VTK file
    visualize_vtk(args.filename, args.scalar_field, args.vector_field, args.screenshot,
                 args.display_mode, args.colormap, clip_plane, args.scale_factor,
                 args.slice_plane, slice_normal, slice_origin)
if __name__ == '__main__':
    main()