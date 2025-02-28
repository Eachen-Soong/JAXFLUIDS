import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np

from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids_postprocess import load_data, create_2D_animation, create_2D_figure

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


# SETUP SIMULATION
input_manager = InputManager("NACA.json", "numerical_setup.json")
initialization_manager = InitializationManager(input_manager)
sim_manager = SimulationManager(input_manager)

# RUN SIMULATION
simulation_buffers, time_control_variables,\
forcing_parameters = initialization_manager.initialization()
sim_manager.simulate(simulation_buffers, time_control_variables)

# LOAD DATA
path = sim_manager.output_writer.save_path_domain
quantities = [
    "density", "schlieren", "mach_number", 
    "levelset", "volume_fraction", "pressure"
]
cell_centers, cell_sizes, times, data_dict = load_data(path, quantities)

mask_fluid = data_dict["volume_fraction"] > 0.0
mask_solid = 1.0 - mask_fluid

# PLOT
nrows_ncols = (2,2)
plot_dict = {
    "density"       : np.ma.masked_where(mask_solid, data_dict["density"]),
    "pressure"      : np.ma.masked_where(mask_solid, data_dict["pressure"]),
    "mach_number"   : np.clip(np.ma.masked_where(mask_solid, data_dict["mach_number"]), 0.0, 3.0),
    "schlieren"     : np.clip(np.ma.masked_where(mask_solid, data_dict["schlieren"]), 1e0, 5e2)
}

# CREATE ANIMATION
create_2D_animation(
    plot_dict, 
    cell_centers, 
    times, 
    nrows_ncols=nrows_ncols, 
    plane="xy", plane_value=0.0,
    interval=100)

# CREATE FIGURE
create_2D_figure(
    plot_dict,
    nrows_ncols=nrows_ncols,
    cell_centers=cell_centers, 
    plane="xy", plane_value=0.0,
    dpi=400, save_fig="NACA.png")
