import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids_postprocess import load_data, create_1D_animation, create_1D_figure

# SETUP SIMULATION
input_manager = InputManager("twophasesod.json", "numerical_setup.json")
initialization_manager = InitializationManager(input_manager)
sim_manager = SimulationManager(input_manager)

# RUN SIMULATION
simulation_buffers, time_control_variables, \
forcing_parameters = initialization_manager.initialization()
sim_manager.simulate(simulation_buffers, time_control_variables)

# LOAD DATA
path = sim_manager.output_writer.save_path_domain
quantities = [
    "real_density", "real_pressure", 
    "real_velocity", "levelset"]
cell_centers, cell_sizes, times, data_dict = load_data(path, quantities)

# PLOT
plot_dict = {
    "real_density": data_dict["real_density"], 
    "real_velocityX": data_dict["real_velocity"][:,0],
    "real_pressure": data_dict["real_pressure"],
}
nrows_ncols = (1,3)

# CREATE ANIMATION
create_1D_animation(
    plot_dict,
    cell_centers,
    times,
    nrows_ncols=nrows_ncols,
    interval=100,
    fig_args={"figsize": (15,5)})

# CREATE FIGURE
create_1D_figure(
    plot_dict,
    cell_centers=cell_centers,
    nrows_ncols=nrows_ncols,
    axis="x", axis_values=(0,0),
    save_fig="twophasesod.png")