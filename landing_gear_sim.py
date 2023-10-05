#Member: Chao-Huai Tsao, Kai-Wen Yang

import numpy as np
from matplotlib import pyplot as plt
from project3_postprocessing import(plot_axialcompression,plot_video)
from elastica import *



class AxialCompressionSimulator(
    BaseSystemCollection, Constraints, Forcing, CallBacks, Damping):
    pass

axialcompression_sim =AxialCompressionSimulator()
final_time = 100

#Options
PLOT_FIGURE = True
SAVE_VIDEO = True
SAVE_FIGURE = True
SAVE_RESULTS = False


n_element = 20
start = np.zeros((3,))
direction = np.array([0.0,0.0,1.0])
normal = np.array([0.0, 1.0, 0.0])
base_length = 3.0
base_radius = 0.1
base_area = np.pi * base_radius ** 2
density = 5000

E = 4.418e6
# For shear modulus of 1e4, nu is 99!
poisson_ratio = 0.5
shear_modulus = E / (poisson_ratio + 1.0) 

shearable_rod = CosseratRod.straight_rod(
    n_element,
    start,
    direction, 
    normal,
    base_length,
    base_radius,
    density,
    0.0, #internal damping constant
    E,
    shear_modulus=shear_modulus,
)

axialcompression_sim.append(shearable_rod)

#add damping
dl = base_length/n_element
dt = 1e-3
axialcompression_sim.dampen(shearable_rod).using(
    AnalyticalLinearDamper,
    damping_constant =0.13595, #Refernence [1]
    time_step = dt,
)

#Define the boundary condition: fix two ends of the rod
axialcompression_sim.constrain(shearable_rod).using(
    OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
    #FixedConstraint,
    #constrained_position_idx = (0,-1),
    #constrained_director_idx = (0,-1)
)

end_force_axial = -5000
end_force = np.array([0.0,0.0,end_force_axial])
#end_force = np.array([end_force_axial,0.0,0.0])
axialcompression_sim.add_forcing_to(shearable_rod).using(
    EndpointForces, 0.0 * end_force, end_force, ramp_up_time=1e-2)

#end_force_axial = -5000
#force_direction = np.array([0.0,0.0,1.0])
#beambulking_sim.add_forcing_to(shearable_rod).using(
    #UniformForces,force,force_direction
#)





#Add call backs
class AxialCompressionCallBack(CallBackBaseClass):
    """
    Tracks the velocity norms of the rod
    """

    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):

        if current_step % self.every == 0:

            self.callback_params["time"].append(time)
            """
            # Collect x
            self.callback_params["position"].append(system.position_collection[0,-1].copy())
            self.callback_params["velocity_norms"].append( np.linalg.norm(system.velocity_collection.copy())

            # Collect y
            self.callback_params["position"].append(system.position_collection[1,-1].copy())
            self.callback_params["velocity_norms"].append( np.linalg.norm(system.velocity_collection.copy())            
            """
            # Collect z
            self.callback_params["position"].append(
                system.position_collection[2,-1].copy()
            )
            self.callback_params["velocity_norms"].append(
                np.linalg.norm(system.velocity_collection.copy())
            )
            return

recorded_history = defaultdict(list)
axialcompression_sim.collect_diagnostics(shearable_rod).using(
    AxialCompressionCallBack, step_skip=500, callback_params=recorded_history
)

axialcompression_sim.finalize()
timestepper = PositionVerlet()

total_steps = int(final_time/dt)
print("Total steps", total_steps)
integrate(timestepper, axialcompression_sim, final_time, total_steps)
np_position =  np.array(recorded_history["position"]) #change data type from str to float
print(np_position)
print(recorded_history["time"])
print(np.shape(recorded_history))
new_position = (np_position-base_length)/base_length #strain
#print(shearable_rod.position_collection)
rendering_fps = 60
if PLOT_FIGURE:
    delta_L = end_force*base_length / base_area/E
    #modified length
    delta_L_improved = (
        end_force*base_length/(base_area*E-end_force)
    )

    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
    ax = fig.add_subplot(111)
    ax.plot(recorded_history["time"], recorded_history["position"], lw=2.0) #z
    plt.xlabel('time (s)')
    plt.ylabel('position (m)')
    ax.hlines(base_length + delta_L, 0.0 , final_time, "k","dashdot",lw=1.0)
    if SAVE_FIGURE:
        fig.savefig("axial_compression.pdf")
    plt.show()
    plt.plot(recorded_history["time"],new_position) #strain: deltaz/z
    plt.xlabel('time (s)')
    plt.ylabel('Axial strain')
    plot_axialcompression(shearable_rod, SAVE_FIGURE)


    #Generating video, but fail.
    if SAVE_VIDEO:
        filename_video = "Landing_gear.mp4"
        plot_video(
            recorded_history,
            video_name = filename_video,
            fps=rendering_fps,
            xlim = (0,4),
            ylim=(0,4),
        )
    

  
 

if SAVE_RESULTS:
    import pickle

    filename = "axial_compression_data.dat"
    file = open(filename, "wb")
    pickle.dump(shearable_rod, file)
    file.close()
    tv = (
        np.asarray(recorded_history["time"]),
        np.asarray(recorded_history["velocity_norms"]),
    )

    def as_time_series(v):
        return v.T

    np.savetxt(
        "velocity_norms.csv",
        as_time_series(np.stack(tv)),
        delimiter=",",
    )
    


