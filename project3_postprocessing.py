#Member: Chao-Huai Tsao, Kai-Wen Yang

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits import mplot3d
from scipy.linalg import norm
from tqdm import tqdm
#from elastica.rod import cosserat_rod
#from elastica._linalg import _batch_matvec
def envelope(arg_pos):
    """
    Given points, computes the arc length and envelope of the curve
    """

    # Computes the direction in which the rod points
    # in our cases it should be the z-axis
    rod_direction = arg_pos[:, -1] - arg_pos[:, 0]
    rod_direction /= norm(rod_direction, ord=2, axis=0)

    # Compute local tangent directions
    tangent_s = np.diff(arg_pos, n=1, axis=-1)  # x_(i+1)-x(i)
    length_s = norm(tangent_s, ord=2, axis=0)
    tangent_s /= length_s

    # Dot product with direction is cos_phi, see RSOS
    cos_phi_s = np.einsum("ij,i->j", tangent_s, rod_direction)

    # Compute phi-max now
    phi = np.arccos(cos_phi_s)
    cos_phi_max = np.cos(np.max(phi))

    # Return envelope and arclength
    envelope = (cos_phi_s - cos_phi_max) / (1.0 - cos_phi_max)
    # -0.5 * length accounts for the element/node business
    arclength = np.cumsum(length_s) - 0.5 * length_s[0]

    return arclength, envelope
    
def plot_axialcompression(rod, SAVE_FIGURE):
    plt.figure()
    plt.axes(projection="3d")
    plt.plot(
        rod.position_collection[0, ...],
        rod.position_collection[1, ...],
        rod.position_collection[2, ...],
    )
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    if SAVE_FIGURE:
        plt.savefig("axialcompression_3d" + str(rod.n_elems) + ".png")
    plt.show()

    base_length = np.sum(rod.rest_lengths)
    #phi_analytical_envelope = analytical_solution(base_length)
    phi_computed_envelope = envelope(rod.position_collection)

    plt.figure()
    #plt.plot(phi_analytical_envelope[0], phi_analytical_envelope[1], label="analytical")
    plt.plot(
        phi_computed_envelope[0],
        phi_computed_envelope[1],
        label="n=" + str(rod.n_elems),
    )
    plt.legend()
    if SAVE_FIGURE:
        plt.savefig("HelicalBuckling_Envelope" + str(rod.n_elems) + ".png")
    plt.show()


#code for plotting video but fail
def plot_video(
    plot_params:dict,
    video_name = "landing_gear",
    fps=15,
    xlim=(0,4),
    ylim=(-4,4)):

    import matplotlib.animation as manimation
    time =plot_params["time"]
    #positions_over_time = np.array([plot_params["position"]])
    positions_over_time = plot_params["position"]
    print("plot video....")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps,metadata=metadata)
    fig = plt.figure(figsize=(10,8), frameon=True, dpi=150)
    with writer.saving(fig, video_name, 100):
        for time in range(len(time),1):
            fig.clf()
            ax = plt.axes(projection="3d") #fig.add_subplot(111)
            ax.plot(positions_over_time)
            ax.set_xlim(0,4)
            ax.set_ylim(0,4)
            ax.set_zlim(0,4)
            ax.set_xlabel("x[m]")
            ax.set_ylabel("y[m]")
            ax.set_zlabel("z[m]")   

            writer.grab_frame()



