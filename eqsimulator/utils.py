import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.animation as animation
from matplotlib.colors import Normalize, SymLogNorm, LogNorm
from matplotlib.cm import ScalarMappable
import matplotlib.colorbar as clb
from matplotlib.collections import PatchCollection, LineCollection
from scipy.interpolate import interp1d

year = 3600.0 * 24.0 * 365.0
# --------------------------------------------------------


def interpolate_shear_stress_history(fault, time):
    interpolators = []
    for i in range(fault.n_patches):
        interpolators.append(
            interp1d(
                fault.fault_patches[i].time, fault.fault_patches[i].shear_stress_history
            )
        )
    shear_stress = np.zeros((fault.n_patches, time.size), dtype=np.float64)
    for i in range(fault.n_patches):
        shear_stress[i, :] = interpolators[i](time)
    return shear_stress


def interpolate_normal_stress_history(fault, time):
    interpolators = []
    for i in range(fault.n_patches):
        interpolators.append(
            interp1d(
                fault.fault_patches[i].time,
                fault.fault_patches[i].normal_stress_history,
            )
        )
    normal_stress = np.zeros((fault.n_patches, time.size), dtype=np.float64)
    for i in range(fault.n_patches):
        normal_stress[i, :] = interpolators[i](time)
    return normal_stress


def interpolate_slip_history(fault, time):
    interpolators = []
    for i in range(fault.n_patches):
        interpolators.append(
            interp1d(
                fault.fault_patches[i].time,
                fault.fault_patches[i].coseismic_slip_history,
            )
        )
    slip = np.zeros((fault.n_patches, time.size), dtype=np.float64)
    for i in range(fault.n_patches):
        slip[i, :] = interpolators[i](time)
    return slip


def interpolate_state_history(fault, time):
    states = -1 * np.ones((fault.n_patches, time.size), dtype=np.int32)
    for i in range(fault.n_patches):
        for n in range(1, fault.fault_patches[i].state_history.size):
            mask = (states[i, :] == -1) & (
                time < fault.fault_patches[i].transition_times_history[n]
            )
            states[i, mask] = fault.fault_patches[i].state_history[n - 1]
        mask = (states[i, :] == -1) & (
            time > fault.fault_patches[i].transition_times_history[-1]
        )
        states[i, mask] = fault.fault_patches[i].state_history[-1]
    return states


def nucleation_length(fault):
    Lc = np.zeros(fault.n_patches, dtype=np.float64)
    for n, fp in enumerate(fault.fault_patches):
        eta = fp.k * fp.L / fp.G
        Lc[n] = (eta * fp.G * fp.Dc) / ((fp.b - fp.a) * fp.normal_stress_history[0])
    return Lc


# =====================================================================================================================
#                                   CATALOG BUILDING FUNCTIONS


def get_event_history(fault):
    import pandas as pd

    list_events = []
    for n in range(fault.n_patches):
        for k in range(fault.fault_patches[n].coseismic_displacements.size):
            list_events.append(
                [
                    fault.fault_patches[n].event_timings[k],
                    fault.fault_patches[n].fault_patch_id,
                    fault.fault_patches[n].x,
                    fault.fault_patches[n].y,
                    fault.fault_patches[n].z,
                    fault.fault_patches[n].coseismic_displacements[k],
                    fault.fault_patches[n].stress_drop_history[k],
                ]
            )
    list_events = sorted(list_events)  # order the list in chronological order
    catalog = np.float64(list_events)  # timings and locations = earthquake catalog
    catalog = catalog[catalog[:, -1] > 0.0, :]
    catalog = pd.DataFrame(
        data=np.asarray(catalog),
        columns=["event_time", "patch_id", "x", "y", "z", "slip", "stress_drop"],
    )
    return catalog


def local_catalog(catalog, fault, patch_index, R=500.0):
    patches = []
    for n in range(fault.n_patches):
        if (
            np.linalg.norm(fault.coords[patch_index, :] - fault.coords[n, :]) < R
            and fault.line_indexes[n] == fault.line_indexes[patch_index]
        ):
            patches.append(n)
    patches = np.float64(patches)
    local_cat = []
    for i in range(len(catalog)):
        if catalog[i][1] in patches:
            local_cat.append(catalog[i])
    local_cat = np.float64(local_cat)
    return local_cat


def define_events(catalog, time_threshold, distance_threshold):
    events = []
    visited_events = set()
    for i in catalog.index:
        if i in visited_events:
            continue
        # print('{:d}/{:d}'.format(i+1, catalog.shape[0]))
        dT = np.abs(catalog["event_time"] - catalog.loc[i, "event_time"]).values
        dX = np.abs(catalog["x"] - catalog.loc[i, "x"]).values
        visited_events.add(i)
        candidates = set(
            catalog.index[
                np.where((dT < time_threshold) & (dX < distance_threshold))[0]
            ]
        ).difference(visited_events)
        group = [i]
        if len(candidates) == 0:
            events.append(catalog.loc[group])
            continue
        for can in candidates:
            D = np.linalg.norm(
                catalog.loc[can, ["x", "y", "z"]].values
                - catalog.loc[i, ["x", "y", "z"]].values
            )
            if D < distance_threshold:
                group.append(can)
        new_events = True
        n0 = 0
        n1 = 0
        while new_events:
            new_events = False
            n1 = len(group)
            for event_idx in group[n0:]:
                # print('----> {:d}'.format(event_idx))
                dT = np.abs(
                    catalog["event_time"] - catalog.loc[event_idx, "event_time"]
                ).values
                dX = np.abs(catalog["x"] - catalog.loc[event_idx, "x"]).values
                visited_events.add(event_idx)
                candidates = set(
                    catalog.index[
                        np.where((dT < time_threshold) & (dX < distance_threshold))[0]
                    ]
                ).difference(visited_events)
                for can in candidates:
                    if (can in visited_events) or (can == event_idx) or (can in group):
                        continue
                    D = np.linalg.norm(
                        catalog.loc[can, ["x", "y", "z"]].values
                        - catalog.loc[event_idx, ["x", "y", "z"]].values
                    )
                    if D < distance_threshold:
                        group.append(can)
                        new_events = True
            n0 = int(n1)
        events.append(catalog.loc[np.unique(group)])
    print("{:d} events ({:d} single events)".format(len(events), len(catalog)))
    return events


# ChatGPT suggestion


def calculate_distance(point1, point2):
    """
    Calculate the Euclidean distance between two 3D points.

    Parameters:
    -----------
    point1 : array-like
        Coordinates of the first point (x, y, z).
    point2 : array-like
        Coordinates of the second point (x, y, z).

    Returns:
    --------
    float
        Euclidean distance between the two points.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))


def find_event_neighbors(catalog, event_index, time_threshold, distance_threshold):
    """
    Find neighboring events for a given event based on time and distance thresholds.

    Parameters:
    -----------
    catalog : pandas.DataFrame
        DataFrame containing event data with columns 'event_time', 'x', 'y', 'z'.
    event_index : int
        Index of the event for which neighbors are to be found.
    time_threshold : float
        Time threshold for considering events as neighbors (in seconds).
    distance_threshold : float
        Distance threshold for considering events as neighbors.

    Returns:
    --------
    set
        Set of indices of neighboring events.
    """
    event = catalog.loc[event_index]
    dT = np.abs(catalog["event_time"] - event["event_time"])
    dX = np.linalg.norm(catalog[["x", "y", "z"]] - event[["x", "y", "z"]], axis=1)
    neighbor_indices = set(
        catalog.index[(dT < time_threshold) & (dX < distance_threshold)]
    )
    neighbor_indices.discard(event_index)  # Remove the event itself
    return neighbor_indices


# def define_events(catalog, time_threshold, distance_threshold):
#    """
#    Cluster events in a catalog based on time and distance thresholds.
#
#    Parameters:
#    -----------
#    catalog : pandas.DataFrame
#        DataFrame containing event data with columns 'event_time', 'x', 'y', 'z'.
#    time_threshold : float
#        Time threshold for considering events as part of the same cluster (in seconds).
#    distance_threshold : float
#        Distance threshold for considering events as part of the same cluster.
#
#    Returns:
#    --------
#    list
#        List of DataFrames, where each DataFrame represents a cluster of events.
#    """
#    if catalog.empty:
#        return []
#
#    events = []
#    visited_events = set()
#
#    for event_index in catalog.index:
#        if event_index in visited_events:
#            continue
#
#        cluster = [event_index]
#        neighbors_to_check = [event_index]
#
#        while neighbors_to_check:
#            current_event_index = neighbors_to_check.pop()
#
#            neighbors = find_event_neighbors(
#                catalog, current_event_index, time_threshold, distance_threshold
#            )
#            neighbors.difference_update(visited_events, cluster)
#
#            cluster.extend(neighbors)
#            neighbors_to_check.extend(neighbors)
#
#            visited_events.update(neighbors)
#
#        events.append(catalog.loc[cluster])
#
#    print(f"{len(events)} events ({len(catalog)} single events)")
#    return events


def build_catalog_from_metadata(events, metadata):
    seismic_moment = np.zeros(len(events), dtype=np.float64)
    average_slip = np.zeros(len(events), dtype=np.float32)
    rupture_area = np.zeros(len(events), dtype=np.float32)
    event_time = np.zeros(len(events), dtype=np.float64)
    for i in range(len(seismic_moment)):
        event_time[i] = events[i]["event_time"].min()
        for j in events[i].index:
            seismic_moment[i] += (
                metadata["SHEAR_MODULUS_PA"]
                * metadata["STRIKE_WIDTH"]
                * metadata["DIP_WIDTH"]
                * events[i].loc[j, "slip"]
            )
            rupture_area[i] += metadata["STRIKE_WIDTH"] * metadata["DIP_WIDTH"]
            average_slip[i] += (
                metadata["STRIKE_WIDTH"]
                * metadata["DIP_WIDTH"]
                * events[i].loc[j, "slip"]
            )
        average_slip[i] /= rupture_area[i]
    catalog = pd.DataFrame(
        {
            "event_time": event_time,
            "average_slip": average_slip,
            "rupture_area": rupture_area,
            "seismic_moment": seismic_moment,
            "moment_magnitude": moment_magnitude(seismic_moment),
        }
    )
    return catalog


def build_catalog_from_fault_instance(events, fault, moment_method="slip"):
    assert moment_method in [
        "slip",
        "stress_drop",
    ], "moment_method should be 'slip' or 'stress_drop'"
    seismic_moment = np.zeros(len(events), dtype=np.float64)
    average_slip = np.zeros(len(events), dtype=np.float32)
    average_stress_drop = np.zeros(len(events), dtype=np.float32)
    rupture_area = np.zeros(len(events), dtype=np.float32)
    event_time = np.zeros(len(events), dtype=np.float64)
    for i in range(len(seismic_moment)):
        event_time[i] = events[i]["event_time"].min()
        for j in events[i].index:
            fp = fault.fault_patches[int(events[i].loc[j, "patch_id"])]
            rupture_area[i] += fp.L * fp.W
            average_slip[i] += fp.L * fp.W * events[i].loc[j, "slip"]
            average_stress_drop[i] += fp.L * fp.W * events[i].loc[j, "stress_drop"]
            if moment_method == "slip":
                seismic_moment[i] += fp.G * fp.L * fp.W * events[i].loc[j, "slip"]
            elif moment_method == "stress_drop":
                seismic_moment[i] += (
                    np.pi / 2.0 * fp.L * fp.W**2 * events[i].loc[j, "stress_drop"]
                )
        average_slip[i] /= rupture_area[i]
        average_stress_drop[i] /= rupture_area[i]
    catalog = pd.DataFrame(
        {
            "event_time": event_time,
            "average_slip": average_slip,
            "average_stress_drop": average_stress_drop,
            "rupture_area": rupture_area,
            "seismic_moment": seismic_moment,
            "moment_magnitude": moment_magnitude(seismic_moment),
        }
    )
    return catalog


def moment_magnitude(m0):
    return 2.0 / 3.0 * (np.log10(m0) - 9.1)


def LSQ(X, Y, W=None):
    """
    LSQ(X, Y, W=None) \n
    (Weighted) least square regression, with analytical solution.
    """
    if W is None:
        W = np.ones(X.size)
    W_sum = W.sum()
    x_mean = np.sum(W * X) / W_sum
    y_mean = np.sum(W * Y) / W_sum
    x_var = np.sum(W * np.power(X - x_mean, 2))
    xy_cov = np.sum(W * (X - x_mean) * (Y - y_mean))
    best_slope = xy_cov / x_var
    best_intercept = y_mean - best_slope * x_mean
    return best_intercept, best_slope


# =====================================================================================================================
#                                      PLOT FUNCTIONS
def tectonic_stressing(fault):
    AXES, LIST_PATCHES, INDEXES_PER_LINE = make_axes_collections_3D(
        fault, figname="Ki_tau"
    )
    cm = plt.get_cmap("coolwarm")
    ktau = np.zeros(fault.n_patches, dtype=np.float64)
    for n in range(fault.n_patches):
        ktau[n] = fault.fault_patches[n].ktau_tectonic / 1.0e6
    cNorm = LogNorm(vmin=ktau.min(), vmax=ktau.max())
    scalarMap = ScalarMappable(norm=cNorm, cmap=cm)
    AXES[0].add_collection(
        LineCollection(
            LIST_PATCHES[0][::-1], cmap=cm, norm=cNorm, lw=2, array=ktau[::-1]
        )
    )
    for i in range(1, len(AXES)):
        edgecolors = ["none" for j in range(INDEXES_PER_LINE[i - 1].size)]
        AXES[i].add_collection(
            PatchCollection(
                LIST_PATCHES[i],
                cmap=cm,
                norm=cNorm,
                array=ktau[INDEXES_PER_LINE[i - 1]],
                edgecolors=edgecolors,
            )
        )
    fig = plt.gcf()
    ax_cbar = fig.add_axes([0.92, 0.10, 0.01, 0.80])
    clb.ColorbarBase(
        ax_cbar,
        cmap=cm,
        norm=cNorm,
        label=r"Tectonic stiffness $K_{i}^{\mathrm{T},\tau}$ (MPa/m)",
    )
    plt.subplots_adjust(right=0.90)
    plt.show()


def tectonic_stressing_strike_slip(fault):
    AXES, LIST_PATCHES, INDEXES_PER_LINE = make_axes_collections_3D(
        fault, figname="Ki_tau"
    )
    plt.close()
    fig = plt.figure(
        "tectonic_loading_{:d}_{:.0f}m_{:.0f}m_patches".format(
            fault.n_patches, fault.fault_patches[0].L, fault.fault_patches[0].W
        )
    )
    ax = plt.subplot(2, 1, 1)
    cm = plt.get_cmap("coolwarm")
    # ------------------------------------------------------------------
    ktau = np.zeros(fault.n_patches, dtype=np.float64)
    for n in range(fault.n_patches):
        ktau[n] = fault.fault_patches[n].ktau_tectonic / 1.0e6
    # cNorm = LogNorm(vmin=ktau.min(), vmax=ktau.max())
    cNorm = Normalize(vmin=ktau.min(), vmax=ktau.max())
    scalarMap = ScalarMappable(norm=cNorm, cmap=cm)
    ax.add_collection(PatchCollection(LIST_PATCHES[1], cmap=cm, norm=cNorm, array=ktau))
    ax.set_xlim(AXES[1].get_xlim())
    ax.set_ylim(AXES[1].get_ylim())
    plt.xlabel("Along strike distance (m)")
    plt.ylabel("Depth (m)")
    ax_cbar = fig.add_axes([0.92, 0.55, 0.01, 0.40])
    clb.ColorbarBase(
        ax_cbar,
        cmap=cm,
        norm=cNorm,
        label=r"Tectonic stiffness $K_{i}^{\mathrm{T},\tau}$ (MPa/m)",
    )
    # -------------------------------------------------------------------
    ax = plt.subplot(2, 1, 2)
    stressing_rates = np.zeros(fault.n_patches, dtype=np.float64)
    for n in range(fault.n_patches):
        stressing_rates[n] = (
            ktau[n] * fault.fault_patches[0].tectonic_slip_speed * 3600.0 * 24.0 * 365.0
        )  # MPa/yr
    # cNorm = LogNorm(vmin=stressing_rates.min(), vmax=stressing_rates.max())
    cNorm = Normalize(vmin=stressing_rates.min(), vmax=stressing_rates.max())
    scalarMap = ScalarMappable(norm=cNorm, cmap=cm)
    ax.add_collection(
        PatchCollection(LIST_PATCHES[1], cmap=cm, norm=cNorm, array=stressing_rates)
    )
    ax.set_xlim(AXES[1].get_xlim())
    ax.set_ylim(AXES[1].get_ylim())
    plt.xlabel("Along strike distance (m)")
    plt.ylabel("Depth (m)")
    ax_cbar = fig.add_axes([0.92, 0.05, 0.01, 0.40])
    clb.ColorbarBase(
        ax_cbar,
        cmap=cm,
        norm=cNorm,
        label=r"Tectonic Loading $K_{i}^{\mathrm{T},\tau} \times V_{i}^{\mathrm{tectonic}}$ (MPa/yr)",
    )
    plt.subplots_adjust(right=0.90)
    plt.show()


def frequency_magnitude_hist_only(
    catalog, fault, log=True, Mw_cutoff=5.0, method="LSQ"
):
    M, average_slip = moment_catalog(catalog, fault)
    n_patches = np.int32([len(catalog[i]) for i in range(len(catalog))])
    fig = plt.figure("frequency_magnitude_hist_only", figsize=(18, 15))
    plt.subplot2grid((2, 2), (0, 0))
    plt.title(r"Frequency-Magnitude distribution")
    Mw = moment_magnitude(M)
    Mw_0 = np.percentile(Mw, Mw_cutoff)
    mask = Mw > Mw_0
    n, bins, patches = plt.hist(Mw[mask])
    N = np.cumsum(n[::-1])[::-1]
    # --------------------------
    if method == "MLE":
        b = np.log10(np.exp(1.0)) / (np.mean(Mw[mask]) - Mw_0)
    elif method == "LSQ":
        intercept, b = LSQ(bins[:-1], np.log10(N))
        b *= -1.0
    plt.text(0.6, 0.85, "b-value = {:.1f}".format(b), transform=plt.gca().transAxes)
    plt.xlabel("Moment Magnitude $M_{w}$")
    plt.ylabel("Number of Events")
    if log:
        plt.semilogy()
    plt.show()


def frequency_magnitude(catalog, fault, log=False, Mw_cutoff=5.0):
    M, average_slip = moment_catalog(catalog, fault)
    n_patches = np.int32([len(catalog[i]) for i in range(len(catalog))])
    fig = plt.figure("frequency_magnitude", figsize=(18, 15))
    plt.subplot2grid((2, 2), (0, 0))
    plt.title(r"Frequency-Magnitude distribution")
    Mw = moment_magnitude(M)
    Mw_0 = np.percentile(Mw, Mw_cutoff)
    mask = Mw > Mw_0
    b_MLE = np.log10(np.exp(1.0)) / (np.mean(Mw[mask]) - Mw_0)
    n, bins, patches = plt.hist(Mw[mask])
    N = np.cumsum(n[::-1])[::-1]
    intercept, slope = LSQ(bins[:-1], np.log10(N))
    plt.text(
        0.6,
        0.85,
        "b-value LSQ = {:.1f} \nb-value MLE = {:.1f}".format(abs(slope), b_MLE),
        transform=plt.gca().transAxes,
    )
    plt.xlabel("Moment Magnitude $M_{w}$")
    plt.ylabel("Number of Events")
    if log:
        plt.semilogy()
    plt.subplot2grid((2, 2), (0, 1))
    plt.title(r"Number of patches per event")
    plt.hist(n_patches)
    plt.ylabel("Number of Events")
    plt.xlabel("Number of Patches")
    if log:
        plt.semilogy()
    plt.subplot2grid((2, 2), (1, 0), colspan=2)
    plt.plot(bins[:-1], np.log10(N), marker="o", ls="")
    plt.plot(
        bins[:-1],
        intercept + slope * bins[:-1],
        color="C3",
        label="b-value LSQ = {:.2f}".format(-1.0 * slope),
    )
    plt.plot(
        bins[:-1],
        np.log10(N).mean() - (bins[:-1] - bins[:-1].mean()) * b_MLE,
        color="C4",
        label="b-value MLE = {:.2f}".format(b_MLE),
    )
    plt.legend(loc="best", fancybox=True)
    plt.xlabel(r"Moment Magnitude $M_{w}$")
    plt.ylabel(r"Log[$N > M_{w}$]")
    plt.show()


def scaling_M0_slip(catalog, fault):
    M, average_slip = moment_catalog(catalog, fault)
    n_patches = np.int32([len(catalog[i]) for i in range(len(catalog))])
    plt.figure("scaling_M0")
    plt.subplot(2, 1, 1)
    # plt.plot(average_slip, M, marker='o', ls='')
    plt.scatter(average_slip, M, s=50, rasterized=True)
    D_s = np.linspace(average_slip.min(), average_slip.max(), 100)
    M0_log = np.log10(M.min()) + 3.0 * (np.log10(D_s) - np.log10(D_s.min()))
    M0 = np.power(10.0 * np.ones(M0_log.size), M0_log)
    plt.plot(D_s, M0, ls="--", color="k", label=r"$M_{0} \propto D^{3}$")
    # M0s = np.linspace(M.min(), M.max(), 100)
    # D_log = np.log10(average_slip.min()) + 1./3. * (np.log10(M0s) - np.log10(M0s.min()))
    # D     = np.power(10.*np.ones(D_log.size, dtype=np.float64), D_log)
    # plt.plot(M0s, D, ls='--', color='k')
    plt.loglog()
    plt.grid()
    plt.ylabel(r"Seismic Moment $M_{0}$ (N.m)")
    plt.xlabel("Average Coseismic Slip (m)")
    plt.legend(loc="upper left", fancybox=True)
    plt.subplot(2, 1, 2)
    A = n_patches * fault.fault_patches[0].L * fault.fault_patches[0].W
    A_s = np.linspace(A.min(), A.max(), 100)
    M0_log = np.log10(M.min()) + 3.0 / 2.0 * (np.log10(A_s) - np.log10(A_s.min()))
    M0 = np.power(10.0 * np.ones(M0_log.size), M0_log)
    # plt.plot(A, M, marker='o', ls='')
    plt.scatter(A, M, s=50, rasterized=True)
    plt.plot(A_s, M0, ls="--", color="k", label=r"$M_{0} \propto A^{3/2}$")
    plt.loglog()
    plt.grid()
    plt.xlabel(r"Fault Surface (m$^{2}$)")
    plt.ylabel(r"Seismic Moment $M_{0}$ (N.m)")
    plt.legend(loc="upper left", fancybox=True)
    plt.show()


def interevent_times(catalog, fault, RT_cutoff=0.001, show=True):
    # non declustered catalog
    # RT = catalog[1:,0] - catalog[:-1,0]
    OT = np.float64([catalog[i][0][0] for i in range(len(catalog))])
    RT = OT[1:] - OT[:-1]
    plt.figure("interevent_times_distribution", figsize=(10, 8))
    plt.hist(np.log10(RT[RT > RT_cutoff]), bins=100)
    plt.xlabel("Interevent Time (log[s])")
    plt.ylabel("Number of occurrences")
    if show:
        plt.show()


def plot_Kij_shear(patch_index, fault):
    cm = plt.get_cmap("coolwarm")
    Kij_tau = fault.Kij_shear[:, patch_index] / 1.0e6
    mask = np.ones(Kij_tau.size, dtype=np.bool)
    mask[patch_index] = False
    boundary = max(np.abs(Kij_tau[mask].min()), np.abs(Kij_tau.max()))
    boundary_linear = min(np.abs(Kij_tau[mask].min()), np.abs(Kij_tau.max()))
    cNorm = SymLogNorm(
        vmin=-boundary,
        vmax=+boundary,
        linthresh=boundary_linear / 10000.0,
        linscale=1.0,
    )
    scalarMap = ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure("Kij_shear_patch_{:d}".format(patch_index))
    for n in range(fault.n_patches):
        Lx = fault.fault_patches[n].L * np.sin(
            fault.fault_patches[n].strike_angle * np.pi / 180.0
        )
        Ly = fault.fault_patches[n].L * np.cos(
            fault.fault_patches[n].strike_angle * np.pi / 180.0
        )
        X = [fault.fault_patches[n].x - Lx / 2.0, fault.fault_patches[n].x + Lx / 2.0]
        Y = [fault.fault_patches[n].y - Ly / 2.0, fault.fault_patches[n].y + Ly / 2.0]
        plt.plot(X, Y, color=scalarMap.to_rgba(Kij_tau[n]), lw=2)
        # plt.text(np.mean(X), np.mean(Y), str(n))
    plt.plot(
        fault.fault_patches[patch_index].x,
        fault.fault_patches[patch_index].y,
        marker="s",
        markersize=10,
        color="k",
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.ylim(fault.coords[:, 1].min() - 100.0, fault.coords[:, 1].max() + 100.0)
    plt.gca().set_aspect("equal", "box")
    ax_cbar = fig.add_axes([0.91, 0.10, 0.01, 0.80])
    clb.ColorbarBase(
        ax_cbar,
        cmap=cm,
        norm=cNorm,
        label=r"Shear Stress Dislocation Array $K_{ij}^{\tau}$ (MPa/m)",
    )
    plt.subplots_adjust(right=0.90)
    plt.show()


def plot_Kij_normal(patch_index, fault):
    cm = plt.get_cmap("coolwarm")
    Kij_sigma = fault.Kij_normal[patch_index, :] / 1.0e6
    mask = np.ones(Kij_sigma.size, dtype=np.bool)
    mask[patch_index] = False
    boundary = max(np.abs(Kij_sigma[mask].min()), np.abs(Kij_sigma.max()))
    boundary_linear = min(np.abs(Kij_sigma[mask].min()), np.abs(Kij_sigma.max()))
    cNorm = SymLogNorm(
        vmin=-boundary,
        vmax=+boundary,
        linthresh=boundary_linear / 10000.0,
        linscale=1.0,
    )
    scalarMap = ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure("Kij_normal_patch_{:d}".format(patch_index))
    for n in range(fault.n_patches):
        Lx = fault.fault_patches[n].L * np.sin(
            fault.fault_patches[n].strike_angle * np.pi / 180.0
        )
        Ly = fault.fault_patches[n].L * np.cos(
            fault.fault_patches[n].strike_angle * np.pi / 180.0
        )
        X = [fault.coords[n, 0] - Lx / 2.0, fault.coords[n, 0] + Lx / 2.0]
        Y = [fault.coords[n, 1] - Ly / 2.0, fault.coords[n, 1] + Ly / 2.0]
        plt.plot(X, Y, color=scalarMap.to_rgba(Kij_sigma[n]), lw=2)
        # plt.text(np.mean(X), np.mean(Y), str(n))
    plt.plot(
        fault.coords[patch_index, 0],
        fault.coords[patch_index, 1],
        marker="s",
        markersize=10,
        color="k",
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.ylim(fault.coords[:, 1].min() - 100.0, fault.coords[:, 1].max() + 100.0)
    plt.gca().set_aspect("equal", "box")
    ax_cbar = fig.add_axes([0.91, 0.10, 0.01, 0.80])
    clb.ColorbarBase(
        ax_cbar,
        cmap=cm,
        norm=cNorm,
        label=r"Normal Stress Dislocation Array $K_{ij}^{\sigma}$ (MPa/m)",
    )
    plt.subplots_adjust(right=0.90)
    plt.show()


def make_animation_stress_slip(shear_stress, normal_stress, slip, time, fault):
    fig = plt.figure("animate_stress_slip", figsize=(18, 11))

    ax1 = fig.add_subplot(3, 1, 1)
    plt.xlabel("Along-strike distance (m)")
    plt.ylabel(r"Normal stress $\sigma$ (MPa)")
    time_text = ax1.text(0.7, 1.1, "Time = 0s", transform=ax1.transAxes)
    lines_normal_stress = []
    for i in range(fault.n_lines):
        i0 = np.sum(fault.n_patches_per_line[:i])
        i1 = i0 + fault.n_patches_per_line[i]
        lines_normal_stress.append(
            ax1.plot(
                fault.coords[i0:i1, 0], normal_stress[i0:i1, 0], drawstyle="steps-mid"
            )[0]
        )
    ax1.set_ylim(normal_stress.min() - 1.0, normal_stress.max() + 1.0)

    ax2 = fig.add_subplot(3, 1, 2)
    plt.xlabel("Along-strike distance (m)")
    plt.ylabel(r"Shear stress $\tau$ (MPa)")
    lines_shear_stress = []
    for i in range(fault.n_lines):
        i0 = np.sum(fault.n_patches_per_line[:i])
        i1 = i0 + fault.n_patches_per_line[i]
        lines_shear_stress.append(
            ax2.plot(
                fault.coords[i0:i1, 0], shear_stress[i0:i1, 0], drawstyle="steps-mid"
            )[0]
        )
    ax2.set_ylim(shear_stress.min() - 1.0, shear_stress.max() + 1.0)

    ax3 = fig.add_subplot(3, 1, 3)
    plt.xlabel("Along-strike distance (m)")
    plt.ylabel(r"Slip $\delta$ (m)")
    lines_slip = []
    lines_tectonic_loading = []
    for i in range(fault.n_lines):
        i0 = np.sum(fault.n_patches_per_line[:i])
        i1 = i0 + fault.n_patches_per_line[i]
        lines_slip.append(
            ax3.plot(
                fault.coords[i0:i1, 0],
                slip[i0:i1, 0],
                drawstyle="steps-mid",
                label="Coseismic slip",
                color=color_cycle[i],
            )[0]
        )
        lines_tectonic_loading.append(
            ax3.plot(
                fault.coords[i0:i1, 0],
                np.zeros(fault.n_patches_per_line[i]),
                drawstyle="steps-mid",
                ls="--",
                label="Tectonic loading",
                color=color_cycle[i],
            )[0]
        )
    ax3.legend(loc="upper right", fancybox=True)
    ax3.set_ylim(
        slip.min(),
        max(
            fault.tectonic_slip_speeds.max() * time[-1] * year,
            slip.max() + 0.1 * slip.max(),
        ),
    )

    plt.subplots_adjust(
        top=0.95, bottom=0.065, left=0.125, right=0.9, hspace=0.24, wspace=0.2
    )

    def animate_stress_slip(i):
        for j in range(fault.n_lines):
            j0 = np.sum(fault.n_patches_per_line[:j])
            j1 = j0 + fault.n_patches_per_line[j]
            lines_normal_stress[j].set_ydata(normal_stress[j0:j1, i])
            lines_shear_stress[j].set_ydata(shear_stress[j0:j1, i])
            lines_slip[j].set_ydata(slip[j0:j1, i])
            lines_tectonic_loading[j].set_ydata(
                fault.tectonic_slip_speeds[j0:j1] * time[i] * year
            )
        time_text.set_text("Time = {:.2f}y".format(time[i]))
        concat_lines = []
        concat_lines.extend(lines_normal_stress)
        concat_lines.extend(lines_shear_stress)
        concat_lines.extend(lines_slip)
        concat_lines.append(lines_tectonic_loading)
        return concat_lines

    anim = animation.FuncAnimation(
        fig, animate_stress_slip, frames=time.size, interval=10, save_count=None
    )
    return anim


def plot_final_stress_slip(fault, idx_history=-1, show=False):
    shear_stress = np.zeros(fault.n_patches, dtype=np.float64)
    normal_stress = np.zeros(fault.n_patches, dtype=np.float64)
    slip = np.zeros(fault.n_patches, dtype=np.float64)

    for i in range(fault.n_patches):
        shear_stress[i] = fault.fault_patches[i].shear_stress_history[idx_history]
        normal_stress[i] = fault.fault_patches[i].normal_stress_history[idx_history]
        slip[i] = fault.fault_patches[i].coseismic_slip_history[idx_history]

    plt.figure("stress_slip_along_fault")

    plt.suptitle("Final stress and slip along the fault (X-direction)")

    plt.subplot(3, 1, 1)
    for i in range(fault.n_lines):
        i0 = np.sum(fault.n_patches_per_line[:i])
        i1 = i0 + fault.n_patches_per_line[i]
        plt.plot(fault.coords[i0:i1, 0], shear_stress[i0:i1] / 1.0e6)
    plt.ylabel(r"Shear Stress $\tau$ (MPa)")
    plt.xlim(fault.coords[:, 0].min(), fault.coords[:, 0].max())

    plt.subplot(3, 1, 2)
    for i in range(fault.n_lines):
        i0 = np.sum(fault.n_patches_per_line[:i])
        i1 = i0 + fault.n_patches_per_line[i]
        plt.plot(fault.coords[i0:i1, 0], normal_stress[i0:i1] / 1.0e6)
    plt.ylabel(r"Normal Stress $\sigma$ (MPa)")
    plt.xlim(fault.coords[:, 0].min(), fault.coords[:, 0].max())

    plt.subplot(3, 1, 3)
    for i in range(fault.n_lines):
        i0 = np.sum(fault.n_patches_per_line[:i])
        i1 = i0 + fault.n_patches_per_line[i]
        plt.plot(fault.coords[i0:i1, 0], slip[i0:i1])
    plt.ylabel(r"Slip $\delta$ (m)")
    plt.xlabel("X (m)")
    plt.xlim(fault.coords[:, 0].min(), fault.coords[:, 0].max())

    plt.subplots_adjust(
        top=0.935, bottom=0.07, left=0.125, right=0.9, hspace=0.2, wspace=0.2
    )
    if show:
        plt.show()


def make_animation_coulomb_events(
    shear_stress, normal_stress, time, catalog, fault, animation_speed=30
):
    coulomb_stress = np.zeros((fault.n_patches, time.size), dtype=np.float64)
    for n in range(fault.n_patches):
        coulomb_stress[n, :] = (
            shear_stress[n, :]
            - (fault.fault_patches[n].mu_0 - fault.fault_patches[n].alpha)
            * normal_stress[n, :]
        )
    cm = plt.get_cmap("jet")
    cNorm = Normalize(vmin=coulomb_stress.min(), vmax=coulomb_stress.max())
    scalarMap = ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure("coulomb_stress_vs_events")
    ax = plt.gca()
    lines = []
    for n in range(fault.n_patches):
        Lx = fault.fault_patches[n].L * np.sin(
            fault.fault_patches[n].strike_angle * np.pi / 180.0
        )
        Ly = fault.fault_patches[n].L * np.cos(
            fault.fault_patches[n].strike_angle * np.pi / 180.0
        )
        X = [fault.coords[n, 0] - Lx / 2.0, fault.coords[n, 0] + Lx / 2.0]
        Y = [fault.coords[n, 1] - Ly / 2.0, fault.coords[n, 1] + Ly / 2.0]
        lines.append(
            ax.plot(X, Y, color=scalarMap.to_rgba(coulomb_stress[n, 0]), lw=5)[0]
        )
    index_EQ = np.where(
        (catalog[:, 0] / year >= time[0]) & (catalog[:, 0] / year < time[1])
    )[0]
    colors = scalarMap.to_rgba(coulomb_stress[np.int32(catalog[index_EQ, 1]), 0])
    global scatter_plot
    scatter_plot = ax.scatter(
        catalog[index_EQ, 2],
        catalog[index_EQ, 3],
        s=150,
        color=colors,
        edgecolors="k",
        animated=True,
    )
    time_text = ax.text(0.7, 1.1, "Time = 0s", transform=ax.transAxes)
    plt.xlabel("X")
    plt.ylabel("Y")
    ax_cbar = fig.add_axes([0.91, 0.10, 0.01, 0.80])
    clb.ColorbarBase(
        ax_cbar,
        cmap=cm,
        norm=cNorm,
        label=r"Modified Coulomb Stress $S = \tau - \mu' \sigma$ (MPa)",
    )
    plt.subplots_adjust(right=0.90)

    def animate_coulomb_catalog(i):
        time_text.set_text("Time = {:.2f}y".format(time[i]))
        for n in range(fault.n_patches):
            lines[n].set_color(scalarMap.to_rgba(coulomb_stress[n, i]))
        global scatter_plot
        scatter_plot.remove()
        if i != time.size - 1:
            index_EQ = np.where(
                (catalog[:, 0] / year >= time[i]) & (catalog[:, 0] / year < time[i + 1])
            )[0]
        else:
            index_EQ = np.where(
                (catalog[:, 0] / year >= time[i])
                & (catalog[:, 0] / year < time[i] + (time[i] - time[i - 1]))
            )[0]
        colors = scalarMap.to_rgba(coulomb_stress[np.int32(catalog[index_EQ, 1]), i])
        scatter_plot = ax.scatter(
            catalog[index_EQ, 2],
            catalog[index_EQ, 3],
            s=150,
            color=colors,
            edgecolors="k",
        )
        return [lines, scatter_plot]

    anim = animation.FuncAnimation(
        fig, animate_coulomb_catalog, frames=time.size, interval=animation_speed
    )
    return anim


def make_axes_collections_3D(fault, figname=""):
    from matplotlib.patches import Polygon
    import matplotlib.gridspec as gridspec
    from matplotlib.collections import PatchCollection, LineCollection

    plt.figure(figname)
    grid1 = gridspec.GridSpec(2 + fault.n_lines / 2, 2)
    AXES = []
    LIST_PATCHES = []
    INDEXES_PER_LINE = []
    AXES.append(plt.subplot(grid1[0, :]))
    # --------------------------------------------------
    rectangles = []
    rectangle_colors = []
    for k in range(fault.n_lines):
        I = np.where(fault.line_indexes == k)[0]
        X = fault.coords[I, 0]
        x_offset = max(fault.fault_patches[0].L, 0.1 * (X.max() - X.min()))
        xmin = X.min() - x_offset
        xmax = X.max() + x_offset
        Y = fault.coords[I, 1]
        y_offset = max(fault.fault_patches[0].L, 0.1 * (Y.max() - Y.min()))
        ymin = Y.min() - y_offset
        ymax = Y.max() + y_offset
        C = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        rectangles.append(Polygon(C))
        rectangle_colors.append(color_cycle[k])
    AXES[-1].add_collection(
        PatchCollection(
            rectangles, edgecolors=rectangle_colors, facecolors="none", lw=2
        )
    )
    # --------------------------------------------------
    C = np.zeros((fault.n_patches, 2, 2), dtype=np.float64)
    for n in range(fault.n_patches):
        Lx = fault.fault_patches[n].L * np.sin(
            fault.fault_patches[n].strike_angle * np.pi / 180.0
        )
        Ly = fault.fault_patches[n].L * np.cos(
            fault.fault_patches[n].strike_angle * np.pi / 180.0
        )
        X = [fault.coords[n, 0] - Lx / 2.0, fault.coords[n, 0] + Lx / 2.0]
        Y = [fault.coords[n, 1] - Ly / 2.0, fault.coords[n, 1] + Ly / 2.0]
        C[n, :, 0] = X
        C[n, :, 1] = Y
    # LIST_PATCHES.append(LineCollection(C, cmap=cm, norm=cNorm, lw=2))
    LIST_PATCHES.append(C)
    # ax1.add_collection(collection_lines)
    # collection_lines.set_array(coulomb_stress[:,0])
    # time_text = ax1.text(0.7, 1.1, 'Time = 0s', transform=ax1.transAxes)
    X = fault.coords[:, 0]
    xmin = X.min() - 3.0 * fault.fault_patches[0].L
    xmax = X.max() + 3.0 * fault.fault_patches[0].L
    Y = fault.coords[:, 1]
    ymin = Y.min() - 3.0 * fault.fault_patches[0].L
    ymax = Y.max() + 3.0 * fault.fault_patches[0].L
    AXES[-1].set_xlim(xmin, xmax)
    AXES[-1].set_ylim(ymin, ymax)
    AXES[-1].set_xlabel("X")
    AXES[-1].set_ylabel("Y")
    grid1.update(bottom=0.15)
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    #           FIRST SEGMENT
    I = np.where(fault.line_indexes == 0)[0]
    AXES.append(plt.subplot2grid((2 + fault.n_lines / 2, 2), (1, 0), colspan=2))
    patches = []
    for n in range(I.size):
        C = np.zeros((4, 3), dtype=np.float64)
        C[0, :] = (
            fault.coords[I[n], :]
            + fault.fault_patches[I[n]].L / 2.0 * fault.fault_patches[I[n]].t_h
            + fault.fault_patches[I[n]].W / 2.0 * fault.fault_patches[I[n]].t_v
        )
        C[1, :] = (
            fault.coords[I[n], :]
            - fault.fault_patches[I[n]].L / 2.0 * fault.fault_patches[I[n]].t_h
            + fault.fault_patches[I[n]].W / 2.0 * fault.fault_patches[I[n]].t_v
        )
        C[2, :] = (
            fault.coords[I[n], :]
            - fault.fault_patches[I[n]].L / 2.0 * fault.fault_patches[I[n]].t_h
            - fault.fault_patches[I[n]].W / 2.0 * fault.fault_patches[I[n]].t_v
        )
        C[3, :] = (
            fault.coords[I[n], :]
            + fault.fault_patches[I[n]].L / 2.0 * fault.fault_patches[I[n]].t_h
            - fault.fault_patches[I[n]].W / 2.0 * fault.fault_patches[I[n]].t_v
        )
        C = np.round(C, decimals=2)
        rectangle = Polygon(C[:, np.array([0, 2])])
        patches.append(rectangle)
    # LIST_PATCHES.append(PatchCollection(patches, cmap=cm, norm=cNorm))
    # collection_patches.set_array(coulomb_stress[I, 0])
    # ax2.add_collection(collection_patches)
    X = fault.coords[I, 0]
    Y = fault.coords[I, 2]
    AXES[-1].set_xlim(
        X.min() - fault.fault_patches[0].L, X.max() + fault.fault_patches[0].L
    )
    AXES[-1].set_ylim(
        Y.min() - fault.fault_patches[0].W, Y.max() + fault.fault_patches[0].W
    )
    AXES[-1].set_ylabel("Depth (m)")
    plt.setp(AXES[-1].spines.values(), color=rectangle_colors[0], lw=2)
    plt.setp(
        [AXES[-1].get_xticklines(), AXES[-1].get_yticklines()],
        color=rectangle_colors[0],
    )
    LIST_PATCHES.append(patches)
    INDEXES_PER_LINE.append(I)
    # ----------------------------------------------------------------------
    for k in range(fault.n_lines - 1):
        AXES.append(plt.subplot2grid((2 + fault.n_lines / 2, 2), (2 + k / 2, k % 2)))
        I = np.where(fault.line_indexes == k + 1)[0]
        patches = []
        for n in range(I.size):
            C = np.zeros((4, 3), dtype=np.float64)
            C[0, :] = (
                fault.coords[I[n], :]
                + fault.fault_patches[I[n]].L / 2.0 * fault.fault_patches[I[n]].t_h
                + fault.fault_patches[I[n]].W / 2.0 * fault.fault_patches[I[n]].t_v
            )
            C[1, :] = (
                fault.coords[I[n], :]
                - fault.fault_patches[I[n]].L / 2.0 * fault.fault_patches[I[n]].t_h
                + fault.fault_patches[I[n]].W / 2.0 * fault.fault_patches[I[n]].t_v
            )
            C[2, :] = (
                fault.coords[I[n], :]
                - fault.fault_patches[I[n]].L / 2.0 * fault.fault_patches[I[n]].t_h
                - fault.fault_patches[I[n]].W / 2.0 * fault.fault_patches[I[n]].t_v
            )
            C[3, :] = (
                fault.coords[I[n], :]
                + fault.fault_patches[I[n]].L / 2.0 * fault.fault_patches[I[n]].t_h
                - fault.fault_patches[I[n]].W / 2.0 * fault.fault_patches[I[n]].t_v
            )
            C = np.round(C, decimals=2)
            rectangle = Polygon(C[:, np.array([0, 2])])
            patches.append(rectangle)
        # collection_patches = PatchCollection(patches, cmap=cm, norm=cNorm)
        # collection_patches.set_array(coulomb_stress[I, 0])
        # AXES[-1].add_collection(collection_patches)
        X = fault.coords[I, 0]
        Y = fault.coords[I, 2]
        AXES[-1].set_xlim(
            X.min() - fault.fault_patches[0].L, X.max() + fault.fault_patches[0].L
        )
        AXES[-1].set_ylim(
            Y.min() - fault.fault_patches[0].W, Y.max() + fault.fault_patches[0].W
        )
        LIST_PATCHES.append(patches)
        INDEXES_PER_LINE.append(I)
        plt.setp(AXES[-1].spines.values(), color=rectangle_colors[k + 1], lw=2)
        plt.setp(
            [AXES[-1].get_xticklines(), AXES[-1].get_yticklines()],
            color=rectangle_colors[k + 1],
        )
        if k / 2 == fault.n_lines / 2 - 1:
            AXES[-1].set_xlabel("Along strike distance (m)")
        if k % 2 == 0:
            AXES[-1].set_ylabel("Depth (m)")
    # ----------------------------------------------------------------------
    plt.subplots_adjust(
        top=0.945, bottom=0.065, left=0.1, right=0.925, hspace=0.31, wspace=0.2
    )
    return AXES, LIST_PATCHES, INDEXES_PER_LINE


def plot_Kij_shear_3D(fault, patch_index):
    AXES, LIST_PATCHES, INDEXES_PER_LINE = make_axes_collections_3D(
        fault, figname="Kij_shear_patch{:d}".format(patch_index)
    )
    cm = plt.get_cmap("coolwarm")
    Kij_tau = fault.Kij_shear[:, patch_index] / 1.0e6
    mask = np.ones(Kij_tau.size, dtype=np.bool)
    mask[patch_index] = False
    boundary = max(np.abs(Kij_tau[mask].min()), np.abs(Kij_tau.max()))
    boundary_linear = min(np.abs(Kij_tau[mask].min()), np.abs(Kij_tau.max()))
    cNorm = SymLogNorm(
        vmin=-boundary,
        vmax=+boundary,
        linthresh=boundary_linear / 10000.0,
        linscale=1.0,
    )
    scalarMap = ScalarMappable(norm=cNorm, cmap=cm)
    AXES[0].add_collection(
        LineCollection(
            LIST_PATCHES[0][::-1], cmap=cm, norm=cNorm, lw=2, array=Kij_tau[::-1]
        )
    )
    for i in range(1, len(AXES)):
        edgecolors = ["none" for j in range(INDEXES_PER_LINE[i - 1].size)]
        if patch_index in INDEXES_PER_LINE[i - 1]:
            edgecolors[np.where(INDEXES_PER_LINE[i - 1] == patch_index)[0][0]] = "k"
        AXES[i].add_collection(
            PatchCollection(
                LIST_PATCHES[i],
                cmap=cm,
                norm=cNorm,
                array=Kij_tau[INDEXES_PER_LINE[i - 1]],
                edgecolors=edgecolors,
            )
        )
    fig = plt.gcf()
    ax_cbar = fig.add_axes([0.92, 0.10, 0.01, 0.40])
    clb.ColorbarBase(ax_cbar, cmap=cm, norm=cNorm, label=r"$K_{ij}^{\tau}$ (MPa/m)")
    plt.subplots_adjust(right=0.90)
    plt.show()


def plot_a_b_3D(fault):
    AXES, LIST_PATCHES, INDEXES_PER_LINE = make_axes_collections_3D(
        fault, figname="b_a"
    )
    xi = np.float64([(fp.b - fp.a) for fp in fault.fault_patches])
    cm = plt.get_cmap("coolwarm_r")
    cNorm = Normalize(vmin=xi.min(), vmax=xi.max())
    AXES[0].add_collection(
        LineCollection(LIST_PATCHES[0], cmap=cm, norm=cNorm, lw=2, array=xi)
    )
    for i in range(1, len(AXES)):
        AXES[i].add_collection(
            PatchCollection(
                LIST_PATCHES[i], cmap=cm, norm=cNorm, array=xi[INDEXES_PER_LINE[i - 1]]
            )
        )
    fig = plt.gcf()
    ax_cbar = fig.add_axes([0.92, 0.10, 0.01, 0.80])
    clb.ColorbarBase(ax_cbar, cmap=cm, norm=cNorm, label=r"$\xi = b - a$")
    plt.subplots_adjust(right=0.90)
    plt.show()


def make_animation_coulomb_events_3D(
    shear_stress, normal_stress, time, catalog, fault, animation_speed=30
):
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection, LineCollection
    import matplotlib.gridspec as gridspec

    coulomb_stress = np.zeros((fault.n_patches, time.size), dtype=np.float64)
    for n in range(fault.n_patches):
        coulomb_stress[n, :] = (
            shear_stress[n, :]
            - (fault.fault_patches[n].mu_0 - fault.fault_patches[n].alpha)
            * normal_stress[n, :]
        )
    cm = plt.get_cmap("coolwarm")
    cNorm = Normalize(
        vmin=max(coulomb_stress.min(), 0.0), vmax=np.percentile(coulomb_stress, 98.0)
    )
    scalarMap = ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure("coulomb_stress_vs_events", figsize=(19, 10))
    grid1 = gridspec.GridSpec(2 + fault.n_lines / 2, 2)
    ax1 = plt.subplot(grid1[0, :])
    # --------------------------------------------------
    rectangles = []
    rectangle_colors = []
    for k in range(fault.n_lines):
        I = np.where(fault.line_indexes == k)[0]
        X = fault.coords[I, 0]
        xmin = X.min() - 2.0 * fault.fault_patches[0].L
        xmax = X.max() + 2.0 * fault.fault_patches[0].L
        Y = fault.coords[I, 1]
        ymin = Y.min() - 2.0 * fault.fault_patches[0].L
        ymax = Y.max() + 2.0 * fault.fault_patches[0].L
        C = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        rectangles.append(Polygon(C))
        rectangle_colors.append(color_cycle[k])
    ax1.add_collection(
        PatchCollection(
            rectangles, edgecolors=rectangle_colors, facecolors="none", lw=2
        )
    )
    # --------------------------------------------------
    C = np.zeros((fault.n_patches, 2, 2), dtype=np.float64)
    for n in range(fault.n_patches):
        Lx = fault.fault_patches[n].L * np.sin(
            fault.fault_patches[n].strike_angle * np.pi / 180.0
        )
        Ly = fault.fault_patches[n].L * np.cos(
            fault.fault_patches[n].strike_angle * np.pi / 180.0
        )
        X = [fault.coords[n, 0] - Lx / 2.0, fault.coords[n, 0] + Lx / 2.0]
        Y = [fault.coords[n, 1] - Ly / 2.0, fault.coords[n, 1] + Ly / 2.0]
        C[n, :, 0] = X
        C[n, :, 1] = Y
    collection_lines = LineCollection(C, cmap=cm, norm=cNorm, lw=2)
    ax1.add_collection(collection_lines)
    collection_lines.set_array(coulomb_stress[:, 0])
    time_text = ax1.text(0.7, 1.1, "Time = 0s", transform=ax1.transAxes)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    grid1.update(bottom=0.15)
    # ----------------------------------------------------------------------
    COLLECTIONS = []
    INDEXES_PER_LINE = []
    # ----------------------------------------------------------------------
    #           FIRST SEGMENT
    I = np.where(fault.line_indexes == 0)[0]
    ax2 = plt.subplot2grid((2 + fault.n_lines / 2, 2), (1, 0), colspan=2)
    patches = []
    for n in range(I.size):
        C = np.zeros((4, 3), dtype=np.float64)
        C[0, :] = (
            fault.coords[I[n], :]
            + fault.fault_patches[I[n]].L / 2.0 * fault.fault_patches[I[n]].t_h
            + fault.fault_patches[I[n]].W / 2.0 * fault.fault_patches[I[n]].t_v
        )
        C[1, :] = (
            fault.coords[I[n], :]
            - fault.fault_patches[I[n]].L / 2.0 * fault.fault_patches[I[n]].t_h
            + fault.fault_patches[I[n]].W / 2.0 * fault.fault_patches[I[n]].t_v
        )
        C[2, :] = (
            fault.coords[I[n], :]
            - fault.fault_patches[I[n]].L / 2.0 * fault.fault_patches[I[n]].t_h
            - fault.fault_patches[I[n]].W / 2.0 * fault.fault_patches[I[n]].t_v
        )
        C[3, :] = (
            fault.coords[I[n], :]
            + fault.fault_patches[I[n]].L / 2.0 * fault.fault_patches[I[n]].t_h
            - fault.fault_patches[I[n]].W / 2.0 * fault.fault_patches[I[n]].t_v
        )
        rectangle = Polygon(C[:, np.array([0, 2])])
        patches.append(rectangle)
    collection_patches = PatchCollection(patches, cmap=cm, norm=cNorm)
    collection_patches.set_array(coulomb_stress[I, 0])
    ax2.add_collection(collection_patches)
    X = fault.coords[I, 0]
    Y = fault.coords[I, 2]
    ax2.set_xlim(X.min() - fault.fault_patches[0].L, X.max() + fault.fault_patches[0].L)
    ax2.set_ylim(Y.min() - fault.fault_patches[0].W, Y.max() + fault.fault_patches[0].W)
    ax2.set_ylabel("Depth (m)")
    plt.setp(ax2.spines.values(), color=rectangle_colors[0], lw=2)
    plt.setp([ax2.get_xticklines(), ax2.get_yticklines()], color=rectangle_colors[0])
    COLLECTIONS.append(collection_patches)
    INDEXES_PER_LINE.append(I)
    # ----------------------------------------------------------------------
    AXES = []
    for k in range(fault.n_lines - 1):
        AXES.append(plt.subplot2grid((2 + fault.n_lines / 2, 2), (2 + k / 2, k % 2)))
        I = np.where(fault.line_indexes == k + 1)[0]
        patches = []
        for n in range(I.size):
            C = np.zeros((4, 3), dtype=np.float64)
            C[0, :] = (
                fault.coords[I[n], :]
                + fault.fault_patches[I[n]].L / 2.0 * fault.fault_patches[I[n]].t_h
                + fault.fault_patches[I[n]].W / 2.0 * fault.fault_patches[I[n]].t_v
            )
            C[1, :] = (
                fault.coords[I[n], :]
                - fault.fault_patches[I[n]].L / 2.0 * fault.fault_patches[I[n]].t_h
                + fault.fault_patches[I[n]].W / 2.0 * fault.fault_patches[I[n]].t_v
            )
            C[2, :] = (
                fault.coords[I[n], :]
                - fault.fault_patches[I[n]].L / 2.0 * fault.fault_patches[I[n]].t_h
                - fault.fault_patches[I[n]].W / 2.0 * fault.fault_patches[I[n]].t_v
            )
            C[3, :] = (
                fault.coords[I[n], :]
                + fault.fault_patches[I[n]].L / 2.0 * fault.fault_patches[I[n]].t_h
                - fault.fault_patches[I[n]].W / 2.0 * fault.fault_patches[I[n]].t_v
            )
            rectangle = Polygon(C[:, np.array([0, 2])])
            patches.append(rectangle)
        collection_patches = PatchCollection(patches, cmap=cm, norm=cNorm)
        collection_patches.set_array(coulomb_stress[I, 0])
        AXES[-1].add_collection(collection_patches)
        X = fault.coords[I, 0]
        Y = fault.coords[I, 2]
        AXES[-1].set_xlim(
            X.min() - fault.fault_patches[0].L, X.max() + fault.fault_patches[0].L
        )
        AXES[-1].set_ylim(
            Y.min() - fault.fault_patches[0].W, Y.max() + fault.fault_patches[0].W
        )
        COLLECTIONS.append(collection_patches)
        INDEXES_PER_LINE.append(I)
        plt.setp(AXES[-1].spines.values(), color=rectangle_colors[k + 1], lw=2)
        plt.setp(
            [AXES[-1].get_xticklines(), AXES[-1].get_yticklines()],
            color=rectangle_colors[k + 1],
        )
        if k / 2 == fault.n_lines / 2 - 1:
            AXES[-1].set_xlabel("Along strike distance (m)")
        if k % 2 == 0:
            AXES[-1].set_ylabel("Depth (m)")
    # ----------------------------------------------------------------------
    plt.subplots_adjust(
        top=0.945, bottom=0.065, left=0.1, right=0.925, hspace=0.31, wspace=0.2
    )
    ax_cbar = fig.add_axes([0.94, 0.10, 0.01, 0.80])
    clb.ColorbarBase(
        ax_cbar,
        cmap=cm,
        norm=cNorm,
        label=r"Modified Coulomb Stress $S = \tau - \mu' \sigma$ (MPa)",
    )
    plt.subplots_adjust(right=0.90)
    # ----------------------------------------------------------------------
    index_EQ = np.where(
        (catalog[:, 0] / year >= time[0]) & (catalog[:, 0] / year < time[1])
    )[0]
    colors = scalarMap.to_rgba(coulomb_stress[np.int32(catalog[index_EQ, 1]), 0])
    global scatter_plot
    scatter_plot = ax1.scatter(
        catalog[index_EQ, 2],
        catalog[index_EQ, 3],
        s=150,
        color=colors,
        edgecolors="k",
        animated=True,
    )

    # global scatter_plots = []
    # scatter_plots.append(ax1.scatter(catalog[index_EQ,2], catalog[index_EQ,3], s=150, color=colors, edgecolors='k', animated=True))
    # index_EQ_line1 = np.int32([idx if idx in INDEXES_PER_LINE[0] for idx in index_EQ])
    # scatter_plots.append(ax2.scatter(fault.fault_coords[np.int32(catalog[index_EQ_line1, 1])], catalog[index_EQ,3], s=150, color=colors, edgecolors='k', animated=True))
    def animate_coulomb_catalog(i):
        time_text.set_text("Time = {:.2f}y".format(time[i]))
        collection_lines.set_array(coulomb_stress[:, i])
        global scatter_plot
        scatter_plot.remove()
        if i != time.size - 1:
            index_EQ = np.where(
                (catalog[:, 0] / year >= time[i]) & (catalog[:, 0] / year < time[i + 1])
            )[0]
        else:
            index_EQ = np.where(
                (catalog[:, 0] / year >= time[i])
                & (catalog[:, 0] / year < time[i] + (time[i] - time[i - 1]))
            )[0]
        patches_index_EQ = np.int32(catalog[index_EQ, 1])
        colors = scalarMap.to_rgba(coulomb_stress[patches_index_EQ, i])
        scatter_plot = ax1.scatter(
            catalog[index_EQ, 2],
            catalog[index_EQ, 3],
            s=150,
            color=colors,
            edgecolors="k",
        )
        for k in range(len(COLLECTIONS)):
            COLLECTIONS[k].set_array(coulomb_stress[INDEXES_PER_LINE[k], i])
            index_EQ_line = []
            for idx in patches_index_EQ:
                if idx in INDEXES_PER_LINE[k]:
                    index_EQ_line.append(np.where(INDEXES_PER_LINE[k] == idx)[0][0])
            index_EQ_line = np.int32(index_EQ_line)
            edgecolors = ["none" for j in range(INDEXES_PER_LINE[k].size)]
            for idx in index_EQ_line:
                edgecolors[idx] = "k"
            COLLECTIONS[k].set_edgecolor(edgecolors)
        return [scatter_plot]

    anim = animation.FuncAnimation(
        fig, animate_coulomb_catalog, frames=time.size, interval=animation_speed
    )
    return anim


def plot_stress_EQ_rate_single_patch(
    patch_index,
    shear_stress,
    normal_stress,
    time,
    fault,
    show=True,
    time_sliding_window=100.0,
    DT=3600.0 * 24.0,
):
    shear_stress = shear_stress[patch_index, :]
    normal_stress = normal_stress[patch_index, :]
    coulomb_stress = (
        shear_stress
        - (
            fault.fault_patches[patch_index].mu_0
            - fault.fault_patches[patch_index].alpha
        )
        * normal_stress
    )
    # --------------------------------------------------------
    catalog_ = get_event_history(fault)
    EQ_rate, new_time, cum_n_events, local_cat = empirical_earthquake_rate(
        fault, patch_index, radius=1.0, DT=DT
    )
    fig = plt.figure("EQ_rate_patch_{:d}".format(patch_index))
    plt.suptitle("Earthquake rate on Patch {:d}".format(patch_index))
    ax1 = fig.add_subplot(311)
    ax1.plot(
        new_time / (3600.0 * 24.0 * 365.0),
        cum_n_events,
        label="Cumulative Number of Events",
        color="C0",
    )
    ax1.set_ylabel("Cumulative Number of Events", color="C0")
    ax1.tick_params("y", color="C0", labelcolor="C0")
    ax1.set_xlabel("Time (year)")
    ax2 = ax1.twinx()
    ax2.plot(
        new_time / (3600.0 * 24.0 * 365.0), EQ_rate, color="C3", label="Earthquake Rate"
    )
    ax2.set_ylabel("Earthquake Rate", color="C3")
    ax2.tick_params("y", color="C3", labelcolor="C3")
    # ---------------------------------
    ax3 = fig.add_subplot(312)
    theoretical_EQ_rate, time_ = earthquake_rate(fault.fault_patches[patch_index])
    ax3.plot(
        new_time / (3600.0 * 24.0 * 365.0),
        EQ_rate,
        color="C3",
        ls="--",
        lw=0.75,
        label="Measured EQ Rate",
    )
    ax3.plot(
        time_ / (3600.0 * 24.0 * 365.0),
        theoretical_EQ_rate,
        color="k",
        ls="--",
        label="Theoretical EQ Rate",
    )
    ax3.set_ylabel("Earthquake Rate")
    ax3.set_xlabel("Time (year)")
    ax3.legend(loc="upper left", fancybox=True)
    ax3.ticklabel_format(style="sci")
    ax4 = ax3.twinx()
    ax4.plot(time, shear_stress, label=r"Shear Stress $\tau$")
    ax4.plot(time, normal_stress, label=r"Normal Stress $\sigma$")
    ax4.plot(time, coulomb_stress, label=r"Coulomb Stress $S = \tau - \mu' \sigma$")
    ax4.set_ylabel("Stress (MPa)")
    ax4.legend(loc="lower left", fancybox=True)
    # ---------------------------------
    ax5 = fig.add_subplot(313)
    for n in range(fault.n_patches):
        Lx = fault.fault_patches[n].L * np.sin(
            fault.fault_patches[n].strike_angle * np.pi / 180.0
        )
        Ly = fault.fault_patches[n].L * np.cos(
            fault.fault_patches[n].strike_angle * np.pi / 180.0
        )
        X = [fault.coords[n, 0] - Lx / 2.0, fault.coords[n, 0] + Lx / 2.0]
        Y = [fault.coords[n, 1] - Ly / 2.0, fault.coords[n, 1] + Ly / 2.0]
        if n == patch_index:
            ax5.plot(
                X,
                Y,
                lw=5,
                color="C2",
                label="Patch {:d}".format(patch_index),
                zorder=fault.n_patches + 100,
            )
        else:
            ax5.plot(X, Y, lw=5, color="C0")
        ax5.set_xlabel("X (m)")
        ax5.set_ylabel("Y (m)")
        ax5.legend(loc="best", fancybox=True)
    plt.subplots_adjust(
        top=0.935, bottom=0.075, left=0.08, right=0.925, hspace=0.26, wspace=0.2
    )
    if show:
        plt.show()


def plot_stress_EQ_rate_group_patches(
    patch_index,
    shear_stress,
    normal_stress,
    time,
    fault,
    show=True,
    smoothing_box=100.0,
    R=500.0,
    DT=3600.0 * 24.0,
):
    # --------------------------------------------------------
    catalog_ = get_event_history(fault)
    R_empirical, new_time, cum_n_events, local_cat = empirical_earthquake_rate(
        fault, patch_index, radius=R, DT=DT, smoothing_box=smoothing_box
    )
    patch_indexes = np.int32(np.unique(local_cat[:, 1]))
    timings = local_cat[:, 0]
    # cum_n_events        = np.arange(timings.size)
    average_coulomb_stress = np.zeros(shear_stress.shape[-1], dtype=np.float64)
    for patch_id in patch_indexes:
        coulomb_stress = (
            shear_stress[patch_id, :]
            - (fault.fault_patches[patch_id].mu_0 - fault.fault_patches[patch_id].alpha)
            * normal_stress[patch_id, :]
        )
        average_coulomb_stress += coulomb_stress
    average_coulomb_stress /= np.float64(patch_indexes.size)
    # --------------------------------------------------------
    fig = plt.figure("EQ_rate_patch_{:d}".format(patch_index))
    plt.suptitle("Earthquake rate around Patch {:d}".format(patch_index))
    ax1 = fig.add_subplot(311)
    # ax1.plot(timings/(3600.*24.*365.), cum_n_events, label='Cumulative Number of Events', color='C0')
    ax1.plot(
        new_time / (3600.0 * 24.0 * 365.0),
        cum_n_events,
        label="Cumulative Number of Events",
        color="C0",
    )
    ax1.set_ylabel("Cumulative Number of Events", color="C0")
    ax1.tick_params("y", color="C0", labelcolor="C0")
    ax1.set_xlabel("Time (year)")
    ax2 = ax1.twinx()
    ax2.plot(
        new_time / (3600.0 * 24.0 * 365.0),
        R_empirical,
        color="C3",
        ls="--",
        label="Daily Earthquake Rate",
    )
    ax2.set_ylabel("Earthquake Rate", color="C3")
    ax2.tick_params("y", color="C3", labelcolor="C3")
    # ---------------------------------
    ax3 = fig.add_subplot(312)
    theoretical_EQ_rate, time_ = earthquake_rate(fault.fault_patches[patch_index])
    average_theoretical_EQ_rate = np.zeros_like(theoretical_EQ_rate)
    # a_average     = 0.
    # sigma_average = np.mean(normal_stress[np.int32(np.unique(local_cat[:,1])), :], axis=0)
    # S_dot_r       = 0.
    for patch_id in np.int32(np.unique(local_cat[:, 1])):
        # a_average += fault.fault_patches[patch_id].a
        # S_dot_r   += (fault.fault_patches[patch_id].ktau_tectonic - (fault.fault_patches[patch_id].mu_0 - fault.fault_patches[patch_id].alpha)) * fault.fault_patches[patch_id].tectonic_slip_speed
        # theoretical_EQ_rate, time_   = earthquake_rate(fault.fault_patches[patch_id], DT=DT)
        average_theoretical_EQ_rate += theoretical_EQ_rate
    average_theoretical_EQ_rate /= np.float64(np.unique(local_cat[:, 1]).size)
    # a_average /= np.float64(np.unique(local_cat[:,1]).size)
    # S_dot_r   /= np.float64(np.unique(local_cat[:,1]).size)
    # r          = cum_n_events[-1] / new_time[-1]
    # average_theoretical_EQ_rate = theoretical_average_earthquake_rate(S_dot_r, r, a_average, average_coulomb_stress, sigma_average, time)
    print(
        "Integrated empirical EQ rate: {:.1e}, Integrated theoretical EQ rate: {:.1e}".format(
            R_empirical.sum(), average_theoretical_EQ_rate.sum()
        )
    )
    ax3.plot(
        time_ / (3600.0 * 24.0 * 365.0),
        average_theoretical_EQ_rate,
        color="C2",
        ls="--",
        label="Theoretical EQ Rate",
    )
    # ax3.plot(time[:-1], average_theoretical_EQ_rate, color='C2', ls='--', label='Theoretical EQ Rate')
    ax3.plot(
        new_time / (3600.0 * 24.0 * 365.0),
        R_empirical,
        color="C3",
        ls="--",
        lw=0.75,
        label="Measured EQ Rate",
    )
    ax3.set_ylabel("Earthquake Rate")
    # ax3.semilogy()
    ax3.legend(loc="upper left", fancybox=True)
    # ax3.tick_params('y', color='C3', labelcolor='C3')
    ax3.ticklabel_format(style="sci")
    ax3.set_xlabel("Time (year)")
    ax4 = ax3.twinx()
    ax4.plot(
        time,
        average_coulomb_stress,
        label=r"Averaged Coulomb Stress $S = \tau - \mu' \sigma$",
    )
    ax4.set_ylabel("Av. Coulomb Stress (MPa)", color="C0")
    ax4.tick_params("y", color="C0", labelcolor="C0")
    # ax4.legend(loc='lower left', fancybox=True)
    # ---------------------------------
    ax5 = fig.add_subplot(313)
    for n in range(fault.n_patches):
        Lx = fault.fault_patches[n].L * np.sin(
            fault.fault_patches[n].strike_angle * np.pi / 180.0
        )
        Ly = fault.fault_patches[n].L * np.cos(
            fault.fault_patches[n].strike_angle * np.pi / 180.0
        )
        X = [fault.coords[n, 0] - Lx / 2.0, fault.coords[n, 0] + Lx / 2.0]
        Y = np.round(
            [fault.coords[n, 1] - Ly / 2.0, fault.coords[n, 1] + Ly / 2.0], decimals=2
        )
        if n == patch_index:
            ax5.plot(
                X,
                Y,
                lw=5,
                color="C2",
                label="Patch {:d}".format(patch_index),
                zorder=fault.n_patches + 200,
            )
        elif n in patch_indexes:
            ax5.plot(X, Y, lw=5, color="C3", zorder=fault.n_patches + 100)
        else:
            ax5.plot(X, Y, lw=5, color="C0")
        ax5.set_xlabel("X (m)")
        ax5.set_ylabel("Y (m)")
        ax5.legend(loc="best", fancybox=True)
    plt.subplots_adjust(
        top=0.935, bottom=0.075, left=0.08, right=0.925, hspace=0.26, wspace=0.2
    )
    if show:
        plt.show()


def plot_stress_slip_history_single_patch(patch_idx, fault):
    time = fault.fault_patches[patch_idx].time / year
    plt.figure("slip_stress_patch_{:d}".format(patch_idx))
    plt.suptitle("Slip and stress history on patch {:d}".format(patch_idx))
    # ---------------------------------------------
    ax = plt.subplot(2, 1, 1)
    plt.plot(
        time,
        fault.fault_patches[patch_idx].coseismic_slip_history,
        color="C0",
        label="Coseismic slip",
    )
    plt.plot(
        time,
        fault.fault_patches[patch_idx].time
        * fault.fault_patches[patch_idx].tectonic_slip_speed,
        color="C3",
        label="Tectonic loading",
    )
    plt.xlim(time.min(), time.max())
    plt.legend(loc="best", fancybox=True)
    plt.xlabel("Time (year)")
    plt.ylabel("Displacement (m)")
    # ---------------------------------------------
    plt.subplot(2, 1, 2, sharex=ax)
    plt.plot(
        time,
        fault.fault_patches[patch_idx].shear_stress_history / 1.0e6,
        label=r"Shear stress $\tau$ (MPa)",
    )
    plt.plot(
        time,
        fault.fault_patches[patch_idx].normal_stress_history / 1.0e6,
        label=r"Normal stress $\sigma$ (MPa)",
    )
    Coulomb = (
        fault.fault_patches[patch_idx].shear_stress_history
        - (fault.fault_patches[patch_idx].mu_0 - fault.fault_patches[patch_idx].alpha)
        * fault.fault_patches[patch_idx].normal_stress_history
    )
    plt.plot(time, Coulomb / 1.0e6, label=r"Modified Coulomb stress (MPa)")
    plt.xlim(time.min(), time.max())
    plt.xlabel("Time (year)")
    plt.ylabel(r"Stress (MPa)")
    plt.legend(loc="best", fancybox=True)
    # ---------------------------------------------
    plt.show()


def plot_geometry(fault):
    plt.figure("faults_network")
    for n in range(fault.n_patches):
        # Lx = fault.fault_patches[n].L*np.sin(fault.fault_patches[n].strike_angle*np.pi/180.)
        # Ly = fault.fault_patches[n].L*np.cos(fault.fault_patches[n].strike_angle*np.pi/180.)
        Lx = fault.fault_patches[n].L * fault.fault_patches[n].t_h[0]
        Ly = fault.fault_patches[n].L * fault.fault_patches[n].t_h[1]
        X = [fault.coords[n, 0] - Lx / 2.0, fault.coords[n, 0] + Lx / 2.0]
        Y = [fault.coords[n, 1] - Ly / 2.0, fault.coords[n, 1] + Ly / 2.0]
        plt.plot(X, Y, color="C0", lw=2)
        # plt.text(np.mean(X), np.mean(Y), str(n))
    plt.gca().set_aspect("equal", "box")
    plt.ylim(fault.coords[:, 1].min() - 100.0, fault.coords[:, 1].max() + 100.0)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.show()


def plot_geometry_3D(fault):
    import mpl_toolkits.mplot3d as a3

    ax = a3.Axes3D(plt.figure("faults_network_3D"))
    for n in range(fault.n_patches):
        C = np.zeros((4, 3), dtype=np.float64)
        t_v_modified = fault.fault_patches[n].t_v
        C[0, :] = (
            fault.coords[n, :]
            + fault.fault_patches[n].L / 2.0 * fault.fault_patches[n].t_h
            + fault.fault_patches[n].W / 2.0 * t_v_modified
        )
        C[1, :] = (
            fault.coords[n, :]
            - fault.fault_patches[n].L / 2.0 * fault.fault_patches[n].t_h
            + fault.fault_patches[n].W / 2.0 * t_v_modified
        )
        C[2, :] = (
            fault.coords[n, :]
            - fault.fault_patches[n].L / 2.0 * fault.fault_patches[n].t_h
            - fault.fault_patches[n].W / 2.0 * t_v_modified
        )
        C[3, :] = (
            fault.coords[n, :]
            + fault.fault_patches[n].L / 2.0 * fault.fault_patches[n].t_h
            - fault.fault_patches[n].W / 2.0 * t_v_modified
        )
        x = list(C[:, 0])
        y = list(C[:, 1])
        z = list(C[:, 2])
        rectangle = a3.art3d.Poly3DCollection([list(zip(x, y, z))])
        rectangle.set_edgecolor("k")
        ax.add_collection3d(rectangle)
    ax.set_xlim(
        fault.coords[:, 0].min() - 1.0 * fault.fault_patches[0].L,
        fault.coords[:, 0].max() + 1.0 * fault.fault_patches[0].L,
    )
    ax.set_ylim(
        fault.coords[:, 1].min() - 3.0 * fault.fault_patches[0].L,
        fault.coords[:, 1].max() + 3.0 * fault.fault_patches[0].L,
    )
    ax.set_zlim(
        fault.coords[:, 2].min() - 3.0 * fault.fault_patches[0].L,
        fault.coords[:, 2].max() + 3.0 * fault.fault_patches[0].L,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_aspect("equal", "box")
    plt.show()


# ========================================================================================================
#                                   EARTHQUAKE RATE ANALYSIS FUNCTIONS
# ========================================================================================================
def modified_coulomb_stress(shear_stress, normal_stress, alpha, mu_0):
    mu_effective = np.zeros(shear_stress.size, dtype=np.float64)
    mu_effective[0] = mu_0 - alpha
    for i in range(1, mu_effective.size):
        mu_effective[i] = shear_stress[i - 1] / normal_stress[i - 1] - alpha
    coulomb_stress = shear_stress - mu_effective * normal_stress
    return coulomb_stress, mu_effective


def get_coulomb_stress(fault_patch):
    time, unique_indexes = np.unique(fault_patch.time, return_index=True)
    S = (
        fault_patch.shear_stress_history[unique_indexes]
        - (fault_patch.mu_0 - fault_patch.alpha)
        * fault_patch.normal_stress_history[unique_indexes]
    )
    S_dot = (S[1:] - S[:-1]) / (time[1:] - time[:-1])
    # ----------------------------------------------------
    return S, S_dot, time


def state_variable_gamma(fault_patch):
    S, S_dot, time = get_coulomb_stress(fault_patch)
    tectonic_shear_stressing_rate = (
        fault_patch.ktau_tectonic * fault_patch.tectonic_slip_speed
    )
    tectonic_normal_stressing_rate = (
        fault_patch.ksigma_tectonic * fault_patch.tectonic_slip_speed
    )
    gamma_0 = 1.0 / (
        tectonic_shear_stressing_rate
        - fault_patch.mu_0 * tectonic_normal_stressing_rate
    )
    gamma = np.zeros(time.size, dtype=np.float64)
    gamma[0] = gamma_0
    for i in range(1, time.size):
        t_a = fault_patch.a * fault_patch.normal_stress_history[i] / S_dot[i - 1]
        gamma[i] = (gamma[i - 1] - 1.0 / S_dot[i - 1]) * np.exp(
            -(time[i] - time[i - 1]) / t_a
        ) + 1.0 / S_dot[i - 1]
    # gamma[np.isnan(gamma)] = gamma_0
    return gamma


def earthquake_rate(fault_patch, DT=3600.0 * 24.0):
    S, S_dot, time = get_coulomb_stress(fault_patch)
    gamma = state_variable_gamma(fault_patch)
    S_dot_0 = 1.0 / gamma[0]
    # R        = 1./(gamma * np.hstack( ([S_dot_0], S_dot) ))
    R = 1.0 / (gamma * S_dot_0)
    r = fault_patch.event_timings.size / fault_patch.time[-1]  # average EQ rate
    # n = 0
    # while (time[n] - time[0]) < DT:
    #    n += 1
    # for i in range(0, R.size-n, n):
    #    # make the theoretical EQ rate comparable to the empirical EQ rate
    #    R[i:i+n] = R[i:i+n].sum()
    return r * R, time


def theoretical_average_earthquake_rate(
    S_dot_r, r, a_average, S_average, sigma_average, time
):
    # S_dot   = np.hstack( (S_dot_r, (S_average[1:] - S_average[:-1]) / (time[1:] - time[:-1])) )
    S_dot = (S_average[1:] - S_average[:-1]) / (time[1:] - time[:-1])
    t_a = a_average * sigma_average[:-1] / S_dot
    gamma_0 = 1.0 / S_dot_r
    gamma = np.zeros(time.size, dtype=np.float64)
    gamma[0] = gamma_0
    for i in range(time.size):
        gamma[i] = (gamma[i - 1] - 1.0 / S_dot[i - 1]) * np.exp(
            -(time[i] - time[i - 1]) / t_a[i - 1]
        ) + 1.0 / S_dot[i - 1]
    delta_t = time[1:] - time[:-1]
    R = (
        r
        * S_dot
        / S_dot_r
        / ((gamma[:-1] * S_dot - 1.0) * np.exp(-delta_t / t_a) + 1.0)
    )
    return R


def empirical_earthquake_rate(
    fault, patch_index, smoothing_box=1.0, DT=3600.0 * 24.0, radius=500.0
):
    """
    empirical_earthquake_rate(fault, patch_index, T_window=10)\n
    T_window is given in days.
    """
    catalog_ = get_event_history(fault)
    local_cat = local_catalog(catalog_, fault, patch_index, R=radius)
    patch_indexes = np.int32(np.unique(local_cat[:, 1]))
    timings = local_cat[:, 0]
    n_events_per_DT, time = np.histogram(
        timings,
        bins=int(fault.fault_patches[patch_index].time[-1] / DT),
        range=(0.0, fault.fault_patches[patch_index].time[-1]),
    )
    time = time[:-1]
    cum_n_events = np.cumsum(n_events_per_DT)
    if smoothing_box < DT:
        smoothing_box = DT
    box = np.ones(int(smoothing_box / DT), dtype=np.float64)
    box /= box.size
    R_empirical = np.float64(n_events_per_DT) / DT
    R_empirical = np.hstack(
        (
            np.convolve(R_empirical, box, mode="valid"),
            np.zeros(box.size - 1, dtype=np.float64),
        )
    )  # causal smoothing
    # R_empirical         = np.hstack( (R_empirical[:(box.size-1)], np.convolve(R_empirical, box, mode='valid')) ) # anti-causal smoothing
    R_empirical[np.isinf(R_empirical)] = 0.0
    return R_empirical, time, cum_n_events, local_cat


def invert_EQ_rate(fault, patch_index, T0=30.0):
    # --------------------------------------------------------
    catalog_ = get_event_history(fault)
    R_empirical, time, cum_n_events, local_cat = empirical_earthquake_rate(
        fault, patch_index
    )
    patch_indexes = np.int32(np.unique(local_cat[:, 1]))
    timings = local_cat[:, 0]
    # --------------------------------------------------------
    mask = time / year > T0
    R_empirical = R_empirical[mask]
    time = time[mask]
    # --------------------------------------------------------
    fault_patch = fault.fault_patches[patch_index]
    tectonic_shear_stressing_rate = (
        fault_patch.ktau_tectonic * fault_patch.tectonic_slip_speed
    )
    tectonic_normal_stressing_rate = (
        fault_patch.ksigma_tectonic * fault_patch.tectonic_slip_speed
    )
    S_dot_r = (
        tectonic_shear_stressing_rate
        - fault_patch.mu_0 * tectonic_normal_stressing_rate
    )
    gamma_0 = 1.0 / S_dot_r
    a = fault_patch.a
    rho = 2700.0
    g = 9.8
    sigma = rho * g * abs(fault.coords[patch_index, 2])
    # --------------------------------------------------------
    S, S_dot, time_ = get_coulomb_stress(fault_patch)
    # --------------------------------------------------------
    r_background_estimate = (np.float64(timings.size) / timings[-1]) * (3600.0 * 24.0)
    from scipy.optimize import fmin

    R_theoretical = (
        lambda S_dot, gamma_minus, delta_t: r_background_estimate
        * (S_dot / S_dot_r)
        / ((gamma_minus * S_dot - 1.0) * np.exp(-S_dot * delta_t / (sigma * a)) + 1.0)
    )
    Psi = (
        lambda S_dot, R_observed, gamma_minus, delta_t: (
            R_observed - R_theoretical(S_dot, gamma_minus, delta_t)
        )
        ** 2
    )
    inverted_S_dot = np.zeros(R_empirical.size - 1, dtype=np.float64)
    integrated_S = np.zeros(R_empirical.size, dtype=np.float64)
    for i in range(1, R_empirical.size):
        gamma_minus = r_background_estimate / (
            S_dot_r * max(r_background_estimate, R_empirical[i - 1])
        )
        first_guess_S_dot = S_dot_r
        R_first_guess = R_theoretical(
            first_guess_S_dot, gamma_minus, time[i] - time[i - 1]
        )
        while R_first_guess < R_empirical[i]:
            first_guess_S_dot *= 10.0
            R_first_guess = R_theoretical(
                first_guess_S_dot, gamma_minus, time[i] - time[i - 1]
            )
        # r_background_estimate = R_empirical[:i].mean()
        opt_output = fmin(
            Psi,
            np.array([S_dot_r]),
            args=(
                max(R_empirical[i], r_background_estimate),
                gamma_minus,
                time[i] - time[i - 1],
            ),
            disp=False,
        )
        inverted_S_dot[i - 1] = opt_output[0]
        # print inverted_S_dot[i], S_dot[i-1], first_guess_S_dot
        integrated_S[i] = integrated_S[i - 1] + inverted_S_dot[i - 1] * (
            time[i] - time[i - 1]
        )
    return inverted_S_dot, integrated_S, time


def invert_EQ_rate_linear_approx(fault, patch_index):
    # --------------------------------------------------------
    catalog_ = get_event_history(fault)
    R_empirical, time, cum_n_events, local_cat = empirical_earthquake_rate(
        fault, patch_index
    )
    patch_indexes = np.int32(np.unique(local_cat[:, 1]))
    timings = local_cat[:, 0]
    # --------------------------------------------------------
    fault_patch = fault.fault_patches[patch_index]
    tectonic_shear_stressing_rate = (
        fault_patch.ktau_tectonic * fault_patch.tectonic_slip_speed
    )
    tectonic_normal_stressing_rate = (
        fault_patch.ksigma_tectonic * fault_patch.tectonic_slip_speed
    )
    S_dot_r = (
        tectonic_shear_stressing_rate
        - fault_patch.mu_0 * tectonic_normal_stressing_rate
    )
    gamma_0 = 1.0 / S_dot_r
    a = fault_patch.a
    rho = 2700.0
    g = 9.8
    sigma = rho * g * abs(fault.coords[patch_index, 2])
    # --------------------------------------------------------
    S, S_dot, time_ = get_coulomb_stress(fault_patch)
    # --------------------------------------------------------
    r = (np.float64(timings.size) / timings[-1]) * (3600.0 * 24.0)
    from scipy.optimize import fmin

    inverted_S_dot_1 = np.zeros(R_empirical.size - 1, dtype=np.float64)
    inverted_S_dot_2 = np.zeros(R_empirical.size - 1, dtype=np.float64)
    for i in range(1, R_empirical.size):
        gamma_minus = r / (S_dot_r * max(r, R_empirical[i - 1]))
        r_background_estimate = R_empirical[:i].mean()
        delta_t = time[i] - time[i - 1]
        # ---------------------------------------------
        R_reduced = max(1, R_empirical[i - 1] / r_background_estimate)
        A = -gamma_minus * delta_t * R_reduced / (sigma * a)
        B = gamma_minus * R_reduced + delta_t * R_reduced / (sigma * a) - 1.0 / S_dot_r
        S_dot_1 = (-B + np.abs(B)) / (2.0 * A)
        S_dot_2 = (-B - np.abs(B)) / (2.0 * A)
        # ---------------------------------------------
        inverted_S_dot_1[i - 1] = S_dot_1
        inverted_S_dot_2[i - 1] = S_dot_2
    return inverted_S_dot_1, inverted_S_dot_2, time


# ==============================================================================================================
#                    CHARACTERISTIC EARTHQUAKE CYCLE SINGLE PATCHES
# ==============================================================================================================


def characteristic_EQ_cycle(fault, simulation_time=0.0):
    import okada

    faults_single_patch = []
    catalogs = []
    for i, fp in enumerate(fault.fault_patches):
        normal_faulting = False
        if -fp.t_v[0] * fp.t_h[0] < 0.0:
            right_lateral_faulting = False
        else:
            right_lateral_faulting = True
        fault_patch = okada.rate_and_state.rate_and_state_fault_patch(
            fp.mu_0,
            fp.a,
            fp.b,
            fp.Dc,
            fp.d_dot_star,
            fp.theta_star,
            fp.normal_stress_history[0],
            fp.d_dot_EQ,
            fp.beta,
            fp.G,
            fp.lbd,
            fp.alpha,
            fp.L,
            fp.W,
            fp.strike_angle,
            fp.dip_angle,
            fault.coords[i, :],
            right_lateral_faulting,
            normal_faulting,
            overshoot=1.0,
            record_history=True,
            tectonic_slip_speed=fp.tectonic_slip_speed,
        )
        fault_patch.initial_state(
            0.0, 10.0 * fp.Dc / fp.d_dot_star, fp.normal_stress_history[0]
        )
        faults_single_patch.append(
            okada.rate_and_state.rate_and_state_fault(
                fault.coords[i, :].reshape(1, -1),
                [fault_patch],
                verbose=False,
                ktau_tectonic=fault.fault_patches[i].ktau_tectonic,
            )
        )
        faults_single_patch[-1].locked_patches = np.array([fault.locked_patches[i]])
        while faults_single_patch[-1].fault_patches[0].time[-1] < max(
            fault.fault_patches[0].time[-1], simulation_time
        ):
            faults_single_patch[-1].evolve_next_patch()
        catalog_ = okada.okada_library.get_event_history(faults_single_patch[-1])
        catalogs.append(
            okada.okada_library.group_events(
                catalog_, 1.1, 1.1 * faults_single_patch[-1].fault_patches[0].L
            )
        )
    return faults_single_patch, catalogs


def plot_single_patches_characteristic_interevent_times(
    faults_single_patch, catalogs, RT_cutoff=0.001
):
    from matplotlib.ticker import FormatStrFormatter

    year = 3600.0 * 24.0 * 365.0
    charact_times = np.zeros(len(faults_single_patch), dtype=np.float64)
    for p in range(len(faults_single_patch)):
        catalog = catalogs[p]
        OT = np.float64([catalog[i][0][0] for i in range(len(catalog))])
        RT = OT[1:] - OT[:-1]
        n, bins = np.histogram((RT[RT > RT_cutoff]), bins=100)
        charact_times[p] = bins[n.argmax()]
    plt.figure(
        "recurrence_times_{:d}_{:.0f}m_{:.0f}m_patches".format(
            len(faults_single_patch),
            faults_single_patch[0].fault_patches[0].L,
            faults_single_patch[0].fault_patches[0].W,
        ),
        figsize=(12, 8),
    )
    plt.plot(charact_times / year, marker="o", ls="")
    plt.xlabel("Patch Index")
    plt.ylabel("Recurrence Time (years)")
    plt.grid()
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2e"))
    plt.show()
    return charact_times
