import numpy as np

dtr = np.pi / 180.0


def make_network(
    n_patches_per_line,
    connecting_points,
    strike_angles_per_line,
    Ls,
    Ws,
    depth,
    offset=0.0,
):
    n_patches = np.sum(n_patches_per_line)
    n_lines = len(n_patches_per_line)
    position_fault_patches = np.zeros((n_patches, 3), dtype=np.float64)
    strike_angles = np.zeros(n_patches, dtype=np.float64)
    line_indexes = np.zeros(n_patches, dtype=np.int32)
    # --------------------------------------------------------------------
    neighbors = [[] for i in range(n_patches)]
    patch_counter = 0
    first_end = np.array([-Ls[0] / 2.0, 0.0, -depth])
    for i in range(n_lines):
        lbd = strike_angles_per_line[i] * dtr
        for n in range(n_patches_per_line[i]):
            neighboring_patches = []
            unwrapped_index = np.sum(n_patches_per_line[:i]) + n
            if (n == 0) and (i == 0):
                second_end = first_end + Ls[unwrapped_index] * np.array(
                    [np.sign(np.sin(lbd)), 0.0, 0.0]
                )
                position_fault_patches[0, :] = np.mean([first_end, second_end], axis=0)
                first_end = second_end
            elif n == 0:
                lbd2 = strike_angles_per_line[line_indexes[connecting_points[i]]] * dtr
                position_fault_patches[patch_counter, :] = (
                    position_fault_patches[connecting_points[i], :]
                    + Ls[connecting_points[i]]
                    / 2.0
                    * np.array([np.sin(lbd2), np.cos(lbd2), 0.0])
                    + (0.5 + offset)
                    * Ls[unwrapped_index]
                    * np.array([np.sin(lbd), np.cos(lbd), 0.0])
                )
                first_end = position_fault_patches[patch_counter, :] + Ls[
                    unwrapped_index
                ] / 2.0 * np.array([np.sin(lbd), np.cos(lbd), 0.0])
                neighbors[patch_counter].append(connecting_points[i])
                neighbors[connecting_points[i]].append(patch_counter)
                # position_fault_patches[patch_counter, :] = position_fault_patches[connecting_points[i],:] + Ls[unwrapped_index] * np.array([np.sin(lbd), np.cos(lbd), 0.])
            else:
                # position_fault_patches[patch_counter, :] = position_fault_patches[patch_counter-1, :] + (Ls[unwrapped_index-1]/2. + Ls[unwrapped_index]/2.) * np.array([np.sin(lbd), np.cos(lbd), 0.])
                second_end = first_end + Ls[unwrapped_index] * np.array(
                    [np.sin(lbd), np.cos(lbd), 0.0]
                )
                position_fault_patches[patch_counter, :] = np.mean(
                    [first_end, second_end], axis=0
                )
                first_end = second_end
                neighbors[patch_counter].append(patch_counter - 1)
                neighbors[patch_counter - 1].append(patch_counter)
            # strike angles between 0 - 360 are useful for tracing the network,
            # but hey have to be between 0 - 180 for the stress computation
            # strike_angles[patch_counter] = strike_angles_per_line[i]%180.
            strike_angles[patch_counter] = strike_angles_per_line[i]
            line_indexes[patch_counter] = i
            patch_counter += 1
    return position_fault_patches, strike_angles, line_indexes, neighbors


def make_network_2D(
    n_patches_per_line,
    connecting_points,
    strike_angles_per_line,
    dip_angles_per_line,
    Ls,
    Ws,
    depth,
    n_depth_layers,
    offset=0.0,
):
    n_patches_one_layer = np.sum(n_patches_per_line)
    n_patches = np.sum(n_depth_layers * n_patches_per_line)
    n_lines = len(n_patches_per_line)
    position_fault_patches = np.zeros((n_patches, 3), dtype=np.float64)
    strike_angles = np.zeros(n_patches, dtype=np.float64)
    dip_angles = np.zeros(n_patches, dtype=np.float64)
    line_indexes = np.zeros(n_patches, dtype=np.int32)
    # --------------------------------------------------------------------
    neighbors = [[] for i in range(n_patches)]
    patch_counter = 0
    for d in range(n_depth_layers):
        for i in range(n_lines):
            lbd = strike_angles_per_line[i] * dtr
            delta = dip_angles_per_line[i] * dtr
            for n in range(n_patches_per_line[i]):
                neighboring_patches = []
                t_v = np.array(
                    [
                        -np.cos(lbd) * np.cos(delta),
                        np.sin(lbd) * np.cos(delta),
                        np.sin(delta),
                    ]
                )  # cf. t_v used in the patch's coordinate system
                if t_v[0] < 0.0:
                    # in my construction, the faults dip toward the north
                    t_v[:-1] *= -1.0
                # t_v = np.array([np.abs(-np.cos(lbd)*np.cos(delta)), 0., np.sin(delta)]) # cf. t_v used in the patch's coordinate system
                # t_v /= np.linalg.norm(t_v)
                if (n == 0) and (i == 0):
                    first_end = (
                        np.array(
                            [
                                -Ls[patch_counter % n_patches_one_layer] / 2.0,
                                0.0,
                                -depth,
                            ]
                        )
                        - d * Ws[patch_counter % n_patches_one_layer] * t_v
                    )
                    second_end = first_end + Ls[
                        patch_counter % n_patches_one_layer
                    ] * np.array([np.sin(lbd), np.cos(lbd), 0.0])
                    position_fault_patches[patch_counter, :] = np.mean(
                        [first_end, second_end], axis=0
                    )
                    first_end = second_end
                elif n == 0:
                    lbd2 = (
                        strike_angles_per_line[line_indexes[connecting_points[i]]] * dtr
                    )
                    delta2 = (
                        dip_angles_per_line[line_indexes[connecting_points[i]]] * dtr
                    )
                    if lbd - lbd2 < 0.0:
                        offset_ = (
                            Ls[patch_counter % n_patches_one_layer]
                            / 2.0
                            * np.array([np.sin(lbd), np.cos(lbd), 0.0])
                            - Ws[patch_counter % n_patches_one_layer]
                            / 2.0
                            * np.array([t_v[0], t_v[1], 0.0])
                            - np.float64(d)
                            * (
                                Ls[patch_counter % n_patches_one_layer]
                                + Ws[patch_counter % n_patches_one_layer]
                                * np.abs(
                                    np.dot(
                                        t_v, np.array([np.sin(lbd2), np.cos(lbd2), 0.0])
                                    )
                                )
                            )
                            * np.array([np.sin(lbd), np.cos(lbd), 0.0])
                        )
                    if lbd - lbd2 > 0.0:
                        offset_ = (
                            Ls[patch_counter % n_patches_one_layer]
                            / 2.0
                            * np.array([np.sin(lbd), np.cos(lbd), 0.0])
                            + Ws[patch_counter % n_patches_one_layer]
                            / 2.0
                            * np.array([t_v[0], t_v[1], 0.0])
                            + np.float64(d)
                            * (
                                Ls[patch_counter % n_patches_one_layer]
                                + Ws[patch_counter % n_patches_one_layer]
                                * np.abs(
                                    np.dot(
                                        t_v, np.array([np.sin(lbd2), np.cos(lbd2), 0.0])
                                    )
                                )
                            )
                            * np.array([np.sin(lbd), np.cos(lbd), 0.0])
                        )
                    if i == 6:
                        # manual correction for the most complex case ...
                        offset_ = Ls[
                            patch_counter % n_patches_one_layer
                        ] / 2.0 * np.array([np.sin(lbd), np.cos(lbd), 0.0]) - (
                            Ws[patch_counter % n_patches_one_layer]
                            / 2.0
                            * np.array([t_v[0], t_v[1], 0.0])
                            + np.float64(d)
                            * (
                                Ls[patch_counter % n_patches_one_layer]
                                + Ws[patch_counter % n_patches_one_layer]
                                * np.abs(
                                    np.dot(
                                        t_v, np.array([np.sin(lbd2), np.cos(lbd2), 0.0])
                                    )
                                )
                            )
                            * np.array([np.sin(lbd), np.cos(lbd), 0.0])
                        )
                    a = np.sign(np.cos(lbd2))
                    position_fault_patches[patch_counter, :] = (
                        position_fault_patches[connecting_points[i], :]
                        + Ls[connecting_points[i]]
                        / 2.0
                        * np.array([np.sin(lbd2), np.cos(lbd2), 0.0])
                        + Ws[connecting_points[i]]
                        / 2.0
                        * a
                        * np.sign(np.cos(lbd2) * np.cos(delta2))
                        * np.array(
                            [
                                -np.cos(lbd2) * np.cos(delta2),
                                np.sin(lbd2) * np.cos(delta2),
                                0.0,
                            ]
                        )
                        + offset_
                        - d * Ws[connecting_points[i]] * t_v
                    )
                    first_end = position_fault_patches[patch_counter, :] + Ls[
                        patch_counter % n_patches_one_layer
                    ] / 2.0 * np.array([np.sin(lbd), np.cos(lbd), 0.0])
                    neighbors[patch_counter].append(connecting_points[i])
                    neighbors[connecting_points[i]].append(patch_counter)
                    # position_fault_patches[patch_counter, :] = position_fault_patches[connecting_points[i],:] + Ls[unwrapped_index] * np.array([np.sin(lbd), np.cos(lbd), 0.])
                else:
                    # position_fault_patches[patch_counter, :] = position_fault_patches[patch_counter-1, :] + (Ls[unwrapped_index-1]/2. + Ls[unwrapped_index]/2.) * np.array([np.sin(lbd), np.cos(lbd), 0.])
                    second_end = first_end + Ls[
                        patch_counter % n_patches_one_layer
                    ] * np.array([np.sin(lbd), np.cos(lbd), 0.0])
                    position_fault_patches[patch_counter, :] = np.mean(
                        [first_end, second_end], axis=0
                    )
                    first_end = second_end
                    neighbors[patch_counter].append(patch_counter - 1)
                    neighbors[patch_counter - 1].append(patch_counter)
                # strike angles between 0 - 360 are useful for tracing the network,
                # but hey have to be between 0 - 180 for the stress computation
                # strike_angles[patch_counter] = strike_angles_per_line[i]%180.
                strike_angles[patch_counter] = strike_angles_per_line[i]
                dip_angles[patch_counter] = dip_angles_per_line[i]
                line_indexes[patch_counter] = i
                if (patch_counter - n_patches_one_layer) >= 0:
                    neighbors[patch_counter].append(patch_counter - n_patches_one_layer)
                if (patch_counter + n_patches_one_layer) < n_patches:
                    neighbors[patch_counter].append(patch_counter + n_patches_one_layer)
                patch_counter += 1
    return position_fault_patches, strike_angles, dip_angles, line_indexes, neighbors


def make_network_stair_approximation(
    n_patches_per_line,
    connecting_points,
    strike_angles_per_line,
    Ls,
    Ws,
    depth,
    offset=0.1,
):
    n_patches = np.sum(n_patches_per_line)
    n_lines = len(n_patches_per_line)
    position_fault_patches = np.zeros((n_patches, 3), dtype=np.float64)
    strike_angles = np.zeros(n_patches, dtype=np.float64)
    line_indexes = np.zeros(n_patches, dtype=np.int32)
    # --------------------------------------------------------------------
    neighbors = [[] for i in range(n_patches)]
    patch_counter = 0
    first_end = np.array([-Ls[0] / 2.0, 0.0, -depth])
    for i in range(n_lines):
        lbd = strike_angles_per_line[i] * dtr
        slope = abs(int(10.0 * np.tan(lbd - np.pi / 2.0)))
        for n in range(n_patches_per_line[i]):
            neighboring_patches = []
            unwrapped_index = np.sum(n_patches_per_line[:i]) + n
            if (n == 0) and (i == 0):
                second_end = first_end + Ls[unwrapped_index] * np.array(
                    [np.sign(np.sin(lbd)), 0.0, 0.0]
                )
                position_fault_patches[0, :] = np.mean([first_end, second_end], axis=0)
                first_end = second_end
                strike_angles[patch_counter] = 90.0
            elif n == 0:
                lbd2 = strike_angles[connecting_points[i]] * dtr
                position_fault_patches[patch_counter, :] = (
                    position_fault_patches[connecting_points[i], :]
                    + Ls[connecting_points[i]]
                    / 2.0
                    * np.array([np.sin(lbd2), np.cos(lbd2), 0.0])
                    + (0.5 + offset)
                    * Ls[unwrapped_index]
                    * np.array([0.0, np.sign(np.cos(lbd)), 0.0])
                )
                first_end = position_fault_patches[patch_counter, :] + Ls[
                    unwrapped_index
                ] / 2.0 * np.array([0.0, np.sign(np.cos(lbd)), 0.0])
                strike_angles[patch_counter] = 90.0 - np.sign(np.cos(lbd)) * 90.0
                neighbors[patch_counter].append(connecting_points[i])
                neighbors[connecting_points[i]].append(patch_counter)
            elif slope != 0 and n % slope == 0:
                second_end = first_end + Ls[unwrapped_index] * np.array(
                    [0.0, np.sign(np.cos(lbd)), 0.0]
                )
                position_fault_patches[patch_counter, :] = np.mean(
                    [first_end, second_end], axis=0
                )
                strike_angles[patch_counter] = 90.0 - np.sign(np.cos(lbd)) * 90.0
                first_end = second_end
                neighbors[patch_counter].append(patch_counter - 1)
                neighbors[patch_counter - 1].append(patch_counter)
            else:
                second_end = first_end + Ls[unwrapped_index] * np.array(
                    [np.sign(np.sin(lbd)), 0.0, 0.0]
                )
                position_fault_patches[patch_counter, :] = np.mean(
                    [first_end, second_end], axis=0
                )
                if np.sin(lbd) > 0.0:
                    strike_angles[patch_counter] = 90.0
                else:
                    strike_angles[patch_counter] = 270.0
                first_end = second_end
                neighbors[patch_counter].append(patch_counter - 1)
                neighbors[patch_counter - 1].append(patch_counter)
            # strike angles between 0 - 360 are useful for tracing the network,
            # but hey have to be between 0 - 180 for the stress computation
            # strike_angles[patch_counter] = strike_angles_per_line[i]%180.
            line_indexes[patch_counter] = i
            patch_counter += 1
    return position_fault_patches, strike_angles, line_indexes, neighbors


def complexity0(n_patches, L, W, depth, offset=0.1, n_depth_layers=1):
    n_patches_per_line = np.int32([n_patches])
    strike_angles_per_line = np.float64([90.0])
    connecting_points = np.int32([])
    Ls = L * np.ones(n_patches, dtype=np.float64)
    Ws = W * np.ones(n_patches, dtype=np.float64)
    # if n_depth_layers == 1:
    #    return make_network(n_patches_per_line, connecting_points, strike_angles_per_line, Ls, Ws, depth, offset=offset)
    # else:
    dip_angles_per_line = np.float64([90.0])
    return make_network_2D(
        n_patches_per_line,
        connecting_points,
        strike_angles_per_line,
        dip_angles_per_line,
        Ls,
        Ws,
        depth,
        n_depth_layers,
        offset=offset,
    )


def complexity1(
    factor_patches, L, W, depth, stair_approximation=False, offset=0.1, n_depth_layers=1
):
    n_patches_per_line = np.int32(factor_patches * np.int32([6, 2, 2]))
    n_patches = np.sum(n_patches_per_line)
    strike_angles_per_line = np.float64([90.0, 240.0, 60.0])
    Ls_extended = L * np.ones(n_patches, dtype=np.float64)
    Ws_extended = W * np.ones(n_patches, dtype=np.float64)
    connecting_points = np.int32(factor_patches * (1 + np.int32([0, 1, 3])) - 1)
    if stair_approximation:
        (
            position_fault_patches,
            strike_angles,
            line_indexes,
            neighbors,
        ) = make_network_stair_approximation(
            n_patches_per_line,
            connecting_points,
            strike_angles_per_line,
            Ls_extended,
            Ws_extended,
            depth,
            offset=offset,
        )
    else:
        # if n_depth_layers == 1:
        #    position_fault_patches, strike_angles, line_indexes, neighbors = \
        #            make_network(n_patches_per_line, connecting_points, strike_angles_per_line, Ls_extended, Ws_extended, depth, offset=offset)
        #    dip_angles = None
        # else:
        dip_angles_per_line = np.float64([90.0, 30.0, 30.0])
        (
            position_fault_patches,
            strike_angles,
            dip_angles,
            line_indexes,
            neighbors,
        ) = make_network_2D(
            n_patches_per_line,
            connecting_points,
            strike_angles_per_line,
            dip_angles_per_line,
            Ls_extended,
            Ws_extended,
            depth,
            offset=offset,
            n_depth_layers=n_depth_layers,
        )
    return (
        position_fault_patches,
        strike_angles,
        dip_angles,
        line_indexes,
        neighbors,
        n_depth_layers * n_patches_per_line,
    )


def complexity2(
    factor_patches, L, W, depth, stair_approximation=False, offset=0.1, n_depth_layers=1
):
    n_patches_per_line = np.int32(factor_patches * np.int32([10, 2, 2, 2, 2]))
    n_patches = np.sum(n_patches_per_line)
    strike_angles_per_line = np.float64([90.0, 240.0, 60.0, 270.0, 90.0])
    Ls_extended = L * np.ones(n_patches, dtype=np.float64)
    Ws_extended = W * np.ones(n_patches, dtype=np.float64)
    # Ls_extended            = np.zeros(n_patches, dtype=np.float64)
    # Ws_extended            = np.zeros(n_patches, dtype=np.float64)
    # for i in range(n_patches_per_line.size):
    #    i0 = np.sum(n_patches_per_line[:i])
    #    Ls_extended[i0:i0+n_patches_per_line[i]] = Ls_per_line[i]
    #    Ws_extended[i0:i0+n_patches_per_line[i]] = Ws_per_line[i]
    connecting_points = np.int32(factor_patches * (1 + np.int32([0, 3, 5, 11, 13])) - 1)
    if stair_approximation:
        (
            position_fault_patches,
            strike_angles,
            line_indexes,
            neighbors,
        ) = make_network_stair_approximation(
            n_patches_per_line,
            connecting_points,
            strike_angles_per_line,
            Ls_extended,
            Ws_extended,
            depth,
            offset=offset,
        )
    else:
        # if n_depth_layers == 1:
        #    position_fault_patches, strike_angles, line_indexes, neighbors = \
        #         make_network(n_patches_per_line, connecting_points, strike_angles_per_line, Ls_extended, Ws_extended, depth, offset=offset)
        #    dip_angles = None
        # else:
        dip_angles_per_line = np.float64([90.0, 30.0, 30.0, 90.0, 90.0])
        (
            position_fault_patches,
            strike_angles,
            dip_angles,
            line_indexes,
            neighbors,
        ) = make_network_2D(
            n_patches_per_line,
            connecting_points,
            strike_angles_per_line,
            dip_angles_per_line,
            Ls_extended,
            Ws_extended,
            depth,
            offset=offset,
            n_depth_layers=n_depth_layers,
        )
    return (
        position_fault_patches,
        strike_angles,
        dip_angles,
        line_indexes,
        neighbors,
        n_depth_layers * n_patches_per_line,
    )


def complexity3(
    factor_patches, L, W, depth, stair_approximation=False, offset=0.1, n_depth_layers=1
):
    n_patches_per_line = np.int32(factor_patches * np.int32([16, 2, 2, 2, 2, 2, 2]))
    n_patches = np.sum(n_patches_per_line)
    strike_angles_per_line = np.float64([90.0, 240.0, 60.0, 270.0, 90.0, 300.0, 120.0])
    Ls_extended = L * np.ones(n_patches, dtype=np.float64)
    Ws_extended = W * np.ones(n_patches, dtype=np.float64)
    # Ls_extended            = np.zeros(n_patches, dtype=np.float64)
    # Ws_extended            = np.zeros(n_patches, dtype=np.float64)
    # for i in range(n_patches_per_line.size):
    #    i0 = np.sum(n_patches_per_line[:i])
    #    Ls_extended[i0:i0+n_patches_per_line[i]] = Ls_per_line[i]
    #    Ws_extended[i0:i0+n_patches_per_line[i]] = Ws_per_line[i]
    connecting_points = np.int32(
        factor_patches * (1 + np.int32([0, 5, 9, 17, 19, 21, 23])) - 1
    )
    if stair_approximation:
        (
            position_fault_patches,
            strike_angles,
            line_indexes,
            neighbors,
        ) = make_network_stair_approximation(
            n_patches_per_line,
            connecting_points,
            strike_angles_per_line,
            Ls_extended,
            Ws_extended,
            depth,
            offset=offset,
        )
    else:
        # if n_depth_layers == 1:
        #    dip_angles = None
        #    position_fault_patches, strike_angles, line_indexes, neighbors = \
        #            make_network(n_patches_per_line, connecting_points, strike_angles_per_line, Ls_extended, Ws_extended, depth, offset=offset)
        # else:
        dip_angles_per_line = np.float64([90.0, 30.0, 30.0, 90.0, 90.0, 30.0, 30.0])
        (
            position_fault_patches,
            strike_angles,
            dip_angles,
            line_indexes,
            neighbors,
        ) = make_network_2D(
            n_patches_per_line,
            connecting_points,
            strike_angles_per_line,
            dip_angles_per_line,
            Ls_extended,
            Ws_extended,
            depth,
            offset=offset,
            n_depth_layers=n_depth_layers,
        )
    return (
        position_fault_patches,
        strike_angles,
        dip_angles,
        line_indexes,
        neighbors,
        n_depth_layers * n_patches_per_line,
    )


def quasi_planar(
    factor_patches,
    L,
    W,
    depth,
    n_segments=6,
    delta_strike_angle=15.0,
    n_depth_layers=1,
    offset=0.0,
):
    n_patches_per_line = np.int32(factor_patches * np.ones(n_segments, dtype=np.int32))
    n_patches = np.sum(n_patches_per_line)
    strike_angles_per_line = np.zeros(n_segments, dtype=np.float64)
    for i in range(n_segments):
        if i % 2 == 0:
            strike_angles_per_line[i] = 90.0 + delta_strike_angle
        else:
            strike_angles_per_line[i] = 90.0 - delta_strike_angle
    connecting_points = np.hstack(
        (
            [0],
            (factor_patches - 1)
            + np.int32([i * factor_patches for i in range(n_segments - 1)]),
        )
    )
    print(connecting_points)
    Ls_extended = L * np.ones(n_patches, dtype=np.float64)
    Ws_extended = W * np.ones(n_patches, dtype=np.float64)
    dip_angles_per_line = 90.0 * np.ones(n_segments, dtype=np.float64)
    (
        position_fault_patches,
        strike_angles,
        dip_angles,
        line_indexes,
        neighbors,
    ) = make_network_2D(
        n_patches_per_line,
        connecting_points,
        strike_angles_per_line,
        dip_angles_per_line,
        Ls_extended,
        Ws_extended,
        depth,
        offset=offset,
        n_depth_layers=n_depth_layers,
    )
    return (
        position_fault_patches,
        strike_angles,
        dip_angles,
        line_indexes,
        neighbors,
        n_depth_layers * n_patches_per_line,
    )
