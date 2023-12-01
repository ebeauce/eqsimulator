import numpy as np
import sys
import concurrent.futures
import h5py as h5
from scipy.optimize import fmin, leastsq
from okada_wrapper import dc3dwrapper
from time import time as give_time

DELTA_T = 1.0  # time taken before instability to compute the friction
DECIMAL_PRECISION = 6
N_POINTS_TIME_SERIES = 2
DTR = np.float64(np.pi / 180.0)
rtd = np.float64(180.0 / np.pi)


class RateStateFaultPatch(object):
    def __init__(
        self,
        mu_0=None,
        a=None,
        b=None,
        Dc=None,
        d_dot_star=None,
        theta_star=None,
        sigma_0=None,
        d_dot_EQ=None,
        beta=None,
        lame2=None,
        lame1=None,
        alpha=None,
        L=None,
        W=None,
        strike_angle=None,
        dip_angle=None,
        x=None,
        y=None,
        z=None,
        right_lateral_faulting=None,
        normal_faulting=None,
        overshoot=0.9,
        record_history=True,
        tectonic_slip_speed=None,
        tectonic_stressing_rate=None,
        path=None,
        gid=None,
        fault_patch_id=None
    ):
        """
        A class representing a rate-state fault patch simulation.

        Parameters
        ----------
        mu_0 : float
            Initial friction coefficient.
        a : float
            Empirical coefficient a.
        b : float
            Empirical coefficient b.
        Dc : float
            Characteristic weakening distance in [m].
        d_dot_star : float
            Reference slip speed in [m/s].
        theta_star : float
            Reference state value in [s].
        sigma_0 : float
            Initial stress value in [Pa].
        d_dot_EQ : float
            Earthquake slip speed in [m].
        beta : float
            S-wave velocity in [m/s].
        lame2 : float
            Second Lame parameter, shear modulus, in [Pa].
        lame1 : float
            First Lame parameter in [Pa].
        alpha : float
            Normal stress coefficient.
        L : float
            Length of the fault patch in [m].
        W : float
            Width of the fault patch in [m].
        strike_angle : float
            Angle of fault strike in [deg].
        dip_angle : float
            Angle of fault dip in [deg].
        x : float
            x-coordinate in [m].
        y : float
            y-coordinate in [m].
        z : float
            z-coordinate in [m].
        right_lateral_faulting : bool
            Flag for right-lateral faulting.
        normal_faulting : bool
            Flag for normal faulting.
        overshoot : float, optional
            Overshoot value. Default is 0.9.
        record_history : bool, optional
            Flag to record simulation history. Default is True.
        tectonic_slip_speed : float or None, optional
            Tectonic slip speed in [m/s]. Default is None.
        tectonic_stressing_rate : float or None, optional
            Tectonic stressing rate in [Pa/s]. Default is None.

        Notes
        -----
        This class initializes various attributes and parameters for a rate-state fault patch simulation.
        It calculates internal values based on input parameters, and handles the geometry and behavior of the fault patch.
        Use this class to perform simulations and study fault behavior under specified conditions.

        """
        # ---------------------------------------------
        #                 Properties
        # ---------------------------------------------
        #       store variable names
        self._property_variables = [
            "a",
            "b",
            "Dc",
            "d_dot_star",
            "theta_star",
            "mu_0",
            "alpha",
            "x",
            "y",
            "z",
            "fault_patch_id",
            "tectonic_slip_speed",
            "tectonic_stressing_rate",
            "overshoot",
            "beta",
            "lame1",
            "lame2",
            "d_dot_EQ",
            "L",
            "W",
            "dip_angle",
            "strike_angle",
            "right_lateral_faulting",
            "normal_faulting"
        ]
        if path:
            # use parameters from file and ignore values given here
            with h5.File(path, mode="r") as fprop:
                if gid is not None:
                    fprop = fprop[gid]
                for var in self._property_variables:
                    setattr(self, var, fprop[var][()])
            self.a_nominal = np.float64(self.a)
        else:
            self.a = np.float64(a)
            self.a_nominal = np.float64(a)
            self.b = np.float64(b)
            self.Dc = np.float64(Dc)
            self.d_dot_star = np.float64(d_dot_star)
            self.theta_star = np.float64(theta_star)
            self.mu_0 = np.float64(mu_0)
            self.alpha = alpha  # normal stress coefficient
            self.x = x
            self.y = y
            self.z = z
            self.fault_patch_id = None
            if tectonic_slip_speed is None:
                self.tectonic_slip_speed = 0.0
            else:
                self.tectonic_slip_speed = np.float64(tectonic_slip_speed)
            if tectonic_stressing_rate is None:
                self.tectonic_stressing_rate = 0.0
            else:
                self.tectonic_stressing_rate = np.float64(tectonic_stressing_rate)
            self.overshoot = np.float64(overshoot)
            self.beta = np.float64(beta)
            self.lame2 = np.float64(lame2)
            self.lame1 = np.float64(lame1)
            self.d_dot_EQ = np.float64(d_dot_EQ)
            self.L = np.float64(L)
            self.W = np.float64(W)
            self.dip_angle = np.float64(dip_angle)
            self.strike_angle = np.float64(strike_angle)
            self.normal_faulting = normal_faulting
            self.right_lateral_faulting = right_lateral_faulting
        self.record_history = record_history
        # ---------------------------------------------
        #         Set property-based variables
        # ---------------------------------------------
        self._set_property_based_variables()

    def _set_property_based_variables(self):
        self.mu_0_ = np.float64(
            self.mu_0
            - self.a * np.log(self.d_dot_star)
            - self.b * np.log(self.theta_star)
        )
        self.sum_term = np.float64(self.b / self.Dc)
        self.xi = self.b - self.a
        self.area = self.L * self.W
        # -------------------------------------------------
        # column vector tangent to the fault element and
        # in the direction of positive strike slip
        self.t_h = np.round(
            np.array(
                [np.sin(self.strike_angle * DTR), np.cos(self.strike_angle * DTR), 0.0]
            ),
            decimals=DECIMAL_PRECISION,
        )
        # vector tangent to the fault element and
        # in the direction of positive dip slip (upward)
        self.t_v = np.round(
            np.array(
                [
                    -np.cos(self.strike_angle * DTR) * np.cos(self.dip_angle * DTR),
                    np.sin(self.strike_angle * DTR) * np.cos(self.dip_angle * DTR),
                    np.sin(self.dip_angle * DTR),
                ]
            ),
            decimals=DECIMAL_PRECISION,
        )
        # normal vector
        z_unit = np.array([0.0, 0.0, 1.0])
        self.n = np.cross(self.t_h, self.t_v)
        # test
        x_unit = np.array([1.0, 0.0, 0.0])
        # define the pitch (i.e. slip vector) for the FOOT WALL (consistent with the normal)
        self.p = np.zeros(3, dtype=np.float64)
        self.p[1] = 0.0  # no displacement parallel to y (north-south)
        if self.normal_faulting:
            # foot wall is slipping upward
            self.p[2] = +np.sin(self.dip_angle * DTR) * np.abs(
                np.round(np.dot(x_unit, self.t_v), decimals=1)
            )
        else:
            # foot wall is slipping downward
            self.p[2] = -np.sin(self.dip_angle * DTR) * np.abs(
                np.round(np.dot(x_unit, self.t_v), decimals=1)
            )
        if self.p[2] == 0.0:
            # right-lateral motion, slip direction given for the north wall
            self.p[0] = self.t_h[0]
        else:
            self.p[0] = -1.0 * (self.n[2] / self.n[0]) * self.p[2]
        # make sure p is a unit vector
        self.p /= np.linalg.norm(self.p)
        self.strike_slip_component = np.dot(self.p, -self.t_h)
        # Okada: positive strike-slip is left-lateral, i.e. if HANGING WALL's slip
        # (given by -pitch) is in the direction of t_h
        self.dip_slip_component = np.dot(self.p, -self.t_v)
        # Okada: positive dip-slip is reverse faulting, i.e. if HANGING WALL's slip
        # is in the direction of t_v

        # -------------------------------------------------
        # change_frame: canonical frame ---> fault frame
        self.change_frame = np.vstack(
            (self.t_h.reshape(1, -1), self.t_v.reshape(1, -1), self.n.reshape(1, -1))
        )
        # change_x_y: rotate the frame around the z axis
        self.change_x_y = np.vstack(
            (
                self.t_h.reshape(1, -1),
                np.round(
                    np.array(
                        [
                            -np.cos(self.strike_angle * DTR),
                            np.sin(self.strike_angle * DTR),
                            0.0,
                        ]
                    ),
                    decimals=DECIMAL_PRECISION,
                ).reshape(1, -1),
                z_unit.reshape(1, -1),
            )
        )

    @property
    def shear_modulus(self):
        # alias for lame2
        return self.lame2

    @property
    def G(self):
        # alias for lame2
        return self.lame2

    @property
    def time(self):
        return self.start_time + np.cumsum(np.asarray(self._time_increments))
        #return np.asarray(self._time)

    @property
    def time_increments(self):
        return np.asarray(self._time_increments)

    @property
    def shear_stress_history(self):
        return np.asarray(self._shear_stress_history)

    @property
    def normal_stress_history(self):
        return np.asarray(self._normal_stress_history)

    @property
    def state_history(self):
        return np.asarray(self._state_history)

    @property
    def transition_times_history(self):
        return np.asarray(self._transition_times_history)

    @property
    def slip_history(self):
        return np.asarray(self._slip_history)

    @property
    def slip_speed_history(self):
        return np.asarray(self._slip_speed_history)

    @property
    def theta_history(self):
        return np.asarray(self._theta_history)

    @property
    def event_stress_drops(self):
        return np.asarray(self._event_stress_drops)

    @property
    def event_timings(self):
        return np.asarray(self._event_timings)

    @property
    def event_slips(self):
        return np.asarray(self._event_slips)

    @property
    def fault_slip_history(self):
        return np.asarray(self._fault_slip_history)

    @property
    def mu_effective(self):
        return self.mu_0 - self.alpha

    def initialize_mechanical_state(
        self,
        d_dot=None,
        theta=None,
        normal_stress=None,
        state=0,
        displacement=0.0,
        friction=0.0,
        a=None,
        initial_shear_stress=None,
        start_time=0.,
        path=None,
        gid=None,
    ):
        # ------------------------------------------
        #       mechanical state variables
        # ------------------------------------------
        self._mechanical_state_variables = [
                "state",
                "normal_stress",
                "shear_stress",
                "displacement",
                "shear_stress_0",
                "theta",
                "friction",
                "a",
                "d_dot",
                "shear_stress_rate",
                "normal_stress_rate",
                "start_time",
                ]
        if path:
            # use parameters from file and ignore values given here
            with h5.File(path, mode="r") as fstate:
                if gid is not None:
                    fstate = fstate[gid]
                for var in self._mechanical_state_variables:
                    setattr(self, var, fstate[var][()])
                    print(var, getattr(self, var))
        else:
            self.state = state
            self.a = self.a_nominal
            self.displacement = displacement
            self.d_dot = np.float64(d_dot)
            self.theta = np.float64(theta)
            if initial_shear_stress is None:
                self.shear_stress = np.float64(0.8 * self.mu_0 * normal_stress)
            elif isinstance(initial_shear_stress, float):
                self.shear_stress = np.float64(
                        initial_shear_stress * self.mu_0 * normal_stress
                        )
            elif initial_shear_stress == "random":
                self.shear_stress = np.float64(
                    np.random.uniform(0.5, 0.9) * self.mu_0 * nornmal_stress
                )
            self.shear_stress_0 = float(self.shear_stress)
            self.normal_stress = np.float64(normal_stress)
            self.friction = (
                friction  # will be initialized to some non-zero value at the end of state 0
            )
            self.shear_stress_rate = None
            self.normal_stress_rate = None
            self.start_time = 0.

    def initialize_history(
            self,
            path=None,
            readall=False,
            gid=None,
            history_variables=None
            ):
        # --------------------------------------------------
        #                  history
        # --------------------------------------------------
        event_variables = [
                "_event_stress_drops",
                "_event_timings",
                "_event_slips",
                ]

        if history_variables is None:
            # by default, keep track of all variables
            self._history_variables = [
                "_state_history",
                "_transition_times_history",
                "_slip_history",
                "_slip_speed_history",
                "_theta_history",
                "_fault_slip_history",
                "_time_increments",
                "_shear_stress_history",
                "_normal_stress_history",
            ] + event_variables
        else:
            self._history_variables = history_variables
        self._non_event_history_variables = list(
                set(self._history_variables).difference(
                    event_variables
                    )
                )
        if path is None:
            # initialize from scratch
            # -------- essential history variables -----
            self._event_timings = [0.0]
            self._event_stress_drops = [0.0]
            self._event_slips = [0.0]
            # -------- optional history variables -------
            self._state_history = [self.state]
            self._transition_times_history = [0.0]
            self._slip_history = [0.0]
            self._slip_speed_history = [self.d_dot]
            self._theta_history = [self.theta]
            self._fault_slip_history = [0.0]
            self._time_increments = [0.0]
            #self._time = [0.0]
            self._shear_stress_history = [self.shear_stress]
            self._normal_stress_history = [self.normal_stress]
        else:
            with h5.File(path, mode="r") as fhist:
                if gid is not None:
                    fhist = fhist[gid]
                for var in fhist:
                    if readall:
                        setattr(self, f"_{var}", fhist[var][()])
                        print(
                                "!! Make sure you are not using readall=True to"
                                " restart a simulation from the last checkpoint !!"
                                )
                    else:
                        setattr(self, f"_{var}", [fhist[var][-1]])
                if hasattr(self, "_time_increments"):
                    self._time_increments[-1] = 0.


    def clean_history(self):
        if self.record_history:
            self.start_time = self.time[-1]
            for attr in self._history_variables:
                setattr(self, attr, [getattr(self, attr)[-1]])

    def state_0(self, t0_1=None):
        # transition happens when driving stress is above steady state friction
        # Note: steady state friction also evolves with time
        if t0_1 is None:
            t0_1 = self.find_t0_1()
            t0_1 = np.round(t0_1, decimals=DECIMAL_PRECISION)
        if self.record_history:
            # -------- UPDATE HISTORY ----------
            self.update_history(t0_1)
            # ----------------------------------
        # self.theta += t0_1
        self.theta = self.evolve_state_variable(t0_1)
        self.shear_stress += self.shear_stress_rate * t0_1
        self.normal_stress += self.normal_stress_rate * t0_1
        self.set_friction_to_steady_state()
        self.set_slip_speed_to_steady_state()
        self.d_dot_state_0 = np.float64(self.d_dot)
        self.state = 1  # state is now 1
        # print('End of state 0: Friction = {:.2e}MPa, Driving stress = {:.2e}MPa, slip speed = {:.2e}m/s'.format(self.friction/1.e6, self.shear_stress/1.e6, self.d_dot))

    def state_1(self, t1_2=None):
        if t1_2 is None:
            t1_2 = self.find_t1_2()
            t1_2 = np.round(t1_2, decimals=DECIMAL_PRECISION)
        if self.record_history:
            # -------- UPDATE HISTORY ----------
            self.update_history(t1_2)
            # ----------------------------------
        self.shear_stress += self.shear_stress_rate * t1_2
        self.normal_stress += self.normal_stress_rate * t1_2
        # print('Current stress state: {:.2e}Pa, Current stressing rate: {:.2e}Pa/s'.format(self.shear_stress, self.shear_stress_rate))
        self.d_dot = self.d_dot_EQ
        self.set_theta_to_steady_state()
        self.set_friction_to_steady_state()
        self.displacement = 0.0  # reset displacement
        self.state = 2  # state is now 2
        # log stress at beginning of earthquake
        self.shear_stress_0 = float(self.shear_stress)
        self._event_timings.append(self.time[-1])

        # print('Driving stress = {:.2e}MPa, friction = {:.2e}MPa'.format(self.shear_stress/1.e6, self.friction/1.e6))

    def state_2(self, t2_0=None, verbose=True):
        # self.d_dot_EQ  = 2. * self.beta * self.Delta_tau / self.lame2
        # self.Delta_tau = self.shear_stress - self.friction
        # sliding stops when shear_stress = friction (+ some overshooting)
        if t2_0 is None:
            t2_0 = self.find_t2_0()
            t2_0 = np.round(t2_0, decimals=DECIMAL_PRECISION)
        if self.record_history:
            # -------- UPDATE HISTORY ----------
            self.update_history(t2_0)
            # ----------------------------------
        self.displacement += t2_0 * self.d_dot_EQ
        self.shear_stress += self.shear_stress_rate * t2_0
        self.normal_stress += self.normal_stress_rate * t2_0
        # stress drop is stress difference between beginning and end
        stress_drop = self.shear_stress_0 - self.shear_stress
        if verbose:
            print(
                "Patch {:d}: Displacement EQ = {:.2e}m, Theta = {:.2e}s, Stress-drop = {:.2e}Pa".format(
                    self.fault_patch_id, self.displacement, self.theta, stress_drop
                )
            )
        self.state = 0  # state is now 0
        self._event_slips.append(self.displacement)
        self._event_stress_drops.append(stress_drop)
        self.displacement = 0.0  # reset displacement
        self.d_dot = self.tectonic_slip_speed
        self.set_theta_to_steady_state()

    def state_3(self, fault_object, patch_index, t3_1=None):
        if t3_1 is None:
            t3_1 = self.find_t3_1()
            t3_1 = np.round(t3_1, decimals=DECIMAL_PRECISION)
        if self.record_history:
            # -------- UPDATE HISTORY ----------
            self.update_history(t3_1)
            # ----------------------------------
        d_dot_end_step = self.slip_speed_creeping_patch(t3_1)
        self.shear_stress += self.shear_stress_rate * t3_1
        self.normal_stress += self.normal_stress_rate * t3_1
        self.d_dot = d_dot_end_step
        self.set_theta_to_steady_state()
        self.state = 1

    def update_history(self, duration):
        total_time = np.linspace(
            self.time[-1], self.time[-1] + duration, N_POINTS_TIME_SERIES
        )
        delta_time = total_time - total_time[0]
        displacement = delta_time * self.d_dot
        #self._time.extend(total_time.tolist())
        self._time_increments.extend(delta_time.tolist())
        self._shear_stress_history.extend(
            list(self._shear_stress_history[-1] + self.shear_stress_rate * delta_time)
        )
        self._normal_stress_history.extend(
            list(self._normal_stress_history[-1] + self.normal_stress_rate * delta_time)
        )
        if self.state == 2:
            self._fault_slip_history.extend(
                list(self._fault_slip_history[-1] + delta_time * self.d_dot_EQ)
            )
        elif self.state == 3:
            self._fault_slip_history.extend(
                list(self._fault_slip_history[-1] + delta_time * self.d_dot)
            )
        else:
            self._fault_slip_history.extend(
                list(self._fault_slip_history[-1] + delta_time * 0.0)
            )

    def evolve_current_state(self, duration, fault_object, patch_index):
        if duration == 0.0:
            pass
        else:
            duration = np.round(duration, decimals=DECIMAL_PRECISION)
            self.shear_stress += self.shear_stress_rate * duration
            self.normal_stress += self.normal_stress_rate * duration
            if self.state == 0:
                # self.theta          += duration
                self.theta = self.evolve_state_variable(duration)
            elif self.state == 1:
                self.evolve_slip_speed_time_change(duration)
                # check what is the steady state slip speed
                self.set_theta_to_steady_state()
                # if self.d_dot < self.d_dot_state_0:
                if self.d_dot < 0.9 * self.get_steady_state_slip_speed():
                    # factor 0.9 to tolerate small numerical errors when checking condition
                    # go back to state 0, since the assumption about state 1 is that
                    # slip speed >> steady state slip speed
                    # self.set_theta_to_steady_state()
                    # self.update_friction()
                    # during state 1, the quasi-static approximation still holds, so friction = driving shear stress
                    # knowing the friction and the slip speed, we get the value of theta
                    # self.theta = np.exp(1./self.b * (self.shear_stress/self.normal_stress - self.mu_0 - self.a_nominal * np.log(self.d_dot/self.d_dot_star))) * self.theta_star
                    self.theta = self.adaptive_rounding(
                        np.exp(
                            1.0
                            / self.b
                            * (
                                self.shear_stress / self.normal_stress
                                - self.mu_0
                                - self.a_nominal * np.log(self.d_dot / self.d_dot_star)
                            )
                        )
                        * self.theta_star
                    )
                    self.friction = self.shear_stress
                    print(
                        "Go back to state 0: Friction = {:.2e}MPa, Driving force = {:.2e}MPa".format(
                            self.friction / 1.0e6, self.shear_stress / 1.0e6
                        )
                    )
                    self.state = 0
            elif self.state == 2:
                self.displacement += duration * self.d_dot_EQ
                # self.set_theta_to_steady_state()
                # self.set_friction_to_steady_state() # update necessary if normal stress has changed
            elif self.state == 3:
                self.d_dot_0 = np.float64(self.d_dot)
                self.d_dot = min(
                    self.d_dot_EQ,
                    self.d_dot_star
                    * np.exp(
                        (self.shear_stress / self.normal_stress - self.mu_0)
                        / (self.a - self.b)
                    ),
                )
            self.update_H()
            if self.record_history:
                self.update_history(duration)

    def time_to_instability_stressing_rate(self):
        # the initial slip speed fixes the time to instability
        # ------------------------------------------------------
        # solution if instability is defined as slip speed = infinity
        # return self.a * self.normal_stress/self.shear_stress_rate * np.log(self.shear_stress_rate / (self.H*self.normal_stress*self.d_dot) + 1.)
        # ------------------------------------------------------
        # solution if instability is defined as slip speed = d_dot_EQ
        self.coulomb_stress_rate = (
            self.shear_stress_rate - self.mu_effective * self.normal_stress_rate
        )
        if self.coulomb_stress_rate < 0.0:
            return np.finfo(np.float64).max
        return (
            self.a
            * self.normal_stress
            / self.coulomb_stress_rate
            * (
                np.log(
                    1.0 / self.d_dot
                    + self.H * self.normal_stress / self.coulomb_stress_rate
                )
                - np.log(
                    1.0 / self.d_dot_EQ
                    + self.H * self.normal_stress / self.coulomb_stress_rate
                )
            )
        )

    def time_to_instability(self):
        # the initial slip speed fixes the time to instability
        # ------------------------------------------------------
        # solution if instability is defined as slip speed = infinity
        # return self.a / self.H * (1./self.d_dot)
        # ------------------------------------------------------
        # solution if instability is defined as slip speed = d_dot_EQ
        return self.a / self.H * (1.0 / self.d_dot - 1.0 / self.d_dot_EQ)

    def evolve_state_variable(self, time):
        if self.normal_stress_rate != 0.0:
            t_c = self.normal_stress / self.normal_stress_rate
            C = (self.alpha + self.b) / self.b
            a_ = np.float64(1.0 + time / t_c)
            if a_ == 1.0:
                # normal stressing rate is not zero, but time/t_c is still roughly zero
                return self.theta + time
            theta_t = t_c / C * (1.0 + time / t_c) + (
                np.abs(self.normal_stress)
                / np.abs(self.normal_stress + self.normal_stress_rate * time)
            ) ** (self.alpha / self.b) * (self.theta - t_c / C)
            if theta_t < 0.0:
                return self.theta_star
        else:
            theta_t = self.theta + time
        return theta_t

    def steady_state_friction(self, time):
        theta_t = self.evolve_state_variable(time)
        # friction_ss = (self.normal_stress + self.normal_stress_rate * time) * (self.mu_0 + (self.b - self.a_nominal)*np.log(theta_t/self.theta_star))
        friction_ss = self.normal_stress * (
            self.mu_0
            + (self.b - self.a_nominal) * np.log(self.d_dot_star / self.Dc)
            + self.b * np.log(theta_t)
            + self.a_nominal
            * np.log(
                1.0 / theta_t
                - self.alpha * self.normal_stress_rate / (self.b * self.normal_stress)
            )
        )
        return friction_ss

    def critical_stiffness(self):
        """
        Returns the critical stiffness below which unstable slip can occur.
        """
        return (self.b - self.a_nominal) * self.normal_stress / self.Dc

    def newton_f(self, t):
        return (
            self.shear_stress
            + self.shear_stress_rate * t
            - self.steady_state_friction(t)
        )

    def newton_log_f(self, ln_t):
        t = np.exp(ln_t)
        return (
            self.shear_stress
            + self.shear_stress_rate * t
            - self.steady_state_friction(t)
        )

    def newton_fprime(self, t):
        projected_normal_stress = self.normal_stress + self.normal_stress_rate * t
        projected_normal_stress_MPa = projected_normal_stress / 1.0e6
        sigma_0_MPa = self.normal_stress / 1.0e6
        c1 = self.alpha / self.b
        c2 = 1.0 + self.alpha / self.b
        if self.normal_stress_rate == 0.0:
            theta_prime = 1.0
        elif projected_normal_stress > 0.0:
            theta_prime = 1.0 / c2 - c2 * 1.0e-6 * abs(
                sigma_0_MPa
            ) ** c1 * projected_normal_stress_MPa ** (-c2) * (
                self.theta - self.normal_stress / (c2 * self.normal_stress_rate)
            )
        elif projected_normal_stress < 0.0:
            theta_prime = 1.0 / c2 + c2 * 1.0e-6 * abs(
                sigma_0_MPa
            ) ** c1 * projected_normal_stress_MPa ** (-c2) * (
                self.theta - self.normal_stress / (c2 * self.normal_stress_rate)
            )
        else:
            print("Unexpected", self.normal_stress, self.normal_stress_rate, t, c1)
            return 0.0
        theta_t = self.evolve_state_variable(t)
        fprime = (
            self.shear_stress_rate
            - self.normal_stress * self.b * theta_prime / theta_t
            - self.normal_stress
            * self.a
            * theta_prime
            / theta_t**2
            * 1.0
            / (theta_t - c1 * self.normal_stress_rate / self.normal_stress)
        )
        return fprime

    def find_t0_1(self):
        from scipy.optimize import newton

        if self.adaptive_rounding(self.shear_stress) > 0.98 * self.adaptive_rounding(
            self.steady_state_friction(0.0)
        ):
            # factor 0.98 to relax effects of numerical imprecision
            # could happen because of the instant drop in steady state friction
            # associated with the drop in 'a' on patches neighbors to rupturing patches
            # print(
            #        'Driving stress: {:.2e}MPa, Friction: {:.2e}MPa'.format(
            #            self.shear_stress / 1.e6, self.steady_state_friction(0.) / 1.e6
            #            )
            #        )
            return 0.0
        first_guess_time = 0.1
        first_guess_diff = (
            self.shear_stress
            + self.shear_stress_rate * first_guess_time
            - self.steady_state_friction(first_guess_time)
        )
        while first_guess_diff < 0.0:
            first_guess_time *= 10.0
            if first_guess_time > 1.0e20:
                # no solution
                return np.finfo(np.float64).max
            first_guess_diff = (
                self.shear_stress
                + self.shear_stress_rate * first_guess_time
                - self.steady_state_friction(first_guess_time)
            )
        try:
            t0_1 = newton(
                self.newton_log_f,
                np.log(first_guess_time / 10.0),  # fprime=self.newton_fprime
            )
            t0_1 = np.exp(t0_1)
            # t0_1 = newton(
            #        self.newton_f, first_guess_time / 10., fprime=self.newton_fprime
            #        )
        except RuntimeError:
            # Psi = lambda t: (
            #    (self.shear_stress + self.shear_stress_rate * t)
            #    - self.steady_state_friction(t)
            # )
            # opt_output, cov = leastsq(Psi, first_guess_time / 10.)
            # t0_1 = opt_output[0]
            Psi = lambda ln_t: (
                (self.shear_stress + self.shear_stress_rate * np.exp(ln_t))
                - self.steady_state_friction(np.exp(ln_t))
            )
            opt_output, cov = leastsq(Psi, np.log(first_guess_time / 10.0))
            t0_1 = opt_output[0]
            t0_1 = np.exp(t0_1)
        if t0_1 < 0.0:
            # print(self.shear_stress / 1.e6, self.steady_state_friction(0.) / 1.e6)
            # return np.finfo(np.float64).max
            return 0.0
        else:
            # return np.round(t0_1, decimals=DECIMAL_PRECISION)
            return t0_1

    def find_t1_2(self):
        # use the analytical solutions for time to instability,
        # depending on the initial slip speed / normal stress / stressing rate
        self.coulomb_stress_rate = (
            self.shear_stress_rate - self.mu_effective * self.normal_stress_rate
        )
        if self.coulomb_stress_rate != 0.0:
            t1_2 = self.time_to_instability_stressing_rate()
        else:
            t1_2 = self.time_to_instability()

        # if np.round(t1_2, decimals=DECIMAL_PRECISION) == 0.:
        #   print(self.shear_stress_rate / 1.e6, self.d_dot, self.d_dot_EQ)
        # return np.round(t1_2, decimals=DECIMAL_PRECISION)
        # return self.adaptive_rounding(t1_2)
        return t1_2

    def find_t2_0(self):
        if self.shear_stress_rate > 0.0:
            return np.finfo(
                np.float64
            ).max  # the fault patch is still accumulating stress
        t2_0 = (self.friction - self.shear_stress) / self.shear_stress_rate
        # mu_ss = self.mu_0 + (self.a - self.b) * np.log(self.d_dot_EQ/self.d_dot_star)
        # delta_driving_friction     = self.shear_stress - self.normal_stress * mu_ss
        # delta_driving_friction_dot = self.shear_stress_rate - self.normal_stress_rate * mu_ss
        # if delta_driving_friction_dot > 0.:
        #    return np.finfo(np.float64).max # the fault patch is still accumulating stress
        # t2_0 = - delta_driving_friction / delta_driving_friction_dot
        # for large normal stress changes, friction will also change a lot
        # can lead to steady-state friction force that are above the driving shear stress,
        # which kills the event.
        # return self.adaptive_rounding(self.overshoot * t2_0)
        return self.overshoot * t2_0
        # return np.round(self.overshoot * t2_0, decimals=DECIMAL_PRECISION)

    def find_t3_1(self):
        """
        Patches in state 3 are creeping. If the critical stiffness evolves above
        the patch stiffness, then frictional instability is possible. Since a
        creeping patch is at steady state, it ends up in state 1 when the stiffness
        condition is satisfied.
        """
        # critical time at which critical stiffness is equal to patch stiffness
        if self.normal_stress_rate == 0.0:
            return np.finfo(np.float64).max
        t_c = (
            self.Dc * self.k / self.xi - self.normal_stress
        ) / self.normal_stress_rate
        if self.xi > 0.0:
            if self.normal_stress_rate > 0.0:
                # critical stiffness is an increasing function of time
                return max(0.0, t_c)
            else:
                # critical stiffness is a decreasing function of time: the patch keeps creeping
                return np.finfo(np.float64).max
        else:
            if self.normal_stress_rate >= 0.0:
                # critical stiffness is a decreasing function of time: the patch keeps creeping
                return np.finfo(np.float64).max
            else:
                # critical stiffness is an increasing function of time
                return max(0.0, t_c)

    def find_tX_3(self):
        """
        Patches in state 0 or 1 are unstable. If the critical stiffness evolves below
        the patch stiffness, then the patch becomes stable and starts creeping. When
        the stiffness condition is satisfied, the patch ends up in state 3.
        """
        # critical time at which critical stiffness is equal to patch stiffness
        if self.normal_stress_rate == 0.0:
            return np.finfo(np.float64).max
        t_c = (
            self.Dc * self.k / self.xi - self.normal_stress
        ) / self.normal_stress_rate
        if self.xi > 0.0:
            if self.normal_stress_rate >= 0.0:
                # critical stiffness is an increasing function of time: the patch remains unstable
                return np.finfo(np.float64).max
            else:
                # critical stiffness is a decreasing function of time
                return max(0.0, t_c)
        else:
            if self.normal_stress_rate > 0.0:
                # critical stiffness is a decreasing function of time
                return max(0.0, t_c)
            else:
                # critical stiffness is an increasing function of time: the patch remains unstable
                return np.finfo(np.float64).max

    def find_t3_V_update(self, V_V0_ratio=1.25):
        """
        ` ln V / V_0 = (shear_stress_rate * time) / (normal_stress * (a - b))`
        For `V = n * V_0`, `time` is:
        ` time = (normal_stress * (a - b) / shear_stress_rate) * ln(n) `
        """
        # ---------------------------------------------------------------------------
        if self.shear_stress_rate == 0.0:
            return np.finfo(np.float64).max
        characteristic_time = (
            self.normal_stress * (self.a - self.b) / np.abs(self.shear_stress_rate)
        )
        # if characteristic_time < (24. * 3600. * 365):
        #    print("Short t3-3 time:", self.fault_patch_id, characteristic_time / (24. * 3600.), self.shear_stress_rate)
        return max(2.0e-5, np.log(V_V0_ratio) * characteristic_time)

    def find_t3_tau_update(self, num_tc=1.0):
        if self.shear_stress_rate == 0.0:
            return np.finfo(np.float64).max
        characteristic_time = (self.shear_stress / 100.0) / np.abs(
            self.shear_stress_rate
        )  # time to change shear stress of 1%
        return max(2.0e-5, num_tc * characteristic_time)

    def update_H(self):
        self.H = -self.k / self.normal_stress + self.sum_term

    def update_friction(self):
        self.friction = self.normal_stress * (
            self.mu_0
            + self.a_nominal * np.log(self.d_dot / self.d_dot_star)
            + self.b * np.log(self.theta / self.theta_star)
        )

    def get_steady_state_slip_speed(self):
        return self.adaptive_rounding(
            self.Dc
            * (
                1.0 / self.theta
                - self.alpha * self.normal_stress_rate / (self.b * self.normal_stress)
            )
        )

    def set_friction_to_steady_state(self):
        # self.friction = self.normal_stress * (self.mu_0 + (self.b - self.a_nominal)*np.log(self.theta/self.theta_star))
        self.friction = self.normal_stress * (
            self.mu_0
            + (self.b - self.a_nominal) * np.log(self.d_dot_star / self.Dc)
            + self.b * np.log(self.theta)
            + self.a_nominal
            * np.log(
                1.0 / self.theta
                - self.alpha * self.normal_stress_rate / (self.b * self.normal_stress)
            )
        )

    def set_theta_to_steady_state(self):
        self.theta = self.adaptive_rounding(
            1.0
            / (
                self.d_dot / self.Dc
                + self.alpha * self.normal_stress_rate / (self.b * self.normal_stress)
            )
        )

    def set_slip_speed_to_steady_state(self):
        self.d_dot = self.adaptive_rounding(
            self.Dc
            * (
                1.0 / self.theta
                - self.alpha * self.normal_stress_rate / (self.b * self.normal_stress)
            )
        )

    def slip_speed_creeping_patch(self, time):
        new_shear_stress = self.shear_stress + time * self.shear_stress_rate
        new_normal_stress = self.normal_stress + time * self.normal_stress_rate
        return min(
            self.d_dot_EQ,
            self.d_dot_star
            * np.exp(
                (new_shear_stress / new_normal_stress - self.mu_0)
                / (self.a_nominal - self.b)
            ),
        )

    def evolve_slip_speed_time_change(self, time):
        self.coulomb_stress_rate = (
            self.shear_stress_rate - self.mu_effective * self.normal_stress_rate
        )
        if self.coulomb_stress_rate != 0.0:
            self.d_dot = np.power(
                (
                    1.0 / self.d_dot
                    + self.H * self.normal_stress / self.coulomb_stress_rate
                )
                * np.exp(
                    -self.coulomb_stress_rate * time / (self.a * self.normal_stress)
                )
                - self.H * self.normal_stress / self.coulomb_stress_rate,
                -1.0,
            )
        else:
            self.d_dot = np.power(1.0 / self.d_dot - self.H * time / self.a, -1.0)
        # self.d_dot = max(self.d_dot, self.d_dot_star)
        self.d_dot = self.adaptive_rounding(max(self.d_dot, self.d_dot_star))

    def evolve_slip_speed_stress_change(self, sigma_1_2, friction_1_2):
        self.d_dot *= (sigma_1_2[1] / sigma_1_2[0]) ** (self.alpha / self.a) * np.exp(
            friction_1_2[1] / (self.a * sigma_1_2[1])
            - friction_1_2[0] / (self.a * sigma_1_2[0])
        )

    def evolve_theta_normal_stress_change(self, sigma_1_2):
        self.theta *= (sigma_1_2[1] / sigma_1_2[0]) ** -(self.alpha / self.b)

    def adaptive_rounding(self, X, precision=DECIMAL_PRECISION):
        if X == 0.0 or X != X:
            return X
        n_decimals = max(int(-np.log10(np.abs(X)) + precision), 1)
        return np.round(X, n_decimals)

    # I/O methods
    def save_history(self, path, var=None, gid=None, overwrite=False):
        with h5.File(path, mode="a") as fhist:
            if gid is not None:
                if gid not in fhist:
                    fhist.create_group(gid)
                fhist = fhist[gid]
            if var is None:
                var_pool = self._history_variables
            else:
                var_pool = var
            for var in var_pool:
                # ignore the trailing character
                var = var[1:]
                if overwrite and var in fhist:
                    del fhist[var]
                time_series = getattr(self, var)
                if var not in fhist:
                    # initialize data set
                    fhist.create_dataset(
                            var,
                            data=time_series,
                            compression="gzip",
                            chunks=True,
                            maxshape=(None,)
                            )
                else:
                    # append to existing data set
                    fhist[var].resize(
                            fhist[var].shape[0] + len(time_series),
                            axis=0
                            )
                    fhist[var][-len(time_series):] = time_series


    def save_properties(self, path, gid=None, overwrite=True):
        with h5.File(path, mode="a") as fprop:
            if gid is not None:
                if overwrite and gid in fprop:
                    del fprop[gid]
                if gid not in fprop:
                    fprop.create_group(gid)
                fprop = fprop[gid]
            for var in self._property_variables:
                if overwrite and var in fprop:
                    del fprop[var]
                if var == "a":
                    fprop.create_dataset(var, data=self.a_nominal)
                else:
                    fprop.create_dataset(var, data=getattr(self, var))

    def save_mechanical_state(self, path, gid=None, overwrite=True):
        with h5.File(path, mode="a") as fstate:
            if gid is not None:
                #if overwrite and gid in fstate:
                #    del fstate[gid]
                if gid not in fstate:
                    fstate.create_group(gid)
                fstate = fstate[gid]
            for var in self._mechanical_state_variables:
                #if overwrite and var in fstate:
                #    del fstate[var]
                #if var == "a":
                #    fstate.create_dataset(var, data=self.a_nominal)
                #elif var == "start_time":
                #    fstate.create_dataset(var, data=self.time[-1])
                #else:
                #    fstate.create_dataset(var, data=getattr(self, var))
                if var == "a":
                    var_ = self.a_nominal
                elif var == "start_time":
                    var_ = self.time[-1]
                else:
                    var_ = getattr(self, var)
                if var in fstate:
                    # dataset already exists
                    fstate[var][...] = var_
                else:
                    fstate.create_dataset(var, data=var_)


class RateStateFault(object):
    def __init__(
        self,
        fault_patches=None,
        neighboring_patches=None,
        a_reduction_factor=0.1,
        V_V0_ratio_for_update=1.25,
        record_history=True,
        verbose=True,
        path=None,
    ):
        if path is None and fault_patches is None:
            print("You need to specify fault_patches or path!")
            return
        if path is None:
            self.fault_patches = fault_patches
        else:
            with h5.File(path, mode="r") as fprop:
                gids = list(fprop.keys())
                print(gids)
            self.fault_patches = [
                    RateStateFaultPatch(
                        path=path,
                        gid=gids[i],
                        record_history=record_history
                        )
                    for i in range(len(gids))
                    ]
        for i, fp in enumerate(self.fault_patches):
            if fp.fault_patch_id is None:
                fp.fault_patch_id = i
            # ----------------------------------------------------------------
            if neighboring_patches is not None:
                fp.neighbors = neighboring_patches[i]
            else:
                fp.neighbors = []
        self.verbose = verbose
        self.n_patches = len(self.fault_patches)
        self.coords = np.asarray(
                [[fp.x, fp.y, fp.z] for fp in self.fault_patches]
                )
        self.a_reduction_factor = a_reduction_factor
        self.V_V0_ratio_for_update = V_V0_ratio_for_update
        self.fault_area = sum([fp.area for fp in self.fault_patches])
        self.locked_patches = np.zeros(self.n_patches, dtype=bool)
        self.record_history = record_history

    @property
    def time(self):
        return self.start_time + np.cumsum(np.asarray(self._time_increments))

    @property
    def time_increments(self):
        return np.asarray(self._time_increments)

    @property
    def shear_stress_history(self):
        return np.asarray(self._shear_stress_history)

    @property
    def normal_stress_history(self):
        return np.asarray(self._normal_stress_history)

    @property
    def shear_stress(self):
        return (
            sum(fp.shear_stress * fp.area for fp in self.fault_patches)
            / self.fault_area
        )

    @property
    def normal_stress(self):
        return (
            sum(fp.normal_stress * fp.area for fp in self.fault_patches)
            / self.fault_area
        )

    def initialize_mechanical_state(self, path=None):
        if path is not None:
            for fp in self.fault_patches:
                fp.initialize_mechanical_state(
                        path=path, gid=str(fp.fault_patch_id)
                        )
        #if path is None:
        self.Kij_shear = np.zeros((self.n_patches, self.n_patches), dtype=np.float64)
        self.Kij_normal = np.zeros((self.n_patches, self.n_patches), dtype=np.float64)
        tectonic_slip_speeds = np.zeros(self.n_patches, dtype=np.float64)
        for i, fp in enumerate(self.fault_patches):
            tectonic_slip_speeds[i] = fp.tectonic_slip_speed
            # ----------------------------------------------------------------
            #      attach a link to the parent fault object to each fault patch
            fp.parent = self
            normal_bloc = np.asarray(fp.n).reshape(-1, 1)
            # if normal_bloc[2] == 0.:
            #    # vertical fault: make sure the normal points southward
            #    if normal_bloc[1] > 0.:
            #        # if this is not the case, take the normal to the opposite wall
            #        normal_bloc *= -1.
            # if normal_bloc[2] < 0.:
            #    # always take the normal upward, i.e. normal to the footwall
            #    normal_bloc *= -1.
            # =================================================================
            # coseismic motion is expected to occur in the direction of tectonic loading,
            # i.e. the x direction
            # ================================================================
            alpha_i = (fp.lame1 + fp.lame2) / (fp.lame1 + 2.0 * fp.lame2)
            for j, fp_j in enumerate(self.fault_patches):
                x = fp.x - fp_j.x
                y = fp.y - fp_j.y
                z = fp.z
                P = np.float64([x, y, z])
                P2 = np.dot(fp_j.change_x_y, P.reshape(-1, 1)).flatten()
                # displacement on fault patch i due to left-lateral slip on fault patch j
                #             strike slip component
                # positive strike slip = left-lateral
                success, U, grad_U = dc3dwrapper(
                    alpha_i,
                    P2,
                    abs(fp_j.z),
                    fp_j.dip_angle,
                    [-fp_j.L / 2, +fp_j.L / 2],
                    [-fp_j.W / 2, +fp_j.W / 2],
                    [fp_j.strike_slip_component, 0.0, 0.0],
                )
                # success, U, grad_U = dc3dwrapper(1., [0., 0., -abs(coords_fault_patches[j,2])], abs(coords_fault_patches[j,2]), \
                #                                 fp_j.dip_angle, [-fp_j.L/2, +fp_j.L/2], [-fp_j.W/2, +fp_j.W/2], \
                #                                 [-1., 0., 0.])
                stress_tensor_fault_j_frame = fp.lame2 * (
                    grad_U + grad_U.T
                ) + fp.lame1 * np.identity(3) * np.trace(grad_U)
                traction = np.dot(stress_tensor_fault_j_frame, normal_bloc).flatten()
                stress_tensor_canonical_frame = np.dot(
                    np.dot(fp_j.change_x_y.T, stress_tensor_fault_j_frame),
                    fp_j.change_x_y,
                )
                traction = np.dot(stress_tensor_canonical_frame, normal_bloc).flatten()
                # traction = np.dot(fp_j.change_x_y.T, np.dot(stress_tensor_fault_j_frame, np.dot(fp_j.change_x_y, normal_bloc))).flatten()
                if success != 0:
                    print(
                        "Displacement field calculation failed for interactions "
                        f"between patch {i} and patch {j}!"
                        )
                self.Kij_shear[i, j] = np.dot(traction, -fp.p)
                # the normal is defined going outward of the fault plane, but normal stress is counted positive for compressional stresses, hence x-1
                self.Kij_normal[i, j] += np.dot(traction, normal_bloc) * -1.0
                # displacement on fault patch i due to reverse faulting on fault patch j
                #             dip slip component
                # positive dip slip = reverse faulting
                success, U, grad_U = dc3dwrapper(
                    alpha_i,
                    P2,
                    abs(fp_j.z),
                    fp_j.dip_angle,
                    [-fp_j.L / 2, +fp_j.L / 2],
                    [-fp_j.W / 2, +fp_j.W / 2],
                    [0.0, fp_j.dip_slip_component, 0.0],
                )
                stress_tensor_fault_j_frame = fp.lame2 * (
                    grad_U + grad_U.T
                ) + fp.lame1 * np.identity(3) * np.trace(grad_U)
                stress_tensor_canonical_frame = np.dot(
                    np.dot(fp_j.change_x_y.T, stress_tensor_fault_j_frame),
                    fp_j.change_x_y,
                )
                traction = np.dot(stress_tensor_canonical_frame, normal_bloc).flatten()
                # traction = np.dot(fp_j.change_x_y.T, np.dot(stress_tensor_fault_j_frame, np.dot(fp_j.change_x_y, normal_bloc))).flatten()
                # if j == i:
                #    print '-------------------'
                #    print normal_bloc
                #    print np.dot(fp_j.change_x_y, normal_bloc)
                #    print np.dot(stress_tensor_fault_j_frame, np.dot(fp_j.change_x_y, normal_bloc))
                #    print np.dot(fp_j.change_x_y.T, np.dot(stress_tensor_fault_j_frame, np.dot(fp_j.change_x_y, normal_bloc)))
                #    print np.dot(np.dot(fp_j.change_x_y.T, np.dot(stress_tensor_fault_j_frame, np.dot(fp_j.change_x_y, normal_bloc))).T, normal_bloc)
                if success != 0:
                    print(
                        "Displacement field calculation failed for interactions between patch {:d} and patch {:d} !".format(
                            i, j
                        )
                    )
                self.Kij_shear[i, j] += np.dot(traction, -fp.p)
                # the normal is defined going outward of the fault plane,
                # but normal stress is counted positive for compressional stresses, hence x-1
                self.Kij_normal[i, j] += np.dot(traction, normal_bloc) * -1.0
                # -----------------------------------------------------------
        self.tectonic_slip_speeds = tectonic_slip_speeds
        for i, fp in enumerate(self.fault_patches):
            fp.ktau_tectonic = np.float64(
                -1.0
                * (
                    np.sum(self.Kij_shear[i, :] * tectonic_slip_speeds)
                    / fp.tectonic_slip_speed
                )
            )
            fp.ksigma_tectonic = np.float64(
                -1.0
                * (
                    np.sum(self.Kij_normal[i, :] * tectonic_slip_speeds)
                    / fp.tectonic_slip_speed
                )
            )
            fp.k = np.float64(abs(self.Kij_shear[i, i]))
            fp.update_H()

        # initialize stressing rates
        self.update_stressing_rates()
        for i, fp in enumerate(self.fault_patches):
            if fp.state == 3:
                # fault patch was initialized from previous simulation
                continue
            elif fp.k > fp.critical_stiffness():
                # stable patch (creeping)
                fp.stable = True
                fp.state = 3
                fp.d_dot = fp.tectonic_slip_speed
                # ! set ktau_tectonic to 0 !
                # when
                fp.set_theta_to_steady_state()
                fp.set_friction_to_steady_state()
                fp.shear_stress = self.fault_patches[
                    i
                ].friction  # stable steady state
                fp.shear_stress_history[0] = self.fault_patches[
                    i
                ].shear_stress
                print(f"Patch {i} is stable.")
                fp.d_dot_0 = np.float64(fp.d_dot)
            else:
                # unstable patch
                fp.stable = False
                fp.d_dot_0 = np.float64(fp.d_dot)
        # update stressing rates to account for creeping patches
        self.update_stressing_rates()
        self.start_time = self.fault_patches[0].start_time
        if self.record_history:
            for fp in self.fault_patches:
                fp.initialize_history()
            self._time_increments = [0.]
            self._shear_stress_history = [self.shear_stress]
            self._normal_stress_history = [self.normal_stress]
            self._history_variables = [
                    "_time_increments",
                    "_shear_stress_history",
                    "_normal_stress_history",
                    ]


    def _evolve_one_patch(self, patch_index):
        fp = self.fault_patches[patch_index]
        if self.locked_patches[patch_index]:
            transition_time = np.finfo(np.float64).max
        else:
            if fp.state == 0:
                t0_1 = fp.find_t0_1()
                t0_3 = fp.find_tX_3()
                if t0_1 < t0_3:
                    fp.next_state = 1
                    transition_time = t0_1
                else:
                    fp.next_state = 3
                    transition_time = t0_3
            elif fp.state == 1:
                t1_2 = fp.find_t1_2()
                t1_3 = fp.find_tX_3()
                if t1_2 < t1_3:
                    fp.next_state = 2
                    transition_time = t1_2
                else:
                    fp.next_state = 3
                    transition_time = t1_3
            elif fp.state == 2:
                transition_time = fp.find_t2_0()
                fp.next_state = 0
            else:
                t3_1 = fp.find_t3_1()
                t3_3 = fp.find_t3_V_update(V_V0_ratio=self.V_V0_ratio_for_update)
                # t3_3 = fp.find_t3_tau_update(num_tc=1.0)
                if t3_3 < t3_1:
                    fp.next_state = 3
                    transition_time = t3_3
                else:
                    fp.next_state = 1
                    transition_time = t3_1
        fp._transition_time = transition_time

    def evolve_next_patch(self):
        for i in range(self.n_patches):
            self._evolve_one_patch(i)
        t = [fp._transition_time for fp in self.fault_patches]
        t = np.round(t, decimals=DECIMAL_PRECISION)
        # print(t / (24. * 3600.))
        evolving_patch_indexes = np.where(t == t.min())[0]
        if len(evolving_patch_indexes) == 0:
            print(t.min(), t)
        # print t.min()
        if t.min() < 0.0:
            # why does this happen??
            # for idx in evolving_patch_indexes:
            #    print(
            #        self.fault_patches[idx].state,
            #        self.fault_patches[idx].shear_stress / 1.0e6,
            #        self.fault_patches[idx].friction / 1.0e6,
            #    )
            t[t < 0.0] = 10.0 ** (-DECIMAL_PRECISION)
        times = np.hstack(
            (t.min(), np.zeros(evolving_patch_indexes.size - 1, dtype=np.float64))
        )
        print_update_message = True
        for i, evolving_patch_idx in enumerate(evolving_patch_indexes):
            # self.time_jumps = np.hstack((self.time_jumps, times[i]))
            state_0 = self.fault_patches[evolving_patch_idx].state
            if (self.fault_patches[evolving_patch_idx].state == 0) and (
                self.fault_patches[evolving_patch_idx].next_state == 1
            ):
                self.update_all_state_01(evolving_patch_idx, times[i])
                # print('Transition time 0 -> 1 = {:.7e}s'.format(times[i]))
            elif (self.fault_patches[evolving_patch_idx].state == 0) and (
                self.fault_patches[evolving_patch_idx].next_state == 3
            ):
                self.update_all_state_03(evolving_patch_idx, times[i])
            elif (self.fault_patches[evolving_patch_idx].state == 1) and (
                self.fault_patches[evolving_patch_idx].next_state == 2
            ):
                self.update_all_state_12(evolving_patch_idx, times[i])
            elif (self.fault_patches[evolving_patch_idx].state == 1) and (
                self.fault_patches[evolving_patch_idx].next_state == 3
            ):
                self.update_all_state_13(evolving_patch_idx, times[i])
            elif self.fault_patches[evolving_patch_idx].state == 2:
                self.update_all_state_2(evolving_patch_idx, times[i])
            elif (self.fault_patches[evolving_patch_idx].state == 3) and (
                self.fault_patches[evolving_patch_idx].next_state == 3
            ):
                self.update_all_state_33(evolving_patch_idx, times[i])
                # print_update_message = False
            elif (self.fault_patches[evolving_patch_idx].state == 3) and (
                self.fault_patches[evolving_patch_idx].next_state == 1
            ):
                self.update_all_state_31(evolving_patch_idx, times[i])
            if print_update_message and self.verbose:
                print(
                    "Evolving patch {:d} from state {:d} to state {:d} (time = {:.2e}s)".format(
                        evolving_patch_idx,
                        state_0,
                        self.fault_patches[evolving_patch_idx].next_state,
                        times[i],
                    )
                )
        # for j in range(self.n_patches):
        #    if self.fault_patches[j].shear_stress_rate < 1.0:
        #        self.fault_patches[j].shear_stress_rate = np.round(
        #            self.fault_patches[j].shear_stress_rate,
        #            decimals=DECIMAL_PRECISION
        #        )
        if self.record_history:
            #self._time.append(times[0] + self.time[-1])
            self._time_increments.append(times[0])
            self._shear_stress_history.append(self.shear_stress)
            self._normal_stress_history.append(self.normal_stress)

    def update_all_state_01(self, evolving_patch_idx, t0_1):
        """
        Update the fault patch with index = evolving_patch_idx from state 0 to state 1.
        """
        for i in range(self.n_patches):
            if i == evolving_patch_idx:
                self.fault_patches[i].state_0(t0_1=t0_1)
                if self.fault_patches[i].record_history:
                    self.fault_patches[i]._state_history.append(1)
                    self.fault_patches[i]._transition_times_history.append(
                        self.fault_patches[i].time[-1]
                    )
                continue
            self.fault_patches[i].evolve_current_state(t0_1, self, i)

    def update_all_state_03(self, evolving_patch_idx, t0_3):
        """
        Update the fault patch with index = evolving_patch_idx from state 0 to state 3.
        """
        for i in range(self.n_patches):
            if i == evolving_patch_idx:
                self.fault_patches[i].stable = True
                self.fault_patches[i].state = 3
                self.fault_patches[i].normal_stress += (
                    self.fault_patches[i].normal_stress_rate * t0_3
                )
                self.fault_patches[i].shear_stress += (
                    self.fault_patches[i].shear_stress_rate * t0_3
                )
                # -------------------------------------------------------------
                #    the current normal and shear stresses, given K_patch > K_critical,
                #    define a new slip speed that satisfies steady state
                self.fault_patches[i].d_dot = self.fault_patches[
                    i
                ].slip_speed_creeping_patch(0.0)
                self.fault_patches[i].set_theta_to_steady_state()
                if self.fault_patches[i].record_history:
                    self.fault_patches[i]._state_history.append(3)
                    self.fault_patches[i]._transition_times_history.append(
                        self.fault_patches[i].time[-1]
                    )
                continue
            self.fault_patches[i].evolve_current_state(t0_3, self, i)
        # -----------------------
        self.update_stressing_rates()

    def update_all_state_12(self, evolving_patch_idx, t1_2):
        """
        Update the fault patch with index = evolving_patch_idx from state 1 to state 2.
        """
        for i in range(self.n_patches):
            if i == evolving_patch_idx:
                self.fault_patches[i].state_1(t1_2=t1_2)
                self.fault_patches[i].stress_0 = self.fault_patches[i].shear_stress
                for patch in self.fault_patches[i].neighbors:
                    # lower rate-state a parameter to facilitate rupture
                    # propagation, following Richards-Dinger and Dieterich 2012
                    self.fault_patches[patch].a = (
                        self.a_reduction_factor * self.fault_patches[patch].a_nominal
                    )
                if self.fault_patches[i].record_history:
                    self.fault_patches[i]._state_history.append(2)
                    self.fault_patches[i]._transition_times_history.append(
                        self.fault_patches[i].time[-1]
                    )
                continue
            self.fault_patches[i].evolve_current_state(t1_2, self, i)
        # -----------------------
        self.update_stressing_rates()

    def update_all_state_13(self, evolving_patch_idx, t1_3):
        """
        Update the fault patch with index = evolving_patch_idx from state 1 to state 3.
        """
        for i in range(self.n_patches):
            if i == evolving_patch_idx:
                self.fault_patches[i].stable = True
                self.fault_patches[i].state = 3
                self.fault_patches[i].normal_stress += (
                    self.fault_patches[i].normal_stress_rate * t1_3
                )
                self.fault_patches[i].shear_stress += (
                    self.fault_patches[i].shear_stress_rate * t1_3
                )
                # -------------------------------------------------------------
                #    the current normal and shear stresses, given K_patch > K_critical,
                #    define a new slip speed that satisfies steady state
                self.fault_patches[i].d_dot = self.fault_patches[
                    i
                ].slip_speed_creeping_patch(0.0)
                self.fault_patches[i].set_theta_to_steady_state()
                if self.fault_patches[i].record_history:
                    self.fault_patches[i]._state_history.append(3)
                    self.fault_patches[i]._transition_times_history.append(
                        self.fault_patches[i].time[-1]
                    )
                continue
            self.fault_patches[i].evolve_current_state(t1_3, self, i)
        # -----------------------
        self.update_stressing_rates()

    def update_all_state_2(self, evolving_patch_idx, t2_0):
        """
        Update the fault patch with index = evolving_patch_idx from state 2 to state 0.
        """
        # print('Duration State 2: {:.2f}sec'.format(t2_0))
        for i in range(self.n_patches):
            if i == evolving_patch_idx:
                self.fault_patches[i].state_2(t2_0=t2_0, verbose=self.verbose)
                for patch1 in self.fault_patches[i].neighbors:
                    increase_a = True
                    for patch2 in self.fault_patches[patch1].neighbors:
                        if self.fault_patches[patch2].state == 2:
                            # patch2 still has a neighboring patch undergoing earthquake
                            increase_a = False
                    if increase_a:
                        self.fault_patches[patch1].a = self.fault_patches[
                            patch1
                        ].a_nominal
                if self.fault_patches[i].record_history:
                    self.fault_patches[i]._state_history.append(0)
                    self.fault_patches[i]._transition_times_history.append(
                        self.fault_patches[i].time[-1]
                    )
                continue
            self.fault_patches[i].evolve_current_state(t2_0, self, i)
        # ---------------------------------------
        self.update_stressing_rates()

    def update_all_state_31(self, evolving_patch_idx, t3_1):
        """
        Update the fault patch with index = evolving_patch_idx from state 3 to state 1.
        """
        for i in range(self.n_patches):
            if i == evolving_patch_idx:
                self.fault_patches[i].state_3(self, i, t3_1=t3_1)
                if self.fault_patches[i].record_history:
                    self.fault_patches[i]._state_history.append(1)
                    self.fault_patches[i]._transition_times_history.append(
                        self.fault_patches[i].time[-1]
                    )
                continue
            self.fault_patches[i].evolve_current_state(t3_1, self, i)

    def update_all_state_33(self, evolving_patch_idx, t3_3):
        """
        Update the fault patch with index = evolving_patch_idx from state 3 to state 3.
        """
        v0 = self.fault_patches[evolving_patch_idx].d_dot
        for i in range(self.n_patches):
            self.fault_patches[i].evolve_current_state(t3_3, self, i)
        delta_d_dot = self.fault_patches[evolving_patch_idx].d_dot - v0
        # -----------------------
        self.update_stressing_rates()

    def update_stressing_rates(self):
        for i, fp1 in enumerate(self.fault_patches):
            fp1.shear_stress_rate = fp1.ktau_tectonic * fp1.tectonic_slip_speed
            fp1.normal_stress_rate = fp1.ksigma_tectonic * fp1.tectonic_slip_speed
            # sliping_patches         = np.ones(self.n_patches, dtype=np.bool)
            for j, fp2 in enumerate(self.fault_patches):
                if (fp2.state == 0) or (fp2.state == 1):
                    # sliping_patches[j] = False
                    continue
                fp1.shear_stress_rate += self.Kij_shear[i, j] * fp2.d_dot
                fp1.normal_stress_rate += self.Kij_normal[i, j] * fp2.d_dot
            # fp1.shear_stress_rate   += np.sum(self.Kij_shear[i,sliping_patches] ) * fp1.d_dot_EQ
            # fp1.normal_stress_rate  += np.sum(self.Kij_normal[i,sliping_patches]) * fp1.d_dot_EQ

    def get_stress_state(self):
        stress = np.zeros(self.n_patches, dtype=np.float64)
        for i in range(self.n_patches):
            stress[i] = self.fault_patches[i].shear_stress
        return stress

    # I/O methods
    def save_properties(self, path, overwrite=True):
        for fp in self.fault_patches:
            fp.save_properties(path, gid=str(fp.fault_patch_id), overwrite=overwrite)

    def save_mechanical_state(self, path, overwrite=True):
        for fp in self.fault_patches:
            fp.save_mechanical_state(
                    path, gid=str(fp.fault_patch_id), overwrite=overwrite
                    )

    def save_history(self, path, var=None, var_fault=None, overwrite=False):
        for fp in self.fault_patches:
            fp.save_history(
                    path, gid=str(fp.fault_patch_id), var=var, overwrite=overwrite
                    )
        if var_fault is None:
            var_fault_pool = self._history_variables
        else:
            var_fault_pool = var_fault
        with h5.File(path, mode="a") as ffault:
            if "fault" not in ffault:
                ffault.create_group("fault")
            ffault = ffault["fault"]
            for var_fault in var_fault_pool:
                # ignore the trailing character
                var_fault = var_fault[1:]
                if overwrite and var_fault in ffault:
                    del ffault[var_fault]
                time_series = getattr(self, var_fault)
                if var_fault not in ffault:
                    # initialize data set
                    ffault.create_dataset(
                            var_fault,
                            data=time_series,
                            compression="gzip",
                            chunks=True,
                            maxshape=(None,),
                            )
                else:
                    # append to existing data set
                    ffault[var_fault].resize(
                            ffault[var_fault].shape[0] + len(time_series),
                            axis=0
                            )
                    ffault[var_fault][-len(time_series):] = time_series

    def clean_history(self):
        if self.record_history:
            for fp in self.fault_patches:
                fp.clean_history()
            self.start_time = self.time[-1]
            for attr in self._history_variables:
                setattr(self, attr, [getattr(self, attr)[-1]])

