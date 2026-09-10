#!/usr/bin/env python

import cffirmware
import pytest

UINT16_MAX = (1 << 16) - 1


def _absolute_z_output(mass, mass_thrust, position_z=0.0, integral_gain_z=None):
    ctrl = cffirmware.controllerMellinger_t()
    cffirmware.controllerMellingerInit(ctrl)
    ctrl.mass = mass
    ctrl.massThrust = mass_thrust
    if integral_gain_z is not None:
        ctrl.ki_z = integral_gain_z

    control = cffirmware.control_t()
    setpoint = cffirmware.setpoint_t()
    setpoint.mode.x = cffirmware.modeAbs
    setpoint.mode.y = cffirmware.modeAbs
    setpoint.mode.z = cffirmware.modeAbs
    setpoint.mode.yaw = cffirmware.modeAbs
    setpoint.position.z = position_z

    state = cffirmware.state_t()
    state.attitudeQuaternion.w = 1.0
    sensors = cffirmware.sensorData_t()

    cffirmware.controllerMellinger(ctrl, control, setpoint, sensors, state, 100)
    return ctrl, control


def test_controller_mellinger():

    ctrl = cffirmware.controllerMellinger_t()

    cffirmware.controllerMellingerInit(ctrl)

    control = cffirmware.control_t()
    setpoint = cffirmware.setpoint_t()
    setpoint.mode.z = cffirmware.modeAbs
    setpoint.position.z = 0
    setpoint.mode.x = cffirmware.modeVelocity
    setpoint.velocity.x = 0
    setpoint.mode.y = cffirmware.modeVelocity
    setpoint.velocity.y = 0
    setpoint.mode.yaw = cffirmware.modeVelocity
    setpoint.attitudeRate.yaw = 0

    state = cffirmware.state_t()
    state.attitude.roll = 0
    state.attitude.pitch = -0 # WARNING: This needs to be negated
    state.attitude.yaw = 0
    state.position.x = 0
    state.position.y = 0
    state.position.z = 0
    state.velocity.x = 0
    state.velocity.y = 0
    state.velocity.z = 0

    sensors = cffirmware.sensorData_t()
    sensors.gyro.x = 0
    sensors.gyro.y = 0
    sensors.gyro.z = 0

    step = 100

    cffirmware.controllerMellinger(ctrl, control, setpoint,sensors,state,step)
    assert control.controlMode == cffirmware.controlModeLegacy
    # control.thrust will be at a (tuned) hover-state
    assert control.roll == 0
    assert control.pitch == 0
    assert control.yaw == 0


def test_mass_thrust_is_compatibility_noop():
    _, low_legacy_scalar = _absolute_z_output(0.037, 1.0, position_z=0.013)
    _, high_legacy_scalar = _absolute_z_output(0.037, 250000.0, position_z=0.013)

    assert low_legacy_scalar.controlMode == cffirmware.controlModeLegacy
    assert low_legacy_scalar.thrust == high_legacy_scalar.thrust


def test_physical_mass_remains_active():
    _, light = _absolute_z_output(0.031, 12345.0, position_z=0.013)
    _, heavy = _absolute_z_output(0.053, 12345.0, position_z=0.013)

    assert heavy.thrust > light.thrust


def test_collective_force_is_normalized_by_platform_max_thrust():
    fraction_of_platform_max = 0.375
    artificial_mass = 0.037
    ctrl, baseline = _absolute_z_output(artificial_mass, 12345.0, integral_gain_z=0.0)
    position_step = (
        fraction_of_platform_max * cffirmware.powerDistributionGetMaxThrust() / ctrl.kp_z
    )
    _, stepped = _absolute_z_output(
        artificial_mass,
        12345.0,
        position_z=position_step,
        integral_gain_z=0.0,
    )

    expected_code_step = fraction_of_platform_max * UINT16_MAX
    assert stepped.thrust - baseline.thrust == pytest.approx(expected_code_step, rel=2e-6)
