#!/usr/bin/env python

import cffirmware
import pytest


def _hover_output(mass, mass_thrust):
    ctrl = cffirmware.controllerMellinger_t()
    cffirmware.controllerMellingerInit(ctrl)
    ctrl.mass = mass
    ctrl.massThrust = mass_thrust

    control = cffirmware.control_t()
    setpoint = cffirmware.setpoint_t()
    setpoint.mode.x = cffirmware.modeAbs
    setpoint.mode.y = cffirmware.modeAbs
    setpoint.mode.z = cffirmware.modeAbs
    setpoint.mode.yaw = cffirmware.modeAbs

    state = cffirmware.state_t()
    state.attitudeQuaternion.w = 1.0
    sensors = cffirmware.sensorData_t()

    cffirmware.controllerMellinger(ctrl, control, setpoint, sensors, state, 100)
    return control


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


def test_collective_force_uses_platform_legacy_force_code():
    mass = 0.04338
    control = _hover_output(mass, 132000.0)

    expected = mass * 9.81 / cffirmware.powerDistributionGetMaxThrust() * 65535.0

    assert control.controlMode == cffirmware.controlModeLegacy
    assert control.thrust == pytest.approx(expected, rel=2e-6)
    assert 0.0 < control.thrust < 65535.0


@pytest.mark.parametrize(
    ("collective_force", "expected_code"),
    ((0.0, 0.0), (0.2, 16383.75), (0.4, 32767.5), (0.8, 65535.0)),
)
def test_collective_force_known_points(collective_force, expected_code):
    mass = collective_force / 9.81
    control = _hover_output(mass, 132000.0)

    assert control.thrust == pytest.approx(expected_code, rel=2e-6, abs=1e-6)


def test_mass_thrust_is_compatible_noop_and_physical_mass_remains_active():
    light = _hover_output(0.040, 1.0)
    same_mass_different_legacy_scalar = _hover_output(0.040, 250000.0)
    heavy = _hover_output(0.050, 1.0)

    assert light.thrust == same_mass_different_legacy_scalar.thrust
    assert heavy.thrust > light.thrust
    assert heavy.thrust / light.thrust == pytest.approx(0.050 / 0.040, rel=2e-6)
