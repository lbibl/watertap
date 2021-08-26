###############################################################################
# ProteusLib Copyright (c) 2021, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National
# Laboratory, National Renewable Energy Laboratory, and National Energy
# Technology Laboratory (subject to receipt of any required approvals from
# the U.S. Dept. of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/nawi-hub/proteuslib/"
#
###############################################################################

"""Seawater feed specifications for supported property packages"""

from pyomo.environ import ConcreteModel, Constraint
from idaes.core import FlowsheetBlock
from idaes.core.util.scaling import calculate_scaling_factors
import proteuslib.property_models.seawater_prop_pack as seawater_prop_pack
import proteuslib.flowsheets.full_treatment_train.example_models.property_seawater_salts as property_seawater_salts
import proteuslib.flowsheets.full_treatment_train.example_models.property_seawater_ions as property_seawater_ions
from proteuslib.flowsheets.full_treatment_train.util import solve_with_user_scaling


def specify_seawater_TDS(sb):
    """
    Fixes the state variables on the stateblock to the base
    seawater composition for the seawater_prop_pack property package.
    """
    # specifying
    feed_flow_mass = 1
    feed_mass_frac_TDS = 0.035
    sb.flow_mass_phase_comp['Liq', 'TDS'].fix(feed_flow_mass * feed_mass_frac_TDS)
    sb.flow_mass_phase_comp['Liq', 'H2O'].fix(feed_flow_mass * (1 - feed_mass_frac_TDS))
    sb.pressure.fix(101325)
    sb.temperature.fix(298.15)


def set_default_scaling_TDS(property_parameter_block):
    property_parameter_block.set_default_scaling('flow_mass_phase_comp', 1, index=('Liq', 'H2O'))
    property_parameter_block.set_default_scaling('flow_mass_phase_comp', 1e2, index=('Liq', 'TDS'))


def specify_seawater_salts(sb):
    """
    Fixes the state variables on the stateblock to the base
    seawater composition for the property_seawater_salts property package.
    """
    # specify
    feed_flow_mass = 1
    feed_mass_frac = {'NaCl': 2.827e-2,
                      'CaSO4': 1.298e-3,
                      'MgSO4': 1.529e-3,
                      'MgCl2': 4.251e-3,
                      'H2O': 0.9647}
    for s in feed_mass_frac:
        sb.flow_mass_phase_comp['Liq', s].fix(feed_flow_mass * feed_mass_frac[s])
    sb.pressure.fix(101325)
    sb.temperature.fix(298.15)


def set_default_scaling_salts(property_parameter_block):
    property_parameter_block.set_default_scaling('flow_mass_phase_comp', 1, index=('Liq', 'H2O'))
    property_parameter_block.set_default_scaling('flow_mass_phase_comp', 1e2, index=('Liq', 'NaCl'))
    property_parameter_block.set_default_scaling('flow_mass_phase_comp', 1e3, index=('Liq', 'CaSO4'))
    property_parameter_block.set_default_scaling('flow_mass_phase_comp', 1e3, index=('Liq', 'MgSO4'))
    property_parameter_block.set_default_scaling('flow_mass_phase_comp', 1e3, index=('Liq', 'MgCl2'))


def specify_seawater_ions(sb):
    """
    Fixes the state variables on the stateblock to the base
    seawater composition for the property_seawater_ions property package.
    """
    # specify
    feed_flow_mass = 1
    feed_mass_frac = {'Na': 11122e-6,
                      'Ca': 382e-6,
                      'Mg': 1394e-6,
                      'SO4': 2136e-6,
                      'Cl': 20316.88e-6}
    sb.flow_mass_phase_comp['Liq', 'H2O'].fix(
        feed_flow_mass * (1 - sum(x for x in feed_mass_frac.values())))
    for j in feed_mass_frac:
        sb.flow_mass_phase_comp['Liq', j].fix(feed_flow_mass * feed_mass_frac[j])
    sb.pressure.fix(101325)
    sb.temperature.fix(298.15)


def set_default_scaling_ions(property_parameter_block):
    property_parameter_block.set_default_scaling('flow_mass_phase_comp', 1, index=('Liq', 'H2O'))
    property_parameter_block.set_default_scaling('flow_mass_phase_comp', 1e2, index=('Liq', 'Na'))
    property_parameter_block.set_default_scaling('flow_mass_phase_comp', 1e4, index=('Liq', 'Ca'))
    property_parameter_block.set_default_scaling('flow_mass_phase_comp', 1e3, index=('Liq', 'Mg'))
    property_parameter_block.set_default_scaling('flow_mass_phase_comp', 1e3, index=('Liq', 'SO4'))
    property_parameter_block.set_default_scaling('flow_mass_phase_comp', 1e2, index=('Liq', 'Cl'))


def run_specify_seawater(case):
    # build state block
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})
    if case == 'TDS':
        m.fs.properties = seawater_prop_pack.SeawaterParameterBlock()
        m.fs.stream = m.fs.properties.build_state_block([0], default={})
        specify_seawater_TDS(m.fs.stream[0])
        set_default_scaling_TDS(m.fs.properties)
    elif case == 'salts':
        m.fs.properties = property_seawater_salts.PropParameterBlock()
        m.fs.stream = m.fs.properties.build_state_block([0], default={})
        specify_seawater_salts(m.fs.stream[0])
        set_default_scaling_salts(m.fs.properties)
    elif case == 'ions':
        m.fs.properties = property_seawater_ions.PropParameterBlock()
        m.fs.stream = m.fs.properties.build_state_block([0], default={})
        specify_seawater_ions(m.fs.stream[0])
        set_default_scaling_ions(m.fs.properties)

    m.fs.stream[0].mass_frac_phase_comp  # touch a variable to have a model with at least one constraint

    # scale
    calculate_scaling_factors(m.fs)
    # solve
    solve_with_user_scaling(m)
    # display
    m.fs.stream.display()


if __name__ == "__main__":
    run_specify_seawater('TDS')
    run_specify_seawater('salts')
    run_specify_seawater('ions')
