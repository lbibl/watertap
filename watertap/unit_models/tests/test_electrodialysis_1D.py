###############################################################################
# WaterTAP Copyright (c) 2021, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National
# Laboratory, National Renewable Energy Laboratory, and National Energy
# Technology Laboratory (subject to receipt of any required approvals from
# the U.S. Dept. of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#
###############################################################################
import pytest
from watertap.property_models.ion_DSPMDE_prop_pack import DSPMDEParameterBlock
from watertap.unit_models.electrodialysis_1D import (
    ElectricalOperationMode,
    Electrodialysis1D,
    PressureDropMethod,
    FrictionFactorMethod,
    HydraulicDiameterMethod,
)
from watertap.costing import WaterTAPCosting
from pyomo.environ import (
    ConcreteModel,
    assert_optimal_termination,
    value,
    Set,
    Param,
    Var,
    units as pyunits,
    Suffix,
    Constraint,
    SolverFactory,
    SolverStatus,
    TerminationCondition,
)
from idaes.core import (
    FlowsheetBlock,
    MaterialFlowBasis,
    MaterialBalanceType,
    MomentumBalanceType,
    EnergyBalanceType,
    UnitModelCostingBlock,
)
from idaes.core.util.exceptions import ConfigurationError
from idaes.core.util.model_statistics import degrees_of_freedom
from pyomo.util.check_units import assert_units_consistent
import idaes.core.util.scaling as iscale
from idaes.core.util.testing import initialization_tester
from idaes.core.solvers import get_solver
import re

__author__ = "Xiangyu Bi"

solver = get_solver()

# -----------------------------------------------------------------------------
# Start test class
class TestElectrodialysisVoltageConst:
    @pytest.fixture(scope="class")
    def electrodialysis_1d_cell1(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        ion_dict = {
            "solute_list": ["Na_+", "Cl_-"],
            "mw_data": {"H2O": 18e-3, "Na_+": 23e-3, "Cl_-": 35.5e-3},
            "elec_mobility_data": {("Liq", "Na_+"): 5.19e-8, ("Liq", "Cl_-"): 7.92e-8},
            "charge": {"Na_+": 1, "Cl_-": -1},
        }
        m.fs.properties = DSPMDEParameterBlock(**ion_dict)
        m.fs.unit = Electrodialysis1D(
            property_package=m.fs.properties,
            operation_mode=ElectricalOperationMode.Constant_Voltage,
            finite_elements=20,
            pressure_drop_method=PressureDropMethod.none,
        )
        return m

    @pytest.mark.unit
    def test_build_model(self, electrodialysis_1d_cell1):
        m = electrodialysis_1d_cell1
        # test configrations
        assert len(m.fs.unit.config) == 19
        assert not m.fs.unit.config.dynamic
        assert not m.fs.unit.config.has_holdup
        assert (
            m.fs.unit.config.operation_mode == ElectricalOperationMode.Constant_Voltage
        )
        assert m.fs.unit.config.material_balance_type == MaterialBalanceType.useDefault
        assert (
            m.fs.unit.config.momentum_balance_type == MomentumBalanceType.pressureTotal
        )
        assert m.fs.unit.config.property_package is m.fs.properties
        assert "H2O" in m.fs.properties.component_list

        # test all essential params and vars are built
        assert isinstance(m.fs.unit.membrane_set, Set)
        assert isinstance(m.fs.unit.cell_pair_num, Var)
        assert isinstance(m.fs.unit.cell_width, Var)
        assert isinstance(m.fs.unit.cell_length, Var)
        assert isinstance(m.fs.unit.channel_height, Var)
        assert isinstance(m.fs.unit.membrane_thickness, Var)
        assert isinstance(m.fs.unit.solute_diffusivity_membrane, Var)
        assert isinstance(m.fs.unit.ion_trans_number_membrane, Var)
        assert isinstance(m.fs.unit.water_trans_number_membrane, Var)
        assert isinstance(m.fs.unit.water_permeability_membrane, Var)
        assert isinstance(m.fs.unit.membrane_areal_resistance, Var)
        assert isinstance(m.fs.unit.current_density_x, Var)
        assert isinstance(m.fs.unit.voltage_applied, Var)
        assert isinstance(m.fs.unit.voltage_x, Var)
        assert isinstance(m.fs.unit.current_utilization, Var)
        assert isinstance(m.fs.unit.diluate.power_electrical_x, Var)
        assert isinstance(m.fs.unit.specific_power_electrical, Var)
        assert isinstance(m.fs.unit.current_efficiency_x, Var)

        assert isinstance(m.fs.unit.eq_get_total_areal_resistance_x, Constraint)
        assert isinstance(m.fs.unit.eq_get_current_density, Constraint)
        assert isinstance(m.fs.unit.eq_get_voltage_x, Constraint)
        assert isinstance(m.fs.unit.eq_mass_transfer_term_diluate, Constraint)
        assert isinstance(m.fs.unit.eq_mass_transfer_term_concentrate, Constraint)
        assert isinstance(m.fs.unit.eq_power_electrical, Constraint)
        assert isinstance(m.fs.unit.eq_specific_power_electrical, Constraint)
        assert isinstance(m.fs.unit.eq_current_efficiency_x, Constraint)
        assert isinstance(m.fs.unit.eq_isothermal_diluate, Constraint)
        assert isinstance(m.fs.unit.eq_isothermal_concentrate, Constraint)

    @pytest.mark.unit
    def test_stats_constant_vol(self, electrodialysis_1d_cell1):
        m = electrodialysis_1d_cell1
        assert_units_consistent(m)
        assert degrees_of_freedom(m) == 34
        # Specify a system
        # Note: Testing scenarios in this file are primarily in accord with an experimental
        # setup reported by Campione et al. in Desalination 465 (2019): 79-93.
        # set the operational parameters
        m.fs.unit.water_trans_number_membrane["cem"].fix(5.8)
        m.fs.unit.water_trans_number_membrane["aem"].fix(4.3)
        m.fs.unit.water_permeability_membrane["cem"].fix(2.16e-14)
        m.fs.unit.water_permeability_membrane["aem"].fix(1.75e-14)
        m.fs.unit.voltage_applied.fix(0.5)
        m.fs.unit.electrodes_resistance.fix(0)
        m.fs.unit.cell_pair_num.fix(10)
        m.fs.unit.current_utilization.fix(1)
        m.fs.unit.channel_height.fix(2.7e-4)
        m.fs.unit.membrane_areal_resistance["cem"].fix(1.89e-4)
        m.fs.unit.membrane_areal_resistance["aem"].fix(1.77e-4)
        m.fs.unit.cell_width.fix(0.1)
        m.fs.unit.cell_length.fix(0.79)
        m.fs.unit.membrane_thickness["aem"].fix(1.3e-4)
        m.fs.unit.membrane_thickness["cem"].fix(1.3e-4)
        m.fs.unit.solute_diffusivity_membrane["cem", "Na_+"].fix(1.8e-10)
        m.fs.unit.solute_diffusivity_membrane["aem", "Na_+"].fix(1.25e-10)
        m.fs.unit.solute_diffusivity_membrane["cem", "Cl_-"].fix(1.8e-10)
        m.fs.unit.solute_diffusivity_membrane["aem", "Cl_-"].fix(1.25e-10)
        m.fs.unit.ion_trans_number_membrane["cem", "Na_+"].fix(1)
        m.fs.unit.ion_trans_number_membrane["aem", "Na_+"].fix(0)
        m.fs.unit.ion_trans_number_membrane["cem", "Cl_-"].fix(0)
        m.fs.unit.ion_trans_number_membrane["aem", "Cl_-"].fix(1)
        m.fs.unit.spacer_porosity.fix(1)

        # check ion transfer number requirements
        assert (
            sum(
                value(m.fs.unit.ion_trans_number_membrane["cem", j])
                for j in m.fs.properties.ion_set
            )
            == 1
        )
        assert (
            sum(
                value(m.fs.unit.ion_trans_number_membrane["aem", j])
                for j in m.fs.properties.ion_set
            )
            == 1
        )
        assert sum(
            value(m.fs.unit.ion_trans_number_membrane["cem", j])
            for j in m.fs.properties.cation_set
        ) == sum(
            value(m.fs.unit.ion_trans_number_membrane["aem", j])
            for j in m.fs.properties.anion_set
        )

        # set the inlet stream
        m.fs.unit.inlet_diluate.pressure.fix(101325)
        m.fs.unit.inlet_diluate.temperature.fix(298.15)
        m.fs.unit.inlet_diluate.flow_mol_phase_comp[0, "Liq", "H2O"].fix(2.40e-1)
        m.fs.unit.inlet_diluate.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(7.38e-4)
        m.fs.unit.inlet_diluate.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(7.38e-4)
        m.fs.unit.inlet_concentrate.pressure.fix(101325)
        m.fs.unit.inlet_concentrate.temperature.fix(298.15)
        m.fs.unit.inlet_concentrate.flow_mol_phase_comp[0, "Liq", "H2O"].fix(2.40e-1)
        m.fs.unit.inlet_concentrate.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(7.38e-4)
        m.fs.unit.inlet_concentrate.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(7.38e-4)

        assert degrees_of_freedom(m) == 0

    @pytest.mark.component
    def test_initialization_scaling(self, electrodialysis_1d_cell1):
        m = electrodialysis_1d_cell1
        # set default scaling for state vars
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e1, index=("Liq", "H2O")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "Na_+")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "Cl_-")
        )
        # set scaling factors for some vars
        iscale.set_scaling_factor(m.fs.unit.cell_width, 100)
        iscale.set_scaling_factor(m.fs.unit.cell_length, 10)
        iscale.calculate_scaling_factors(m.fs)

        # Added this unit check scaling
        assert_units_consistent(m)

        initialization_tester(m)
        badly_scaled_var_values = {
            var.name: val for (var, val) in iscale.badly_scaled_var_generator(m)
        }
        assert not badly_scaled_var_values
        # check to make sure DOF does not change
        assert degrees_of_freedom(m) == 0

    @pytest.mark.component
    def test_solve(self, electrodialysis_1d_cell1):
        m = electrodialysis_1d_cell1
        # run solver and check for optimal solution
        results = solver.solve(m)
        assert_optimal_termination(results)
        badly_scaled_var_values = {
            var.name: val
            for (var, val) in iscale.badly_scaled_var_generator(m, small=1e-9)
        }
        assert not badly_scaled_var_values

    @pytest.mark.component
    def test_solution(self, electrodialysis_1d_cell1):
        m = electrodialysis_1d_cell1

        assert value(
            m.fs.unit.outlet_diluate.flow_mol_phase_comp[0, "Liq", "H2O"]
        ) == pytest.approx(2.328e-1, rel=1e-3)
        assert value(
            m.fs.unit.outlet_diluate.flow_mol_phase_comp[0, "Liq", "Na_+"]
        ) == pytest.approx(3.223e-04, rel=5e-3)
        assert value(
            m.fs.unit.outlet_diluate.flow_mol_phase_comp[0, "Liq", "Cl_-"]
        ) == pytest.approx(3.223e-04, rel=5e-3)
        assert value(
            m.fs.unit.outlet_concentrate.flow_mol_phase_comp[0, "Liq", "H2O"]
        ) == pytest.approx(2.472e-1, rel=5e-3)
        assert value(
            m.fs.unit.outlet_concentrate.flow_mol_phase_comp[0, "Liq", "Na_+"]
        ) == pytest.approx(1.154e-3, rel=5e-3)
        assert value(
            m.fs.unit.outlet_concentrate.flow_mol_phase_comp[0, "Liq", "Cl_-"]
        ) == pytest.approx(1.154e-3, rel=5e-3)

    @pytest.mark.component
    def test_performance_contents(self, electrodialysis_1d_cell1):
        m = electrodialysis_1d_cell1
        perform_dict = m.fs.unit._get_performance_contents()
        assert "vars" in perform_dict
        assert value(
            perform_dict["vars"]["Total electrical power consumption(Watt)"]
        ) == pytest.approx(3.0, rel=5e-3)
        assert value(
            perform_dict["vars"]["Specific electrical power consumption (kW*h/m**3)"]
        ) == pytest.approx(0.197, rel=5e-3)
        assert value(perform_dict["vars"]["Water recovery by mass"]) == pytest.approx(
            0.485, rel=5e-3
        )

    @pytest.mark.component
    def test_costing(self, electrodialysis_1d_cell1):
        m = electrodialysis_1d_cell1
        blk = m.fs.unit

        m.fs.costing = WaterTAPCosting()

        m.fs.unit.costing = UnitModelCostingBlock(
            flowsheet_costing_block=m.fs.costing,
            costing_method_arguments={"cost_electricity_flow": True},
        )
        m.fs.costing.cost_process()

        assert_units_consistent(m)

        assert degrees_of_freedom(m) == 0

        results = solver.solve(m, tee=True)
        assert_optimal_termination(results)

        assert pytest.approx(388.6800, rel=1e-3) == value(
            m.fs.costing.total_capital_cost
        )
        assert pytest.approx(45.86804, rel=1e-3) == value(
            m.fs.costing.total_operating_cost
        )
        assert pytest.approx(777.3600, rel=1e-3) == value(
            m.fs.costing.total_investment_cost
        )


class TestElectrodialysisCurrentConst:
    @pytest.fixture(scope="class")
    def electrodialysis_1d_cell2(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        ion_dict = {
            "solute_list": ["Na_+", "Cl_-"],
            "mw_data": {"H2O": 18e-3, "Na_+": 23e-3, "Cl_-": 35.5e-3},
            "elec_mobility_data": {("Liq", "Na_+"): 5.19e-8, ("Liq", "Cl_-"): 7.92e-8},
            "charge": {"Na_+": 1, "Cl_-": -1},
        }
        m.fs.properties = DSPMDEParameterBlock(**ion_dict)
        m.fs.unit = Electrodialysis1D(
            property_package=m.fs.properties,
            operation_mode=ElectricalOperationMode.Constant_Current,
            finite_elements=20,
            pressure_drop_method=PressureDropMethod.none,
        )
        return m

    @pytest.mark.unit
    def test_build_model(self, electrodialysis_1d_cell2):
        m = electrodialysis_1d_cell2

        # test configrations
        assert len(m.fs.unit.config) == 19
        assert not m.fs.unit.config.dynamic
        assert not m.fs.unit.config.has_holdup
        assert (
            m.fs.unit.config.operation_mode == ElectricalOperationMode.Constant_Current
        )
        assert m.fs.unit.config.material_balance_type == MaterialBalanceType.useDefault
        assert (
            m.fs.unit.config.momentum_balance_type == MomentumBalanceType.pressureTotal
        )
        assert m.fs.unit.config.property_package is m.fs.properties
        assert "H2O" in m.fs.properties.component_list

        # test all essential params and vars are built
        assert isinstance(m.fs.unit.membrane_set, Set)
        assert isinstance(m.fs.unit.cell_pair_num, Var)
        assert isinstance(m.fs.unit.cell_width, Var)
        assert isinstance(m.fs.unit.cell_length, Var)
        assert isinstance(m.fs.unit.channel_height, Var)
        assert isinstance(m.fs.unit.membrane_thickness, Var)
        assert isinstance(m.fs.unit.solute_diffusivity_membrane, Var)
        assert isinstance(m.fs.unit.ion_trans_number_membrane, Var)
        assert isinstance(m.fs.unit.water_trans_number_membrane, Var)
        assert isinstance(m.fs.unit.water_permeability_membrane, Var)
        assert isinstance(m.fs.unit.membrane_areal_resistance, Var)
        assert isinstance(m.fs.unit.current_applied, Var)
        assert isinstance(m.fs.unit.current_density_x, Var)
        assert isinstance(m.fs.unit.voltage_x, Var)
        assert isinstance(m.fs.unit.current_utilization, Var)
        assert isinstance(m.fs.unit.diluate.power_electrical_x, Var)
        assert isinstance(m.fs.unit.specific_power_electrical, Var)
        assert isinstance(m.fs.unit.current_efficiency_x, Var)

        assert isinstance(m.fs.unit.eq_get_total_areal_resistance_x, Constraint)
        assert isinstance(m.fs.unit.eq_get_current_density, Constraint)
        assert isinstance(m.fs.unit.eq_get_voltage_x, Constraint)
        assert isinstance(m.fs.unit.eq_mass_transfer_term_diluate, Constraint)
        assert isinstance(m.fs.unit.eq_mass_transfer_term_concentrate, Constraint)
        assert isinstance(m.fs.unit.eq_power_electrical, Constraint)
        assert isinstance(m.fs.unit.eq_specific_power_electrical, Constraint)
        assert isinstance(m.fs.unit.eq_current_efficiency_x, Constraint)
        assert isinstance(m.fs.unit.eq_isothermal_diluate, Constraint)
        assert isinstance(m.fs.unit.eq_isothermal_concentrate, Constraint)

    @pytest.mark.unit
    def test_stats_constant_vol(self, electrodialysis_1d_cell2):
        m = electrodialysis_1d_cell2
        assert_units_consistent(m)
        # Specify a system
        # set the operational parameters
        m.fs.unit.water_trans_number_membrane["cem"].fix(5.8)
        m.fs.unit.water_trans_number_membrane["aem"].fix(4.3)
        m.fs.unit.water_permeability_membrane["cem"].fix(2.16e-14)
        m.fs.unit.water_permeability_membrane["aem"].fix(1.75e-14)
        m.fs.unit.current_applied.fix(8)
        m.fs.unit.electrodes_resistance.fix(0)
        m.fs.unit.cell_pair_num.fix(10)
        m.fs.unit.current_utilization.fix(1)
        m.fs.unit.channel_height.fix(2.7e-4)
        m.fs.unit.membrane_areal_resistance["cem"].fix(1.89e-4)
        m.fs.unit.membrane_areal_resistance["aem"].fix(1.77e-4)
        m.fs.unit.cell_width.fix(0.1)
        m.fs.unit.cell_length.fix(0.79)
        m.fs.unit.membrane_thickness["aem"].fix(1.3e-4)
        m.fs.unit.membrane_thickness["cem"].fix(1.3e-4)
        m.fs.unit.solute_diffusivity_membrane["cem", "Na_+"].fix(1.8e-10)
        m.fs.unit.solute_diffusivity_membrane["aem", "Na_+"].fix(1.25e-10)
        m.fs.unit.solute_diffusivity_membrane["cem", "Cl_-"].fix(1.8e-10)
        m.fs.unit.solute_diffusivity_membrane["aem", "Cl_-"].fix(1.25e-10)
        m.fs.unit.ion_trans_number_membrane["cem", "Na_+"].fix(1)
        m.fs.unit.ion_trans_number_membrane["aem", "Na_+"].fix(0)
        m.fs.unit.ion_trans_number_membrane["cem", "Cl_-"].fix(0)
        m.fs.unit.ion_trans_number_membrane["aem", "Cl_-"].fix(1)
        m.fs.unit.spacer_porosity.fix(1)

        # check ion transfer number requirements
        assert (
            sum(
                value(m.fs.unit.ion_trans_number_membrane["cem", j])
                for j in m.fs.properties.ion_set
            )
            == 1
        )
        assert (
            sum(
                value(m.fs.unit.ion_trans_number_membrane["aem", j])
                for j in m.fs.properties.ion_set
            )
            == 1
        )
        assert sum(
            value(m.fs.unit.ion_trans_number_membrane["cem", j])
            for j in m.fs.properties.cation_set
        ) == sum(
            value(m.fs.unit.ion_trans_number_membrane["aem", j])
            for j in m.fs.properties.anion_set
        )

        # set the inlet stream
        m.fs.unit.inlet_diluate.pressure.fix(101325)
        m.fs.unit.inlet_diluate.temperature.fix(298.15)
        m.fs.unit.inlet_diluate.flow_mol_phase_comp[0, "Liq", "H2O"].fix(2.40e-1)
        m.fs.unit.inlet_diluate.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(7.38e-4)
        m.fs.unit.inlet_diluate.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(7.38e-4)
        m.fs.unit.inlet_concentrate.pressure.fix(101325)
        m.fs.unit.inlet_concentrate.temperature.fix(298.15)
        m.fs.unit.inlet_concentrate.flow_mol_phase_comp[0, "Liq", "H2O"].fix(2.40e-1)
        m.fs.unit.inlet_concentrate.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(7.38e-4)
        m.fs.unit.inlet_concentrate.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(7.38e-4)
        assert degrees_of_freedom(m) == 0

    @pytest.mark.component
    def test_initialization_scaling(self, electrodialysis_1d_cell2):
        m = electrodialysis_1d_cell2
        # set default scaling for state vars
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 5e1, index=("Liq", "H2O")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "Na_+")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "Cl_-")
        )
        # set scaling factors for some vars
        iscale.set_scaling_factor(m.fs.unit.cell_width, 100)
        iscale.set_scaling_factor(m.fs.unit.cell_length, 10)
        iscale.calculate_scaling_factors(m.fs)
        initialization_tester(m)
        badly_scaled_var_values = {
            var.name: val for (var, val) in iscale.badly_scaled_var_generator(m)
        }
        assert not badly_scaled_var_values
        # check to make sure DOF does not change
        assert degrees_of_freedom(m) == 0

    @pytest.mark.component
    def test_solve(self, electrodialysis_1d_cell2):
        m = electrodialysis_1d_cell2
        # run solver and check for optimal solution
        results = solver.solve(m)
        assert_optimal_termination(results)
        badly_scaled_var_values = {
            var.name: val
            for (var, val) in iscale.badly_scaled_var_generator(m, small=1e-9)
        }
        assert not badly_scaled_var_values

    @pytest.mark.component
    def test_solution(self, electrodialysis_1d_cell2):
        m = electrodialysis_1d_cell2

        assert value(
            m.fs.unit.outlet_diluate.flow_mol_phase_comp[0, "Liq", "H2O"]
        ) == pytest.approx(2.304e-1, rel=5e-3)
        assert value(
            m.fs.unit.outlet_diluate.flow_mol_phase_comp[0, "Liq", "Na_+"]
        ) == pytest.approx(1.754e-04, rel=5e-3)
        assert value(
            m.fs.unit.outlet_diluate.flow_mol_phase_comp[0, "Liq", "Cl_-"]
        ) == pytest.approx(1.754e-04, rel=5e-3)
        assert value(
            m.fs.unit.outlet_concentrate.flow_mol_phase_comp[0, "Liq", "H2O"]
        ) == pytest.approx(2.496e-1, rel=5e-3)
        assert value(
            m.fs.unit.outlet_concentrate.flow_mol_phase_comp[0, "Liq", "Na_+"]
        ) == pytest.approx(1.301e-3, rel=5e-3)
        assert value(
            m.fs.unit.outlet_concentrate.flow_mol_phase_comp[0, "Liq", "Cl_-"]
        ) == pytest.approx(1.301e-3, rel=5e-3)

    @pytest.mark.component
    def test_performance_contents(self, electrodialysis_1d_cell2):
        m = electrodialysis_1d_cell2
        perform_dict = m.fs.unit._get_performance_contents()
        assert "vars" in perform_dict
        assert value(
            perform_dict["vars"]["Total electrical power consumption(Watt)"]
        ) == pytest.approx(5.83, rel=5e-3)
        assert value(
            perform_dict["vars"]["Specific electrical power consumption (kW*h/m**3)"]
        ) == pytest.approx(0.390, rel=5e-3)
        assert value(perform_dict["vars"]["Water recovery by mass"]) == pytest.approx(
            0.480, rel=5e-3
        )


class TestElectrodialysis_withNeutralSPecies:
    @pytest.fixture(scope="class")
    def electrodialysis_1d_cell3(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        ion_dict = {
            "solute_list": ["Na_+", "Cl_-", "N"],
            "mw_data": {"H2O": 18e-3, "Na_+": 23e-3, "Cl_-": 35.5e-3, "N": 61.8e-3},
            "elec_mobility_data": {("Liq", "Na_+"): 5.19e-8, ("Liq", "Cl_-"): 7.92e-8},
            "charge": {"Na_+": 1, "Cl_-": -1},
        }
        m.fs.properties = DSPMDEParameterBlock(**ion_dict)
        m.fs.unit = Electrodialysis1D(
            property_package=m.fs.properties,
            operation_mode=ElectricalOperationMode.Constant_Current,
            finite_elements=20,
        )
        return m

    @pytest.mark.unit
    def test_build_model(self, electrodialysis_1d_cell3):
        m = electrodialysis_1d_cell3

        # test configrations
        assert len(m.fs.unit.config) == 19
        assert not m.fs.unit.config.dynamic
        assert not m.fs.unit.config.has_holdup
        assert (
            m.fs.unit.config.operation_mode == ElectricalOperationMode.Constant_Current
        )
        assert m.fs.unit.config.material_balance_type == MaterialBalanceType.useDefault
        assert (
            m.fs.unit.config.momentum_balance_type == MomentumBalanceType.pressureTotal
        )
        assert m.fs.unit.config.property_package is m.fs.properties
        assert "H2O" in m.fs.properties.component_list

        # test all essential params and vars are built
        assert isinstance(m.fs.unit.membrane_set, Set)
        assert isinstance(m.fs.unit.cell_pair_num, Var)
        assert isinstance(m.fs.unit.cell_width, Var)
        assert isinstance(m.fs.unit.cell_length, Var)
        assert isinstance(m.fs.unit.channel_height, Var)
        assert isinstance(m.fs.unit.membrane_thickness, Var)
        assert isinstance(m.fs.unit.solute_diffusivity_membrane, Var)
        assert isinstance(m.fs.unit.ion_trans_number_membrane, Var)
        assert isinstance(m.fs.unit.water_trans_number_membrane, Var)
        assert isinstance(m.fs.unit.water_permeability_membrane, Var)
        assert isinstance(m.fs.unit.membrane_areal_resistance, Var)
        assert isinstance(m.fs.unit.current_applied, Var)
        assert isinstance(m.fs.unit.current_density_x, Var)
        assert isinstance(m.fs.unit.voltage_x, Var)
        assert isinstance(m.fs.unit.current_utilization, Var)
        assert isinstance(m.fs.unit.diluate.power_electrical_x, Var)
        assert isinstance(m.fs.unit.specific_power_electrical, Var)
        assert isinstance(m.fs.unit.current_efficiency_x, Var)

        assert isinstance(m.fs.unit.eq_get_total_areal_resistance_x, Constraint)
        assert isinstance(m.fs.unit.eq_get_current_density, Constraint)
        assert isinstance(m.fs.unit.eq_get_voltage_x, Constraint)
        assert isinstance(m.fs.unit.eq_mass_transfer_term_diluate, Constraint)
        assert isinstance(m.fs.unit.eq_mass_transfer_term_concentrate, Constraint)
        assert isinstance(m.fs.unit.eq_power_electrical, Constraint)
        assert isinstance(m.fs.unit.eq_specific_power_electrical, Constraint)
        assert isinstance(m.fs.unit.eq_current_efficiency_x, Constraint)
        assert isinstance(m.fs.unit.eq_isothermal_diluate, Constraint)
        assert isinstance(m.fs.unit.eq_isothermal_concentrate, Constraint)

    @pytest.mark.unit
    def test_stats_constant_vol(self, electrodialysis_1d_cell3):
        m = electrodialysis_1d_cell3
        assert_units_consistent(m)
        assert degrees_of_freedom(m) == 38
        # Specify a system
        # set the operational parameters
        m.fs.unit.water_trans_number_membrane["cem"].fix(5.8)
        m.fs.unit.water_trans_number_membrane["aem"].fix(4.3)
        m.fs.unit.water_permeability_membrane["cem"].fix(2.16e-14)
        m.fs.unit.water_permeability_membrane["aem"].fix(1.75e-14)
        m.fs.unit.current_applied.fix(8)
        m.fs.unit.electrodes_resistance.fix(0)
        m.fs.unit.cell_pair_num.fix(10)
        m.fs.unit.current_utilization.fix(1)
        m.fs.unit.channel_height.fix(2.7e-4)
        m.fs.unit.membrane_areal_resistance["cem"].fix(1.89e-4)
        m.fs.unit.membrane_areal_resistance["aem"].fix(1.77e-4)
        m.fs.unit.cell_width.fix(0.1)
        m.fs.unit.cell_length.fix(0.79)
        m.fs.unit.membrane_thickness["aem"].fix(1.3e-4)
        m.fs.unit.membrane_thickness["cem"].fix(1.3e-4)
        m.fs.unit.solute_diffusivity_membrane["cem", "Na_+"].fix(1.8e-10)
        m.fs.unit.solute_diffusivity_membrane["aem", "Na_+"].fix(1.25e-10)
        m.fs.unit.solute_diffusivity_membrane["cem", "Cl_-"].fix(1.8e-10)
        m.fs.unit.solute_diffusivity_membrane["aem", "Cl_-"].fix(1.25e-10)
        m.fs.unit.solute_diffusivity_membrane["cem", "N"].fix(1.8e-10)
        m.fs.unit.solute_diffusivity_membrane["aem", "N"].fix(1.25e-10)
        m.fs.unit.ion_trans_number_membrane["cem", "Na_+"].fix(1)
        m.fs.unit.ion_trans_number_membrane["aem", "Na_+"].fix(0)
        m.fs.unit.ion_trans_number_membrane["cem", "Cl_-"].fix(0)
        m.fs.unit.ion_trans_number_membrane["aem", "Cl_-"].fix(1)
        m.fs.unit.spacer_porosity.fix(1)

        # check ion transfer number requirements
        assert (
            sum(
                value(m.fs.unit.ion_trans_number_membrane["cem", j])
                for j in m.fs.properties.ion_set
            )
            == 1
        )
        assert (
            sum(
                value(m.fs.unit.ion_trans_number_membrane["aem", j])
                for j in m.fs.properties.ion_set
            )
            == 1
        )
        assert sum(
            value(m.fs.unit.ion_trans_number_membrane["cem", j])
            for j in m.fs.properties.cation_set
        ) == sum(
            value(m.fs.unit.ion_trans_number_membrane["aem", j])
            for j in m.fs.properties.anion_set
        )

        # set the inlet stream
        m.fs.unit.inlet_diluate.pressure.fix(101325)
        m.fs.unit.inlet_diluate.temperature.fix(298.15)
        m.fs.unit.inlet_diluate.flow_mol_phase_comp[0, "Liq", "H2O"].fix(2.40e-1)
        m.fs.unit.inlet_diluate.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(7.38e-4)
        m.fs.unit.inlet_diluate.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(7.38e-4)
        m.fs.unit.inlet_diluate.flow_mol_phase_comp[0, "Liq", "N"].fix(7.38e-5)
        m.fs.unit.inlet_concentrate.pressure.fix(101325)
        m.fs.unit.inlet_concentrate.temperature.fix(298.15)
        m.fs.unit.inlet_concentrate.flow_mol_phase_comp[0, "Liq", "H2O"].fix(2.40e-1)
        m.fs.unit.inlet_concentrate.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(7.38e-4)
        m.fs.unit.inlet_concentrate.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(7.38e-4)
        m.fs.unit.inlet_concentrate.flow_mol_phase_comp[0, "Liq", "N"].fix(7.38e-5)
        assert degrees_of_freedom(m) == 0

    @pytest.mark.requires_idaes_solver
    @pytest.mark.component
    def test_initialization_scaling(self, electrodialysis_1d_cell3):
        m = electrodialysis_1d_cell3
        # set default scaling for state vars
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 5e1, index=("Liq", "H2O")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "Na_+")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "Cl_-")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e4, index=("Liq", "N")
        )
        # set scaling factors for some vars
        iscale.set_scaling_factor(m.fs.unit.cell_width, 100)
        iscale.set_scaling_factor(m.fs.unit.cell_length, 10)

        iscale.calculate_scaling_factors(m.fs)
        initialization_tester(m)
        badly_scaled_var_values = {
            var.name: val for (var, val) in iscale.badly_scaled_var_generator(m)
        }
        assert not badly_scaled_var_values
        # check to make sure DOF does not change
        assert degrees_of_freedom(m) == 0

    @pytest.mark.requires_idaes_solver
    @pytest.mark.component
    def test_solve(self, electrodialysis_1d_cell3):
        m = electrodialysis_1d_cell3
        # run solver and check for optimal solution
        results = solver.solve(m)
        assert_optimal_termination(results)
        badly_scaled_var_values = {
            var.name: val for (var, val) in iscale.badly_scaled_var_generator(m)
        }
        assert not badly_scaled_var_values

    @pytest.mark.requires_idaes_solver
    @pytest.mark.component
    def test_solution(self, electrodialysis_1d_cell3):
        m = electrodialysis_1d_cell3

        assert value(
            m.fs.unit.outlet_diluate.flow_mol_phase_comp[0, "Liq", "H2O"]
        ) == pytest.approx(2.304e-1, rel=5e-3)
        assert value(
            m.fs.unit.outlet_diluate.flow_mol_phase_comp[0, "Liq", "Na_+"]
        ) == pytest.approx(1.752e-04, rel=5e-3)
        assert value(
            m.fs.unit.outlet_diluate.flow_mol_phase_comp[0, "Liq", "Cl_-"]
        ) == pytest.approx(1.752e-04, rel=5e-3)
        assert value(
            m.fs.unit.outlet_diluate.flow_mol_phase_comp[0, "Liq", "N"]
        ) == pytest.approx(7.264e-05, rel=5e-3)
        assert value(
            m.fs.unit.outlet_concentrate.flow_mol_phase_comp[0, "Liq", "H2O"]
        ) == pytest.approx(2.496e-1, rel=5e-3)
        assert value(
            m.fs.unit.outlet_concentrate.flow_mol_phase_comp[0, "Liq", "Na_+"]
        ) == pytest.approx(1.301e-3, rel=5e-3)
        assert value(
            m.fs.unit.outlet_concentrate.flow_mol_phase_comp[0, "Liq", "Cl_-"]
        ) == pytest.approx(1.301e-3, rel=5e-3)
        assert value(
            m.fs.unit.outlet_concentrate.flow_mol_phase_comp[0, "Liq", "N"]
        ) == pytest.approx(7.496e-05, rel=5e-3)

    @pytest.mark.requires_idaes_solver
    @pytest.mark.component
    def test_performance_contents(self, electrodialysis_1d_cell3):
        m = electrodialysis_1d_cell3
        perform_dict = m.fs.unit._get_performance_contents()
        assert "vars" in perform_dict
        assert value(
            perform_dict["vars"]["Total electrical power consumption(Watt)"]
        ) == pytest.approx(5.837, rel=5e-3)
        assert value(
            perform_dict["vars"]["Specific electrical power consumption (kW*h/m**3)"]
        ) == pytest.approx(0.3896, rel=5e-3)
        assert value(perform_dict["vars"]["Water recovery by mass"]) == pytest.approx(
            0.480, rel=5e-3
        )


class Test_ED_MembNonohm_On_ConstV:
    @pytest.fixture(scope="class")
    def electrodialysis_1d_cell4(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        ion_dict = {
            "solute_list": ["Na_+", "Cl_-"],
            "mw_data": {"H2O": 18e-3, "Na_+": 23e-3, "Cl_-": 35.5e-3},
            "elec_mobility_data": {("Liq", "Na_+"): 5.19e-8, ("Liq", "Cl_-"): 7.92e-8},
            "charge": {"Na_+": 1, "Cl_-": -1},
        }
        m.fs.properties = DSPMDEParameterBlock(**ion_dict)
        m.fs.unit = Electrodialysis1D(
            property_package=m.fs.properties,
            operation_mode=ElectricalOperationMode.Constant_Voltage,
            finite_elements=10,
            has_nonohmic_potential_membrane=True,
        )
        return m

    @pytest.mark.unit
    def test_build_model(self, electrodialysis_1d_cell4):
        m = electrodialysis_1d_cell4
        # test configrations
        assert len(m.fs.unit.config) == 19
        assert not m.fs.unit.config.dynamic
        assert not m.fs.unit.config.has_holdup
        assert (
            m.fs.unit.config.operation_mode == ElectricalOperationMode.Constant_Voltage
        )
        assert m.fs.unit.config.material_balance_type == MaterialBalanceType.useDefault
        assert (
            m.fs.unit.config.momentum_balance_type == MomentumBalanceType.pressureTotal
        )
        assert m.fs.unit.config.property_package is m.fs.properties
        assert "H2O" in m.fs.properties.component_list

        # test all essential params and vars are built
        assert isinstance(m.fs.unit.membrane_set, Set)
        assert isinstance(m.fs.unit.cell_pair_num, Var)
        assert isinstance(m.fs.unit.cell_width, Var)
        assert isinstance(m.fs.unit.cell_length, Var)
        assert isinstance(m.fs.unit.channel_height, Var)
        assert isinstance(m.fs.unit.membrane_thickness, Var)
        assert isinstance(m.fs.unit.solute_diffusivity_membrane, Var)
        assert isinstance(m.fs.unit.ion_trans_number_membrane, Var)
        assert isinstance(m.fs.unit.water_trans_number_membrane, Var)
        assert isinstance(m.fs.unit.water_permeability_membrane, Var)
        assert isinstance(m.fs.unit.membrane_areal_resistance, Var)
        assert isinstance(m.fs.unit.current_density_x, Var)
        assert isinstance(m.fs.unit.voltage_applied, Var)
        assert isinstance(m.fs.unit.voltage_x, Var)
        assert isinstance(m.fs.unit.current_utilization, Var)
        assert isinstance(m.fs.unit.diluate.power_electrical_x, Var)
        assert isinstance(m.fs.unit.specific_power_electrical, Var)
        assert isinstance(m.fs.unit.current_efficiency_x, Var)
        assert isinstance(m.fs.unit.potential_nonohm_membrane_x, Var)
        assert isinstance(m.fs.unit.conc_mem_surf_mol_x, Var)

        assert isinstance(m.fs.unit.eq_get_total_areal_resistance_x, Constraint)
        assert isinstance(m.fs.unit.eq_get_current_density, Constraint)
        assert isinstance(m.fs.unit.eq_get_voltage_x, Constraint)
        assert isinstance(m.fs.unit.eq_mass_transfer_term_diluate, Constraint)
        assert isinstance(m.fs.unit.eq_mass_transfer_term_concentrate, Constraint)
        assert isinstance(m.fs.unit.eq_power_electrical, Constraint)
        assert isinstance(m.fs.unit.eq_specific_power_electrical, Constraint)
        assert isinstance(m.fs.unit.eq_current_efficiency_x, Constraint)
        assert isinstance(m.fs.unit.eq_isothermal_diluate, Constraint)
        assert isinstance(m.fs.unit.eq_isothermal_concentrate, Constraint)
        assert isinstance(m.fs.unit.eq_set_surface_conc, Constraint)
        assert isinstance(m.fs.unit.eq_potential_nonohm_membrane_x, Constraint)

    @pytest.mark.unit
    def test_stats_constant_vol(self, electrodialysis_1d_cell4):
        m = electrodialysis_1d_cell4
        assert_units_consistent(m)
        # Specify a system
        # Note: Testing scenarios in this file are primarily in accord with an experimental
        # setup reported by Campione et al. in Desalination 465 (2019): 79-93.
        # set the operational parameters
        m.fs.unit.water_trans_number_membrane["cem"].fix(5.8)
        m.fs.unit.water_trans_number_membrane["aem"].fix(4.3)
        m.fs.unit.water_permeability_membrane["cem"].fix(2.16e-14)
        m.fs.unit.water_permeability_membrane["aem"].fix(1.75e-14)
        m.fs.unit.voltage_applied.fix(0.5)
        m.fs.unit.electrodes_resistance.fix(0)
        m.fs.unit.cell_pair_num.fix(10)
        m.fs.unit.current_utilization.fix(1)
        m.fs.unit.channel_height.fix(5e-4)
        m.fs.unit.membrane_areal_resistance["cem"].fix(1.89e-4)
        m.fs.unit.membrane_areal_resistance["aem"].fix(1.77e-4)
        m.fs.unit.cell_width.fix(0.1)
        m.fs.unit.cell_length.fix(0.79)
        m.fs.unit.membrane_thickness["aem"].fix(1.3e-4)
        m.fs.unit.membrane_thickness["cem"].fix(1.3e-4)
        m.fs.unit.solute_diffusivity_membrane["cem", "Na_+"].fix(1.8e-10)
        m.fs.unit.solute_diffusivity_membrane["aem", "Na_+"].fix(1.25e-10)
        m.fs.unit.solute_diffusivity_membrane["cem", "Cl_-"].fix(1.8e-10)
        m.fs.unit.solute_diffusivity_membrane["aem", "Cl_-"].fix(1.25e-10)
        m.fs.unit.ion_trans_number_membrane["cem", "Na_+"].fix(1)
        m.fs.unit.ion_trans_number_membrane["aem", "Na_+"].fix(0)
        m.fs.unit.ion_trans_number_membrane["cem", "Cl_-"].fix(0)
        m.fs.unit.ion_trans_number_membrane["aem", "Cl_-"].fix(1)
        m.fs.unit.spacer_porosity.fix(1)

        # check ion transfer number requirements
        assert (
            sum(
                value(m.fs.unit.ion_trans_number_membrane["cem", j])
                for j in m.fs.properties.ion_set
            )
            == 1
        )
        assert (
            sum(
                value(m.fs.unit.ion_trans_number_membrane["aem", j])
                for j in m.fs.properties.ion_set
            )
            == 1
        )
        assert sum(
            value(m.fs.unit.ion_trans_number_membrane["cem", j])
            for j in m.fs.properties.cation_set
        ) == sum(
            value(m.fs.unit.ion_trans_number_membrane["aem", j])
            for j in m.fs.properties.anion_set
        )

        # set the inlet stream
        m.fs.unit.inlet_diluate.pressure.fix(101325)
        m.fs.unit.inlet_diluate.temperature.fix(298.15)
        m.fs.unit.inlet_diluate.flow_mol_phase_comp[0, "Liq", "H2O"].fix(2.40e-1)
        m.fs.unit.inlet_diluate.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(7.38e-4)
        m.fs.unit.inlet_diluate.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(7.38e-4)
        m.fs.unit.inlet_concentrate.pressure.fix(101325)
        m.fs.unit.inlet_concentrate.temperature.fix(298.15)
        m.fs.unit.inlet_concentrate.flow_mol_phase_comp[0, "Liq", "H2O"].fix(2.40e-1)
        m.fs.unit.inlet_concentrate.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(7.38e-4)
        m.fs.unit.inlet_concentrate.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(7.38e-4)

        assert degrees_of_freedom(m) == 0

    @pytest.mark.component
    def test_initialization_scaling(self, electrodialysis_1d_cell4):
        m = electrodialysis_1d_cell4
        # set default scaling for state vars
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e1, index=("Liq", "H2O")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "Na_+")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "Cl_-")
        )
        # set scaling factors for some vars
        iscale.set_scaling_factor(m.fs.unit.cell_width, 10)
        iscale.set_scaling_factor(m.fs.unit.cell_length, 20)
        iscale.set_scaling_factor(m.fs.unit.cell_pair_num, 0.01)
        iscale.calculate_scaling_factors(m.fs)

        # Added this unit check scaling
        assert_units_consistent(m)

        initialization_tester(m)
        badly_scaled_var_values = {
            var.name: val for (var, val) in iscale.badly_scaled_var_generator(m)
        }
        assert not badly_scaled_var_values
        # check to make sure DOF does not change
        assert degrees_of_freedom(m) == 0

    @pytest.mark.component
    def test_solve(self, electrodialysis_1d_cell4):
        m = electrodialysis_1d_cell4
        # run solver and check for optimal solution
        results = solver.solve(m)
        assert_optimal_termination(results)
        badly_scaled_var_values = {
            var.name: val for (var, val) in iscale.badly_scaled_var_generator(m)
        }
        assert not badly_scaled_var_values

    @pytest.mark.component
    def test_solution(self, electrodialysis_1d_cell4):
        m = electrodialysis_1d_cell4

        assert value(
            m.fs.unit.outlet_diluate.flow_mol_phase_comp[0, "Liq", "H2O"]
        ) == pytest.approx(2.364e-1, rel=1e-3)
        assert value(
            m.fs.unit.outlet_diluate.flow_mol_phase_comp[0, "Liq", "Na_+"]
        ) == pytest.approx(5.425e-04, rel=1e-3)
        assert value(
            m.fs.unit.outlet_diluate.flow_mol_phase_comp[0, "Liq", "Cl_-"]
        ) == pytest.approx(5.425e-04, rel=1e-3)
        assert value(
            m.fs.unit.outlet_concentrate.flow_mol_phase_comp[0, "Liq", "H2O"]
        ) == pytest.approx(2.436e-1, rel=1e-3)
        assert value(
            m.fs.unit.outlet_concentrate.flow_mol_phase_comp[0, "Liq", "Na_+"]
        ) == pytest.approx(9.335e-4, rel=1e-3)
        assert value(
            m.fs.unit.outlet_concentrate.flow_mol_phase_comp[0, "Liq", "Cl_-"]
        ) == pytest.approx(9.335e-4, rel=1e-3)

    @pytest.mark.component
    def test_performance_contents(self, electrodialysis_1d_cell4):
        m = electrodialysis_1d_cell4
        perform_dict = m.fs.unit._get_performance_contents()
        assert "vars" in perform_dict
        assert value(
            perform_dict["vars"]["Total electrical power consumption(Watt)"]
        ) == pytest.approx(1.4735, rel=1e-3)
        assert value(
            perform_dict["vars"]["Specific electrical power consumption (kW*h/m**3)"]
        ) == pytest.approx(0.0955, rel=1e-3)
        assert value(perform_dict["vars"]["Water recovery by mass"]) == pytest.approx(
            0.4925, rel=1e-3
        )


class Test_ED_MembNonohm_On_DL_On_ConstV:
    @pytest.fixture(scope="class")
    def electrodialysis_1d_cell5(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        ion_dict = {
            "solute_list": ["Na_+", "Cl_-"],
            "mw_data": {"H2O": 18e-3, "Na_+": 23e-3, "Cl_-": 35.5e-3},
            "elec_mobility_data": {("Liq", "Na_+"): 5.19e-8, ("Liq", "Cl_-"): 7.92e-8},
            "charge": {"Na_+": 1, "Cl_-": -1},
        }
        m.fs.properties = DSPMDEParameterBlock(**ion_dict)
        m.fs.unit = Electrodialysis1D(
            property_package=m.fs.properties,
            operation_mode=ElectricalOperationMode.Constant_Voltage,
            finite_elements=10,
            has_nonohmic_potential_membrane=True,
            has_Nernst_diffusion_layer=True,
            limiting_current_density_data=800,
        )
        return m

    @pytest.mark.unit
    def test_build_model(self, electrodialysis_1d_cell5):
        m = electrodialysis_1d_cell5
        # test configrations
        assert len(m.fs.unit.config) == 19
        assert not m.fs.unit.config.dynamic
        assert not m.fs.unit.config.has_holdup
        assert (
            m.fs.unit.config.operation_mode == ElectricalOperationMode.Constant_Voltage
        )
        assert m.fs.unit.config.material_balance_type == MaterialBalanceType.useDefault
        assert (
            m.fs.unit.config.momentum_balance_type == MomentumBalanceType.pressureTotal
        )
        assert m.fs.unit.config.property_package is m.fs.properties
        assert "H2O" in m.fs.properties.component_list

        # test all essential params and vars are built
        assert isinstance(m.fs.unit.membrane_set, Set)
        assert isinstance(m.fs.unit.cell_pair_num, Var)
        assert isinstance(m.fs.unit.cell_width, Var)
        assert isinstance(m.fs.unit.cell_length, Var)
        assert isinstance(m.fs.unit.channel_height, Var)
        assert isinstance(m.fs.unit.membrane_thickness, Var)
        assert isinstance(m.fs.unit.solute_diffusivity_membrane, Var)
        assert isinstance(m.fs.unit.ion_trans_number_membrane, Var)
        assert isinstance(m.fs.unit.water_trans_number_membrane, Var)
        assert isinstance(m.fs.unit.water_permeability_membrane, Var)
        assert isinstance(m.fs.unit.membrane_areal_resistance, Var)
        assert isinstance(m.fs.unit.current_density_x, Var)
        assert isinstance(m.fs.unit.voltage_applied, Var)
        assert isinstance(m.fs.unit.voltage_x, Var)
        assert isinstance(m.fs.unit.current_utilization, Var)
        assert isinstance(m.fs.unit.diluate.power_electrical_x, Var)
        assert isinstance(m.fs.unit.specific_power_electrical, Var)
        assert isinstance(m.fs.unit.current_efficiency_x, Var)
        assert isinstance(m.fs.unit.potential_nonohm_membrane_x, Var)
        assert isinstance(m.fs.unit.conc_mem_surf_mol_x, Var)
        assert isinstance(m.fs.unit.current_dens_lim_x, Var)
        assert isinstance(m.fs.unit.potential_nonohm_dl_x, Var)
        assert isinstance(m.fs.unit.potential_ohm_dl_x, Var)
        assert isinstance(m.fs.unit.dl_thickness_x, Var)

        assert isinstance(m.fs.unit.eq_get_total_areal_resistance_x, Constraint)
        assert isinstance(m.fs.unit.eq_get_current_density, Constraint)
        assert isinstance(m.fs.unit.eq_get_voltage_x, Constraint)
        assert isinstance(m.fs.unit.eq_mass_transfer_term_diluate, Constraint)
        assert isinstance(m.fs.unit.eq_mass_transfer_term_concentrate, Constraint)
        assert isinstance(m.fs.unit.eq_power_electrical, Constraint)
        assert isinstance(m.fs.unit.eq_specific_power_electrical, Constraint)
        assert isinstance(m.fs.unit.eq_current_efficiency_x, Constraint)
        assert isinstance(m.fs.unit.eq_isothermal_diluate, Constraint)
        assert isinstance(m.fs.unit.eq_isothermal_concentrate, Constraint)
        assert isinstance(m.fs.unit.eq_set_surface_conc, Constraint)
        assert isinstance(m.fs.unit.eq_potential_nonohm_membrane_x, Constraint)
        assert isinstance(m.fs.unit.eq_current_dens_lim_x, Constraint)
        assert isinstance(m.fs.unit.eq_conc_polarization_ratio, Constraint)
        assert isinstance(m.fs.unit.eq_potential_nonohm_dl, Constraint)
        assert isinstance(m.fs.unit.eq_potential_ohm_dl_x, Constraint)
        assert isinstance(m.fs.unit.eq_dl_thickness, Constraint)

    @pytest.mark.unit
    def test_stats_constant_vol(self, electrodialysis_1d_cell5):
        m = electrodialysis_1d_cell5
        assert_units_consistent(m)
        # Specify a system
        # Note: Testing scenarios in this file are primarily in accord with an experimental
        # setup reported by Campione et al. in Desalination 465 (2019): 79-93.
        # set the operational parameters
        m.fs.unit.water_trans_number_membrane["cem"].fix(5.8)
        m.fs.unit.water_trans_number_membrane["aem"].fix(4.3)
        m.fs.unit.water_permeability_membrane["cem"].fix(2.16e-14)
        m.fs.unit.water_permeability_membrane["aem"].fix(1.75e-14)
        m.fs.unit.voltage_applied.fix(0.5)
        m.fs.unit.electrodes_resistance.fix(0)
        m.fs.unit.cell_pair_num.fix(10)
        m.fs.unit.current_utilization.fix(1)
        m.fs.unit.channel_height.fix(5e-4)
        m.fs.unit.membrane_areal_resistance["cem"].fix(1.89e-4)
        m.fs.unit.membrane_areal_resistance["aem"].fix(1.77e-4)
        m.fs.unit.cell_width.fix(0.1)
        m.fs.unit.cell_length.fix(0.79)
        m.fs.unit.membrane_thickness["aem"].fix(1.3e-4)
        m.fs.unit.membrane_thickness["cem"].fix(1.3e-4)
        m.fs.unit.solute_diffusivity_membrane["cem", "Na_+"].fix(1.8e-10)
        m.fs.unit.solute_diffusivity_membrane["aem", "Na_+"].fix(1.25e-10)
        m.fs.unit.solute_diffusivity_membrane["cem", "Cl_-"].fix(1.8e-10)
        m.fs.unit.solute_diffusivity_membrane["aem", "Cl_-"].fix(1.25e-10)
        m.fs.unit.ion_trans_number_membrane["cem", "Na_+"].fix(1)
        m.fs.unit.ion_trans_number_membrane["aem", "Na_+"].fix(0)
        m.fs.unit.ion_trans_number_membrane["cem", "Cl_-"].fix(0)
        m.fs.unit.ion_trans_number_membrane["aem", "Cl_-"].fix(1)
        m.fs.unit.spacer_porosity.fix(1)

        # set the inlet stream
        m.fs.unit.inlet_diluate.pressure.fix(101325)
        m.fs.unit.inlet_diluate.temperature.fix(298.15)
        m.fs.unit.inlet_diluate.flow_mol_phase_comp[0, "Liq", "H2O"].fix(2.40e-1)
        m.fs.unit.inlet_diluate.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(7.38e-4)
        m.fs.unit.inlet_diluate.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(7.38e-4)
        m.fs.unit.inlet_concentrate.pressure.fix(101325)
        m.fs.unit.inlet_concentrate.temperature.fix(298.15)
        m.fs.unit.inlet_concentrate.flow_mol_phase_comp[0, "Liq", "H2O"].fix(2.40e-1)
        m.fs.unit.inlet_concentrate.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(7.38e-4)
        m.fs.unit.inlet_concentrate.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(7.38e-4)

        assert degrees_of_freedom(m) == 0

    @pytest.mark.component
    def test_initialization_scaling(self, electrodialysis_1d_cell5):
        m = electrodialysis_1d_cell5
        # set default scaling for state vars
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e1, index=("Liq", "H2O")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "Na_+")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "Cl_-")
        )
        # set scaling factors for some vars
        iscale.set_scaling_factor(m.fs.unit.cell_width, 10)
        iscale.set_scaling_factor(m.fs.unit.cell_length, 20)
        iscale.set_scaling_factor(m.fs.unit.cell_pair_num, 0.01)
        iscale.calculate_scaling_factors(m.fs)

        # Added this unit check scaling
        assert_units_consistent(m)

        initialization_tester(m)
        badly_scaled_var_values = {
            var.name: val for (var, val) in iscale.badly_scaled_var_generator(m)
        }
        assert not badly_scaled_var_values
        # check to make sure DOF does not change
        assert degrees_of_freedom(m) == 0

    @pytest.mark.component
    def test_solve(self, electrodialysis_1d_cell5):
        m = electrodialysis_1d_cell5
        # run solver and check for optimal solution
        results = solver.solve(m)
        assert_optimal_termination(results)
        badly_scaled_var_values = {
            var.name: val for (var, val) in iscale.badly_scaled_var_generator(m)
        }
        assert not badly_scaled_var_values

    @pytest.mark.component
    def test_solution(self, electrodialysis_1d_cell5):
        m = electrodialysis_1d_cell5

        assert value(
            m.fs.unit.outlet_diluate.flow_mol_phase_comp[0, "Liq", "H2O"]
        ) == pytest.approx(2.3654e-1, rel=1e-3)
        assert value(
            m.fs.unit.outlet_diluate.flow_mol_phase_comp[0, "Liq", "Na_+"]
        ) == pytest.approx(5.722e-04, rel=1e-3)
        assert value(
            m.fs.unit.outlet_diluate.flow_mol_phase_comp[0, "Liq", "Cl_-"]
        ) == pytest.approx(5.722e-04, rel=1e-3)
        assert value(
            m.fs.unit.outlet_concentrate.flow_mol_phase_comp[0, "Liq", "H2O"]
        ) == pytest.approx(2.4347e-1, rel=1e-3)
        assert value(
            m.fs.unit.outlet_concentrate.flow_mol_phase_comp[0, "Liq", "Na_+"]
        ) == pytest.approx(9.038e-4, rel=1e-3)
        assert value(
            m.fs.unit.outlet_concentrate.flow_mol_phase_comp[0, "Liq", "Cl_-"]
        ) == pytest.approx(9.038e-4, rel=1e-3)

    @pytest.mark.component
    def test_performance_contents(self, electrodialysis_1d_cell5):
        m = electrodialysis_1d_cell5
        perform_dict = m.fs.unit._get_performance_contents()
        assert "vars" in perform_dict
        assert value(
            perform_dict["vars"]["Total electrical power consumption(Watt)"]
        ) == pytest.approx(1.3907, rel=1e-3)
        assert value(
            perform_dict["vars"]["Specific electrical power consumption (kW*h/m**3)"]
        ) == pytest.approx(0.0900, rel=1e-3)
        assert value(perform_dict["vars"]["Water recovery by mass"]) == pytest.approx(
            0.4928, rel=1e-3
        )


class Test_ED_MembNonohm_On_DL_On_ConstC:
    @pytest.fixture(scope="class")
    def electrodialysis_1d_cell6(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        ion_dict = {
            "solute_list": ["Na_+", "Cl_-"],
            "mw_data": {"H2O": 18e-3, "Na_+": 23e-3, "Cl_-": 35.5e-3},
            "elec_mobility_data": {("Liq", "Na_+"): 5.19e-8, ("Liq", "Cl_-"): 7.92e-8},
            "charge": {"Na_+": 1, "Cl_-": -1},
        }
        m.fs.properties = DSPMDEParameterBlock(**ion_dict)
        m.fs.unit = Electrodialysis1D(
            property_package=m.fs.properties,
            operation_mode=ElectricalOperationMode.Constant_Current,
            finite_elements=10,
            has_nonohmic_potential_membrane=True,
            has_Nernst_diffusion_layer=True,
            limiting_current_density_data=800,
        )
        return m

    @pytest.mark.unit
    def test_build_model(self, electrodialysis_1d_cell6):
        m = electrodialysis_1d_cell6
        # test configrations
        assert len(m.fs.unit.config) == 19
        assert not m.fs.unit.config.dynamic
        assert not m.fs.unit.config.has_holdup
        assert (
            m.fs.unit.config.operation_mode == ElectricalOperationMode.Constant_Current
        )
        assert m.fs.unit.config.material_balance_type == MaterialBalanceType.useDefault
        assert (
            m.fs.unit.config.momentum_balance_type == MomentumBalanceType.pressureTotal
        )
        assert m.fs.unit.config.property_package is m.fs.properties
        assert "H2O" in m.fs.properties.component_list

        # test all essential params and vars are built
        assert isinstance(m.fs.unit.membrane_set, Set)
        assert isinstance(m.fs.unit.cell_pair_num, Var)
        assert isinstance(m.fs.unit.cell_width, Var)
        assert isinstance(m.fs.unit.cell_length, Var)
        assert isinstance(m.fs.unit.channel_height, Var)
        assert isinstance(m.fs.unit.membrane_thickness, Var)
        assert isinstance(m.fs.unit.solute_diffusivity_membrane, Var)
        assert isinstance(m.fs.unit.ion_trans_number_membrane, Var)
        assert isinstance(m.fs.unit.water_trans_number_membrane, Var)
        assert isinstance(m.fs.unit.water_permeability_membrane, Var)
        assert isinstance(m.fs.unit.membrane_areal_resistance, Var)
        assert isinstance(m.fs.unit.current_density_x, Var)
        assert isinstance(m.fs.unit.current_applied, Var)
        assert isinstance(m.fs.unit.voltage_x, Var)
        assert isinstance(m.fs.unit.current_utilization, Var)
        assert isinstance(m.fs.unit.diluate.power_electrical_x, Var)
        assert isinstance(m.fs.unit.specific_power_electrical, Var)
        assert isinstance(m.fs.unit.current_efficiency_x, Var)
        assert isinstance(m.fs.unit.potential_nonohm_membrane_x, Var)
        assert isinstance(m.fs.unit.conc_mem_surf_mol_x, Var)
        assert isinstance(m.fs.unit.current_dens_lim_x, Var)
        assert isinstance(m.fs.unit.potential_nonohm_dl_x, Var)
        assert isinstance(m.fs.unit.potential_ohm_dl_x, Var)
        assert isinstance(m.fs.unit.dl_thickness_x, Var)

        assert isinstance(m.fs.unit.eq_get_total_areal_resistance_x, Constraint)
        assert isinstance(m.fs.unit.eq_get_current_density, Constraint)
        assert isinstance(m.fs.unit.eq_get_voltage_x, Constraint)
        assert isinstance(m.fs.unit.eq_mass_transfer_term_diluate, Constraint)
        assert isinstance(m.fs.unit.eq_mass_transfer_term_concentrate, Constraint)
        assert isinstance(m.fs.unit.eq_power_electrical, Constraint)
        assert isinstance(m.fs.unit.eq_specific_power_electrical, Constraint)
        assert isinstance(m.fs.unit.eq_current_efficiency_x, Constraint)
        assert isinstance(m.fs.unit.eq_isothermal_diluate, Constraint)
        assert isinstance(m.fs.unit.eq_isothermal_concentrate, Constraint)
        assert isinstance(m.fs.unit.eq_set_surface_conc, Constraint)
        assert isinstance(m.fs.unit.eq_potential_nonohm_membrane_x, Constraint)
        assert isinstance(m.fs.unit.eq_current_dens_lim_x, Constraint)
        assert isinstance(m.fs.unit.eq_conc_polarization_ratio, Constraint)
        assert isinstance(m.fs.unit.eq_potential_nonohm_dl, Constraint)
        assert isinstance(m.fs.unit.eq_potential_ohm_dl_x, Constraint)
        assert isinstance(m.fs.unit.eq_dl_thickness, Constraint)

    @pytest.mark.unit
    def test_stats_constant_vol(self, electrodialysis_1d_cell6):
        m = electrodialysis_1d_cell6
        assert_units_consistent(m)
        # Specify a system
        # Note: Testing scenarios in this file are primarily in accord with an experimental
        # setup reported by Campione et al. in Desalination 465 (2019): 79-93.
        # set the operational parameters
        m.fs.unit.water_trans_number_membrane["cem"].fix(5.8)
        m.fs.unit.water_trans_number_membrane["aem"].fix(4.3)
        m.fs.unit.water_permeability_membrane["cem"].fix(2.16e-14)
        m.fs.unit.water_permeability_membrane["aem"].fix(1.75e-14)
        m.fs.unit.current_applied.fix(8)
        m.fs.unit.electrodes_resistance.fix(0)
        m.fs.unit.cell_pair_num.fix(10)
        m.fs.unit.current_utilization.fix(1)
        m.fs.unit.channel_height.fix(5e-4)
        m.fs.unit.membrane_areal_resistance["cem"].fix(1.89e-4)
        m.fs.unit.membrane_areal_resistance["aem"].fix(1.77e-4)
        m.fs.unit.cell_width.fix(0.1)
        m.fs.unit.cell_length.fix(0.79)
        m.fs.unit.membrane_thickness["aem"].fix(1.3e-4)
        m.fs.unit.membrane_thickness["cem"].fix(1.3e-4)
        m.fs.unit.solute_diffusivity_membrane["cem", "Na_+"].fix(1.8e-10)
        m.fs.unit.solute_diffusivity_membrane["aem", "Na_+"].fix(1.25e-10)
        m.fs.unit.solute_diffusivity_membrane["cem", "Cl_-"].fix(1.8e-10)
        m.fs.unit.solute_diffusivity_membrane["aem", "Cl_-"].fix(1.25e-10)
        m.fs.unit.ion_trans_number_membrane["cem", "Na_+"].fix(1)
        m.fs.unit.ion_trans_number_membrane["aem", "Na_+"].fix(0)
        m.fs.unit.ion_trans_number_membrane["cem", "Cl_-"].fix(0)
        m.fs.unit.ion_trans_number_membrane["aem", "Cl_-"].fix(1)
        m.fs.unit.spacer_porosity.fix(1)

        # set the inlet stream
        m.fs.unit.inlet_diluate.pressure.fix(101325)
        m.fs.unit.inlet_diluate.temperature.fix(298.15)
        m.fs.unit.inlet_diluate.flow_mol_phase_comp[0, "Liq", "H2O"].fix(2.40e-1)
        m.fs.unit.inlet_diluate.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(7.38e-4)
        m.fs.unit.inlet_diluate.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(7.38e-4)
        m.fs.unit.inlet_concentrate.pressure.fix(101325)
        m.fs.unit.inlet_concentrate.temperature.fix(298.15)
        m.fs.unit.inlet_concentrate.flow_mol_phase_comp[0, "Liq", "H2O"].fix(2.40e-1)
        m.fs.unit.inlet_concentrate.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(7.38e-4)
        m.fs.unit.inlet_concentrate.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(7.38e-4)

        assert degrees_of_freedom(m) == 0

    @pytest.mark.component
    def test_initialization_scaling(self, electrodialysis_1d_cell6):
        m = electrodialysis_1d_cell6
        # set default scaling for state vars
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e1, index=("Liq", "H2O")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "Na_+")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "Cl_-")
        )
        # set scaling factors for some vars
        iscale.set_scaling_factor(m.fs.unit.cell_width, 10)
        iscale.set_scaling_factor(m.fs.unit.cell_length, 20)
        iscale.set_scaling_factor(m.fs.unit.cell_pair_num, 0.01)
        iscale.calculate_scaling_factors(m.fs)

        # Added this unit check scaling
        assert_units_consistent(m)

        initialization_tester(m)
        badly_scaled_var_values = {
            var.name: val for (var, val) in iscale.badly_scaled_var_generator(m)
        }
        assert not badly_scaled_var_values
        # check to make sure DOF does not change
        assert degrees_of_freedom(m) == 0

    @pytest.mark.component
    def test_solve(self, electrodialysis_1d_cell6):
        m = electrodialysis_1d_cell6
        # run solver and check for optimal solution
        results = solver.solve(m)
        assert_optimal_termination(results)
        badly_scaled_var_values = {
            var.name: val for (var, val) in iscale.badly_scaled_var_generator(m)
        }
        assert not badly_scaled_var_values

    @pytest.mark.component
    def test_solution(self, electrodialysis_1d_cell6):
        m = electrodialysis_1d_cell6

        assert value(
            m.fs.unit.outlet_diluate.flow_mol_phase_comp[0, "Liq", "H2O"]
        ) == pytest.approx(2.2994e-1, rel=1e-3)
        assert value(
            m.fs.unit.outlet_diluate.flow_mol_phase_comp[0, "Liq", "Na_+"]
        ) == pytest.approx(2.7663e-04, rel=1e-3)
        assert value(
            m.fs.unit.outlet_diluate.flow_mol_phase_comp[0, "Liq", "Cl_-"]
        ) == pytest.approx(2.7663e-04, rel=1e-3)
        assert value(
            m.fs.unit.outlet_concentrate.flow_mol_phase_comp[0, "Liq", "H2O"]
        ) == pytest.approx(2.5006e-1, rel=1e-3)
        assert value(
            m.fs.unit.outlet_concentrate.flow_mol_phase_comp[0, "Liq", "Na_+"]
        ) == pytest.approx(1.1994e-3, rel=1e-3)
        assert value(
            m.fs.unit.outlet_concentrate.flow_mol_phase_comp[0, "Liq", "Cl_-"]
        ) == pytest.approx(1.1994e-3, rel=1e-3)

    @pytest.mark.component
    def test_performance_contents(self, electrodialysis_1d_cell6):
        m = electrodialysis_1d_cell6
        perform_dict = m.fs.unit._get_performance_contents()
        assert "vars" in perform_dict
        assert value(
            perform_dict["vars"]["Total electrical power consumption(Watt)"]
        ) == pytest.approx(12.890, rel=1e-3)
        assert value(
            perform_dict["vars"]["Specific electrical power consumption (kW*h/m**3)"]
        ) == pytest.approx(0.8617, rel=1e-3)
        assert value(perform_dict["vars"]["Water recovery by mass"]) == pytest.approx(
            0.4791, rel=1e-3
        )


class Test_ED_pressure_drop_components:
    @pytest.fixture(scope="class")
    def ed_m0(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        ion_dict = {
            "solute_list": ["Na_+", "Cl_-"],
            "mw_data": {"H2O": 18e-3, "Na_+": 23e-3, "Cl_-": 35.5e-3},
            "elec_mobility_data": {("Liq", "Na_+"): 5.19e-8, ("Liq", "Cl_-"): 7.92e-8},
            "charge": {"Na_+": 1, "Cl_-": -1},
        }
        m.fs.properties = DSPMDEParameterBlock(**ion_dict)
        m.fs.unit = Electrodialysis1D(
            property_package=m.fs.properties,
            operation_mode=ElectricalOperationMode.Constant_Voltage,
            finite_elements=10,
            has_nonohmic_potential_membrane=False,
            has_Nernst_diffusion_layer=False,
            pressure_drop_method=PressureDropMethod.Experimental,
        )
        return m

    @pytest.fixture(scope="class")
    def ed_m1(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        ion_dict = {
            "solute_list": ["Na_+", "Cl_-"],
            "mw_data": {"H2O": 18e-3, "Na_+": 23e-3, "Cl_-": 35.5e-3},
            "elec_mobility_data": {("Liq", "Na_+"): 5.19e-8, ("Liq", "Cl_-"): 7.92e-8},
            "charge": {"Na_+": 1, "Cl_-": -1},
        }
        m.fs.properties = DSPMDEParameterBlock(**ion_dict)
        m.fs.unit = Electrodialysis1D(
            property_package=m.fs.properties,
            operation_mode=ElectricalOperationMode.Constant_Voltage,
            finite_elements=10,
            has_nonohmic_potential_membrane=False,
            has_Nernst_diffusion_layer=False,
            pressure_drop_method=PressureDropMethod.Darcy_Weisbach,
            friction_factor_method=FrictionFactorMethod.Ponzio,
            hydraulic_diameter_method=HydraulicDiameterMethod.conventional,
        )
        return m

    @pytest.fixture(scope="class")
    def ed_m2(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        ion_dict = {
            "solute_list": ["Na_+", "Cl_-"],
            "mw_data": {"H2O": 18e-3, "Na_+": 23e-3, "Cl_-": 35.5e-3},
            "elec_mobility_data": {("Liq", "Na_+"): 5.19e-8, ("Liq", "Cl_-"): 7.92e-8},
            "charge": {"Na_+": 1, "Cl_-": -1},
        }
        m.fs.properties = DSPMDEParameterBlock(**ion_dict)
        m.fs.unit = Electrodialysis1D(
            property_package=m.fs.properties,
            operation_mode=ElectricalOperationMode.Constant_Voltage,
            finite_elements=10,
            has_nonohmic_potential_membrane=False,
            has_Nernst_diffusion_layer=False,
            pressure_drop_method=PressureDropMethod.Darcy_Weisbach,
            friction_factor_method=FrictionFactorMethod.defined,
            hydraulic_diameter_method=HydraulicDiameterMethod.conventional,
        )
        return m

    @pytest.fixture(scope="class")
    def ed_m3(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        ion_dict = {
            "solute_list": ["Na_+", "Cl_-"],
            "mw_data": {"H2O": 18e-3, "Na_+": 23e-3, "Cl_-": 35.5e-3},
            "elec_mobility_data": {("Liq", "Na_+"): 5.19e-8, ("Liq", "Cl_-"): 7.92e-8},
            "charge": {"Na_+": 1, "Cl_-": -1},
        }
        m.fs.properties = DSPMDEParameterBlock(**ion_dict)
        m.fs.unit = Electrodialysis1D(
            property_package=m.fs.properties,
            operation_mode=ElectricalOperationMode.Constant_Voltage,
            finite_elements=10,
            has_nonohmic_potential_membrane=False,
            has_Nernst_diffusion_layer=False,
            pressure_drop_method=PressureDropMethod.Darcy_Weisbach,
            friction_factor_method=FrictionFactorMethod.Gurreri,
            hydraulic_diameter_method=HydraulicDiameterMethod.conventional,
        )
        return m

    @pytest.fixture(scope="class")
    def ed_m4(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        ion_dict = {
            "solute_list": ["Na_+", "Cl_-"],
            "mw_data": {"H2O": 18e-3, "Na_+": 23e-3, "Cl_-": 35.5e-3},
            "elec_mobility_data": {("Liq", "Na_+"): 5.19e-8, ("Liq", "Cl_-"): 7.92e-8},
            "charge": {"Na_+": 1, "Cl_-": -1},
        }
        m.fs.properties = DSPMDEParameterBlock(**ion_dict)
        m.fs.unit = Electrodialysis1D(
            property_package=m.fs.properties,
            operation_mode=ElectricalOperationMode.Constant_Voltage,
            finite_elements=10,
            has_nonohmic_potential_membrane=False,
            has_Nernst_diffusion_layer=False,
            pressure_drop_method=PressureDropMethod.Darcy_Weisbach,
            friction_factor_method=FrictionFactorMethod.Kuroda,
            hydraulic_diameter_method=HydraulicDiameterMethod.conventional,
        )
        return m

    @pytest.fixture(scope="class")
    def ed_m5(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        ion_dict = {
            "solute_list": ["Na_+", "Cl_-"],
            "mw_data": {"H2O": 18e-3, "Na_+": 23e-3, "Cl_-": 35.5e-3},
            "elec_mobility_data": {("Liq", "Na_+"): 5.19e-8, ("Liq", "Cl_-"): 7.92e-8},
            "charge": {"Na_+": 1, "Cl_-": -1},
        }
        m.fs.properties = DSPMDEParameterBlock(**ion_dict)
        m.fs.unit = Electrodialysis1D(
            property_package=m.fs.properties,
            operation_mode=ElectricalOperationMode.Constant_Voltage,
            finite_elements=10,
            has_nonohmic_potential_membrane=False,
            has_Nernst_diffusion_layer=False,
            pressure_drop_method=PressureDropMethod.Darcy_Weisbach,
            friction_factor_method=FrictionFactorMethod.Kuroda,
            hydraulic_diameter_method=HydraulicDiameterMethod.defined,
        )
        return m

    @pytest.fixture(scope="class")
    def ed_m6(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        ion_dict = {
            "solute_list": ["Na_+", "Cl_-"],
            "mw_data": {"H2O": 18e-3, "Na_+": 23e-3, "Cl_-": 35.5e-3},
            "elec_mobility_data": {("Liq", "Na_+"): 5.19e-8, ("Liq", "Cl_-"): 7.92e-8},
            "charge": {"Na_+": 1, "Cl_-": -1},
        }
        m.fs.properties = DSPMDEParameterBlock(**ion_dict)
        m.fs.unit = Electrodialysis1D(
            property_package=m.fs.properties,
            operation_mode=ElectricalOperationMode.Constant_Voltage,
            finite_elements=10,
            has_nonohmic_potential_membrane=False,
            has_Nernst_diffusion_layer=False,
            pressure_drop_method=PressureDropMethod.Darcy_Weisbach,
            friction_factor_method=FrictionFactorMethod.Kuroda,
            hydraulic_diameter_method=HydraulicDiameterMethod.spacer_specific_area_known,
        )
        return m

    @pytest.mark.unit
    def test_deltaP_various_methods(
        self, ed_m0, ed_m1, ed_m2, ed_m3, ed_m4, ed_m5, ed_m6
    ):
        ed_m = (ed_m0, ed_m1, ed_m2, ed_m3, ed_m4, ed_m5, ed_m6)
        for m in ed_m:
            m.fs.unit.inlet_diluate.pressure.fix(101325)
            m.fs.unit.inlet_diluate.temperature.fix(298.15)
            m.fs.unit.inlet_concentrate.pressure.fix(101325)
            m.fs.unit.inlet_concentrate.temperature.fix(298.15)
            m.fs.unit.inlet_diluate.flow_mol_phase_comp[0, "Liq", "H2O"].fix(17.875)
            m.fs.unit.inlet_diluate.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(5.56e-2)
            m.fs.unit.inlet_diluate.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(5.56e-2)
            m.fs.unit.inlet_concentrate.flow_mol_phase_comp[0, "Liq", "H2O"].fix(17.875)
            m.fs.unit.inlet_concentrate.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(
                5.56e-2
            )
            m.fs.unit.inlet_concentrate.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(
                5.56e-2
            )
            m.fs.unit.water_trans_number_membrane["cem"].fix(5.8)
            m.fs.unit.water_trans_number_membrane["aem"].fix(4.3)
            m.fs.unit.water_permeability_membrane["cem"].fix(2.16e-14)
            m.fs.unit.water_permeability_membrane["aem"].fix(1.75e-14)
            m.fs.unit.electrodes_resistance.fix(0)
            m.fs.unit.cell_pair_num.fix(56)
            m.fs.unit.current_utilization.fix(1)
            m.fs.unit.channel_height.fix(7.1e-4)
            m.fs.unit.membrane_areal_resistance["cem"].fix(1.89e-4)
            m.fs.unit.membrane_areal_resistance["aem"].fix(1.77e-4)
            m.fs.unit.cell_width.fix(0.197)
            m.fs.unit.cell_length.fix(1.68)
            m.fs.unit.membrane_thickness["aem"].fix(1.3e-4)
            m.fs.unit.membrane_thickness["cem"].fix(1.3e-4)
            m.fs.unit.solute_diffusivity_membrane["cem", "Na_+"].fix(1.8e-10)
            m.fs.unit.solute_diffusivity_membrane["aem", "Na_+"].fix(1.25e-10)
            m.fs.unit.solute_diffusivity_membrane["cem", "Cl_-"].fix(1.8e-10)
            m.fs.unit.solute_diffusivity_membrane["aem", "Cl_-"].fix(1.25e-10)
            m.fs.unit.ion_trans_number_membrane["cem", "Na_+"].fix(1)
            m.fs.unit.ion_trans_number_membrane["aem", "Na_+"].fix(0)
            m.fs.unit.ion_trans_number_membrane["cem", "Cl_-"].fix(0)
            m.fs.unit.ion_trans_number_membrane["aem", "Cl_-"].fix(1)
            m.fs.unit.spacer_porosity.fix(0.83)
            m.fs.unit.voltage_applied.fix(40)
            m.fs.unit.spacer_porosity.fix(0.83)
            m.fs.properties.set_default_scaling(
                "flow_mol_phase_comp", 0.1, index=("Liq", "H2O")
            )
            m.fs.properties.set_default_scaling(
                "flow_mol_phase_comp", 1e2, index=("Liq", "Na_+")
            )
            m.fs.properties.set_default_scaling(
                "flow_mol_phase_comp", 1e2, index=("Liq", "Cl_-")
            )
            iscale.set_scaling_factor(m.fs.unit.cell_width, 5)
            iscale.set_scaling_factor(m.fs.unit.cell_length, 1)
            iscale.set_scaling_factor(m.fs.unit.cell_pair_num, 0.1)

        # Test ed_m0
        ed_m[0].fs.unit.pressure_drop.fix(40000)
        iscale.calculate_scaling_factors(ed_m[0])
        assert degrees_of_freedom(ed_m[0]) == 0
        initialization_tester(ed_m[0])
        badly_scaled_var_values = {
            var.name: val for (var, val) in iscale.badly_scaled_var_generator(ed_m[0])
        }
        assert not badly_scaled_var_values
        assert value(ed_m[0].fs.unit.pressure_drop_total[0]) == pytest.approx(
            67200, rel=1e-3
        )

        # Test ed_m1
        ed_m[1].fs.unit.general_mass_diffusivity.fix(1.6e-9)
        iscale.calculate_scaling_factors(ed_m[1])
        assert degrees_of_freedom(ed_m[1]) == 0
        initialization_tester(ed_m[1])
        badly_scaled_var_values = {
            var.name: val for (var, val) in iscale.badly_scaled_var_generator(ed_m[1])
        }
        assert not badly_scaled_var_values
        assert value(ed_m[1].fs.unit.pressure_drop[0]) == pytest.approx(
            24640.839, rel=1e-3
        )

        assert value(ed_m[1].fs.unit.pressure_drop_total[0]) == pytest.approx(
            41396.609, rel=1e-3
        )

        # Test ed_m2
        ed_m[2].fs.unit.general_mass_diffusivity.fix(1.6e-9)
        ed_m[2].fs.unit.friction_factor.fix(20)
        iscale.calculate_scaling_factors(ed_m[2])
        assert degrees_of_freedom(ed_m[2]) == 0
        initialization_tester(ed_m[2])
        badly_scaled_var_values = {
            var.name: val for (var, val) in iscale.badly_scaled_var_generator(ed_m[2])
        }
        assert not badly_scaled_var_values
        assert value(ed_m[2].fs.unit.Re) == pytest.approx(58.708, rel=1e-3)

        assert value(ed_m[2].fs.unit.pressure_drop[0]) == pytest.approx(
            21280.815, rel=1e-3
        )

        assert value(ed_m[2].fs.unit.pressure_drop_total[0]) == pytest.approx(
            35751.769, rel=1e-3
        )

        # Test ed_m3
        ed_m[3].fs.unit.general_mass_diffusivity.fix(1.6e-9)
        iscale.calculate_scaling_factors(ed_m[3])
        assert degrees_of_freedom(ed_m[3]) == 0
        initialization_tester(ed_m[3])
        badly_scaled_var_values = {
            var.name: val for (var, val) in iscale.badly_scaled_var_generator(ed_m[3])
        }
        assert not badly_scaled_var_values
        assert value(ed_m[3].fs.unit.Re) == pytest.approx(58.708, rel=1e-3)

        assert value(ed_m[3].fs.unit.pressure_drop[0]) == pytest.approx(
            13670.276, rel=1e-3
        )

        assert value(ed_m[3].fs.unit.pressure_drop_total[0]) == pytest.approx(
            22966.063, rel=1e-3
        )

        # Test ed_m4
        ed_m[4].fs.unit.general_mass_diffusivity.fix(1.6e-9)
        iscale.calculate_scaling_factors(ed_m[4])
        assert degrees_of_freedom(ed_m[4]) == 0
        initialization_tester(ed_m[4])
        badly_scaled_var_values = {
            var.name: val for (var, val) in iscale.badly_scaled_var_generator(ed_m[4])
        }
        assert not badly_scaled_var_values
        assert value(ed_m[4].fs.unit.Re) == pytest.approx(58.708, rel=1e-3)

        assert value(ed_m[4].fs.unit.pressure_drop[0]) == pytest.approx(
            5332.605, rel=1e-3
        )

        assert value(ed_m[4].fs.unit.pressure_drop_total[0]) == pytest.approx(
            8958.776, rel=1e-3
        )

        # Test ed_m5
        ed_m[5].fs.unit.general_mass_diffusivity.fix(1.6e-9)
        ed_m[5].fs.unit.hydraulic_diameter.fix(1.5e-3)
        iscale.calculate_scaling_factors(ed_m[5])
        assert degrees_of_freedom(ed_m[5]) == 0
        initialization_tester(ed_m[5])
        badly_scaled_var_values = {
            var.name: val for (var, val) in iscale.badly_scaled_var_generator(ed_m[5])
        }
        assert not badly_scaled_var_values
        assert value(ed_m[5].fs.unit.Re) == pytest.approx(74.987, rel=1e-3)

        assert value(ed_m[5].fs.unit.pressure_drop[0]) == pytest.approx(
            3694.099, rel=1e-3
        )

        assert value(ed_m[5].fs.unit.pressure_drop_total[0]) == pytest.approx(
            6206.087, rel=1e-3
        )

        # Test ed_m6
        ed_m[6].fs.unit.general_mass_diffusivity.fix(1.6e-9)
        ed_m[6].fs.unit.spacer_specific_area.fix(10700)
        iscale.calculate_scaling_factors(ed_m[6])
        assert degrees_of_freedom(ed_m[6]) == 0
        initialization_tester(ed_m[6])
        iscale.calculate_scaling_factors(ed_m[6])
        badly_scaled_var_values = {
            var.name: val for (var, val) in iscale.badly_scaled_var_generator(ed_m[6])
        }
        assert not badly_scaled_var_values
        assert value(ed_m[6].fs.unit.Re) == pytest.approx(35.801, rel=1e-3)

        assert value(ed_m[6].fs.unit.pressure_drop[0]) == pytest.approx(
            11197.978, rel=1e-3
        )

        assert value(ed_m[6].fs.unit.pressure_drop_total[0]) == pytest.approx(
            18812.603, rel=1e-3
        )
