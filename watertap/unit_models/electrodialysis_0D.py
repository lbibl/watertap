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

# Import Pyomo libraries
from cmath import inf
from tkinter.messagebox import NO
from xmlrpc.client import Boolean
from attr import mutable
from numpy import integer
from pyomo.environ import (Block,
                           Set,
                           Var,
                           Param,
                           Expression,
                           Suffix,
                           NonNegativeReals,
                           Reference,
                           value,
                           units as pyunits)
from pyomo.common.config import ConfigBlock, ConfigValue, In

# Import IDAES cores
from idaes.core import (ControlVolume0DBlock,
                        declare_process_block_class,
                        MaterialBalanceType,
                        EnergyBalanceType,
                        MomentumBalanceType,
                        UnitModelBlockData,
                        useDefault,
                        MaterialFlowBasis)
from idaes.core.util import get_solver
from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.exceptions import ConfigurationError
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
from sympy import Domain, Integer, Integers

__author__ = "Austin Ladshaw, Xiangyu Bi"

_log = idaeslog.getLogger(__name__)

# Name of the unit model
@declare_process_block_class("Electrodialysis0D")
class Electrodialysis0DData(UnitModelBlockData):
    """
    0D Electrodialysis Model
    """
    # CONFIG are options for the unit model 
    CONFIG = ConfigBlock()

    CONFIG.declare("dynamic", ConfigValue(
        domain=In([False]),
        default=False,
        description="Dynamic model flag - must be False",
        doc="""Indicates whether this model will be dynamic or not,
    **default** = False. The filtration unit does not support dynamic
    behavior, thus this must be False."""))

    CONFIG.declare("has_holdup", ConfigValue(
        default=False,
        domain=In([False]),
        description="Holdup construction flag - must be False",
        doc="""Indicates whether holdup terms should be constructed or not.
    **default** - False. The filtration unit does not have defined volume, thus
    this must be False."""))

    CONFIG.declare("constant_current_operation_mode", ConfigValue(
        default=True,
        domain=Boolean,
        description="The electrical operation mode is constatn current",
    ))

    #TO DO  CONFIG Operational modeb(constant current or voltage)

    CONFIG.declare("material_balance_type", ConfigValue(
        default=MaterialBalanceType.useDefault,
        domain=In(MaterialBalanceType),
        description="Material balance construction flag",
        doc="""Indicates what type of mass balance should be constructed,
    **default** - MaterialBalanceType.useDefault.
    **Valid values:** {
    **MaterialBalanceType.useDefault - refer to property package for default
    balance type
    **MaterialBalanceType.none** - exclude material balances,
    **MaterialBalanceType.componentPhase** - use phase component balances,
    **MaterialBalanceType.componentTotal** - use total component balances,
    **MaterialBalanceType.elementTotal** - use total element balances,
    **MaterialBalanceType.total** - use total material balance.}"""))

    # # TODO: For now, Adam's prop pack does not support and energy balance
    #           so we are making this none for now.
    # # TODO: Temporarily disabling energy balances
    '''
    CONFIG.declare("energy_balance_type", ConfigValue(
        default=EnergyBalanceType.none,
        domain=In(EnergyBalanceType),
        description="Energy balance construction flag",
        doc="""Indicates what type of energy balance should be constructed,
    **default** - EnergyBalanceType.useDefault.
    **Valid values:** {
    **EnergyBalanceType.useDefault - refer to property package for default
    balance type
    **EnergyBalanceType.none** - exclude energy balances,
    **EnergyBalanceType.enthalpyTotal** - single enthalpy balance for material,
    **EnergyBalanceType.enthalpyPhase** - enthalpy balances for each phase,
    **EnergyBalanceType.energyTotal** - single energy balance for material,
    **EnergyBalanceType.energyPhase** - energy balances for each phase.}"""))
    '''

    CONFIG.declare("momentum_balance_type", ConfigValue(
        default=MomentumBalanceType.pressureTotal,
        domain=In(MomentumBalanceType),
        description="Momentum balance construction flag",
        doc="""Indicates what type of momentum balance should be constructed,
    **default** - MomentumBalanceType.pressureTotal.
    **Valid values:** {
    **MomentumBalanceType.none** - exclude momentum balances,
    **MomentumBalanceType.pressureTotal** - single pressure balance for material,
    **MomentumBalanceType.pressurePhase** - pressure balances for each phase,
    **MomentumBalanceType.momentumTotal** - single momentum balance for material,
    **MomentumBalanceType.momentumPhase** - momentum balances for each phase.}"""))

    CONFIG.declare("property_package", ConfigValue(
        default=useDefault,
        domain=is_physical_parameter_block,
        description="Property package to use for control volume",
        doc="""Property parameter object used to define property calculations,
    **default** - useDefault.
    **Valid values:** {
    **useDefault** - use default package from parent model or flowsheet,
    **PhysicalParameterObject** - a PhysicalParameterBlock object.}"""))

    CONFIG.declare("property_package_args", ConfigBlock(
        implicit=True,
        description="Arguments to use for constructing property packages",
        doc="""A ConfigBlock with arguments to be passed to a property block(s)
    and used when constructing these,
    **default** - None.
    **Valid values:** {
    see property package for documentation.}"""))


    def build(self):
        # build always starts by calling super().build()
        # This triggers a lot of boilerplate in the background for you
        super().build()
        # this creates blank scaling factors, which are populated later
        self.scaling_factor = Suffix(direction=Suffix.EXPORT)

        # Next, get the base units of measurement from the property definition
        units_meta = self.config.property_package.get_metadata().get_derived_units
        #create a set as index of ion species
        ion_set=self.config.property_package.solute_set 
      
        # Add unit variables and parameters
        # # TODO: Add material props for membranes and such here
        self.R = Param (
            initialize = 8.3145,
            mutable = False,
            units =  pyunits.joule * pyunits.mole ** -1 * pyunits.kelvin ** -1,
            doc = "ideal gas constant"
        )
        self.T = Param (
            initialize = 298.15,
            mutable = True,
            units = pyunits.kelvin,
            doc = "temperature"
        )

        self.water_density = Param(
            initialize = 1000,
            mutable = False,
            units = pyunits.kg * pyunits.m ** -3,
            doc = "density of water"
        )
        self.water_MW = Param(
            initialize = 18.015e-3,
            mutable = False,
            units = pyunits.kg * pyunits.mole ** -1,
            doc = "molecular weight of water"
        )
    

    
        # electrodialysis cell properties 
        self.cell_width = Var(
            initialize = 1,
            bounds = (1e-3, 1e2),
            units = pyunits.meter,
            doc = 'The width of the electrodialysis cell, denoted as b in the model description') # modified Param to Var
        self.cell_length = Var(
            initialize = 1,
            bounds = (1e-3, 1e2),           
            units = pyunits.meter,
            doc = 'The length of the electrodialysis cell, denoted as l in the model description')
        self.cell_chanel_width = Var(
            initialize = 0.001,
            bounds = (1e-5, 1),
            units = pyunits.meter,
            doc ='The distance between concecutive aem and cem, denoted as s in the m.d.')
        #self.cell_channel_pair_num = Var(
         #   initialize = 1,
          #  domain = Integers,
           # bounds = (0, inf),
            #units = pyunits.dimensionless,
            #doc = 'The number of diluate-concentrate channel pairs in a stack'
        #)

        # membrane-related properties

        self.membrane_set = Set(initialize = ['cem','aem'])  
        
        self.membrane_thickness = Var (
            self.membrane_set, 
            initialize = 0.001,
            bounds = (1e-6, 1e-1),
            units = pyunits.meter,
            doc = 'Membrane thickness')
        
        self.ion_diffusivity_membrane = Var (
            self.membrane_set, ion_set, 
            initialize = 1e-11,
            bounds = (0, 1),
            units = pyunits.meter ** 2 * pyunits.second ** -1,
            doc = 'ion diffusivity in the membrane phase')
        
        self.ion_trans_number_membrane = Var (
            self.membrane_set, ion_set, 
            bounds = (0, 1),
            units = pyunits.dimensionless,
            doc = 'ion diffusivity in the membrane phase')

        self.water_trans_number_membrane = Var (
            self.membrane_set,
            bounds = (0, 1),
            units = pyunits.dimensionless,
            doc = 'transference number of water in membranes'
        )
        self.water_permeability_membrane = Var (
            self.membrane_set,
            initialize = 6,
            units = pyunits.meter * pyunits.second ** -1 * pyunits.pascal ** -1,
            doc = "water permeability coefficient"
        )
        self.vHcoef = Var (
            initialize = 1,
            units = pyunits.dimensionless,
            doc = "van't Hoff coefficient"
        )

        self.osmotic_coef = Var (
            initialize = 1,
            units = pyunits.dimensionless,
            doc = "Osmotic coefficient"
        )
        self.memb_surf_resistence = Var (
            self.membrane_set, 
            initialize = 2e-4,
            bounds = (1e-6,1),
            units = pyunits.ohm * pyunits.meter ** 2,
            doc = 'the surface resistence of membrane')
        #membrane surface resistence is used for constant voltage mode and power evaluation. 
        #self.solution_equiv_conductivity = Var() need ion mobility param. To be used in constant voltage mode. 

        #def current_init (self, i):
         #   if i == 'constant_current':
          #      return 1
           # else:
            #    return self.voltage / ((value(self.memb_surf_resistence[self.membrane_set[1]]))+(value(self.memb_surf_resistence[self.membrane_set[2]])))
        #electrical opertaion properties         
        self.current = Var (
            initialize = 1,
            bounds = (0,100),
            units=pyunits.amp,
            doc = "the current input for the constant current operation mode"
            )
        self.current_utilization = Var(
            initialize = 1,
            bounds = (0, 1),
            units = pyunits.dimensionless,
            doc = "The current utilization efficiency"
        )
        self.faraday_const = Param(
            initialize = 96485,
            mutable = False,
            units = pyunits.coulomb * pyunits.mole ** -1
        )
        # Build control volume for dilute channel
        self.diluate_channel = ControlVolume0DBlock(default={
            "dynamic": False,
            "has_holdup": False,
            "property_package": self.config.property_package,
            "property_package_args": self.config.property_package_args})

        self.diluate_channel.add_state_blocks(
            has_phase_equilibrium=False)

        self.diluate_channel.add_material_balances(
            balance_type=self.config.material_balance_type,
            has_mass_transfer=True)

        # # TODO: Temporarily disabling energy balances
        '''
        self.dilute_side.add_energy_balances(
            balance_type=self.config.energy_balance_type,
            has_enthalpy_transfer=False)
        '''

        self.diluate_channel.add_momentum_balances(
            balance_type=self.config.momentum_balance_type,
            has_pressure_change=False)

        # Build control volume for concentrate channel
        self.concentrate_channel = ControlVolume0DBlock(default={
            "dynamic": False,
            "has_holdup": False,
            "property_package": self.config.property_package,
            "property_package_args": self.config.property_package_args})

        self.concentrate_channel.add_state_blocks(
            has_phase_equilibrium=False)

        self.concentrate_channel.add_material_balances(
            balance_type=self.config.material_balance_type,
            has_mass_transfer=True)

        # # TODO: Temporarily disabling energy balances
        '''
        self.concentrate_side.add_energy_balances(
            balance_type=self.config.energy_balance_type,
            has_enthalpy_transfer=False)
        '''

        self.concentrate_channel.add_momentum_balances(
            balance_type=self.config.momentum_balance_type,
            has_pressure_change=False)

        # Add ports (creates inlets and outlets for each channel)
        self.add_inlet_port(name='inlet_diluate', block=self.diluate_channel)
        self.add_outlet_port(name='outlet_diluate', block=self.diluate_channel)

        self.add_inlet_port(name='inlet_concentrate', block=self.concentrate_channel)
        self.add_outlet_port(name='outlet_concentrate', block=self.concentrate_channel)

        # -------- Add constraints ---------
        # # TODO: Add vars and associated constraints for all flux terms
        #           There will be 1 flux var for water and 1 flux var for all ions
        #           (vars can be indexed by species, so we only write it once)
        #
        #           Those vars will be coupled into the mass_transfer_term below
        #           and will be of opposite sign for each channel
        self.elec_migration_flux = Var(
            self.flowsheet().config.time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            initialize = 1e-9,
            bounds = (1e-18, 1),
            units = pyunits.mole * pyunits.meter ** -2 * pyunits.second ** -1,
            #units_meta('amount')*units_meta('time')**-1*units_meta('length')**-2,
            doc='Molar flux of a component across the membrane driven by electrical migration')

        self.nonelec_flux = Var(
            self.flowsheet().config.time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            initialize = 1e-9,
            bounds = (1e-18, 1),
            units = pyunits.mole * pyunits.meter ** -2 * pyunits.second ** -1,
            #units_meta('amount')*units_meta('time')**-1*units_meta('length')**-2,
            doc='Molar flux of a component across the membrane driven by electrical migration')

        @self.Constraint(self.flowsheet().config.time,
                         self.config.property_package.phase_list,
                         self.config.property_package.component_list,
                         doc="Equation for electrical migration flux") #@xb Deos this need to be repeated for each constraint equa?
        def eq_elec_migration_flux(self, t, p, j):
            if j == 'H2O':
                return self.elec_migration_flux[t, p, j] == (self.water_trans_number_membrane['cem'] + self.water_trans_number_membrane['aem']) \
                    * (self.current / (self.cell_width * self.cell_length) / self.faraday_const) #to add osmositic flux
            else:
                return self.elec_migration_flux[t, p, j] == (self.ion_trans_number_membrane['cem', j]-self.ion_trans_number_membrane['aem', j]) \
                    * (self.current_utilization * self.current / (self.cell_width * self.cell_length)) / (self.config.property_package.charge_comp[j] * self.faraday_const)
         
        @self.Constraint(self.flowsheet().config.time,
                         self.config.property_package.phase_list,
                         self.config.property_package.component_list,
                         doc="Equation for non-electrical flux")

        def eq_nonelec_flux(self, t, p, j):
            if j == 'H2O':
                return self.nonelec_flux[t, p, j] == self.water_density / self.water_MW * self.vHcoef * self.R * self.T * self.osmotic_coef * (self.water_permeability_membrane['cem'] + self.water_permeability_membrane['aem']) \
                    * (sum(self.concentrate_channel.properties_out[t].conc_mol_phase_comp[p, j] for j in ion_set) \
                        - sum(self.diluate_channel.properties_out[t].conc_mol_phase_comp[p, j] for j in ion_set))  
            else:
                return self.nonelec_flux[t, p, j] == - (self.ion_diffusivity_membrane['cem', j] / self.membrane_thickness ['cem'] +self.ion_diffusivity_membrane['aem', j] / self.membrane_thickness ['aem']) \
                     * (self.concentrate_channel.properties_out[t].conc_mol_phase_comp[p, j] - self.diluate_channel.properties_out[t].conc_mol_phase_comp[p, j])

        # # TODO: Summate the flux terms for each mass transfer term in each domain
        # Add constraints for mass transfer terms (diluate_channel)
        @self.Constraint(self.flowsheet().config.time,
                         self.config.property_package.phase_list,
                         self.config.property_package.component_list,
                         doc="Mass transfer term for the diluate channel")
        def eq_mass_transfer_term_diluate(self, t, p, j):
            return self.diluate_channel.mass_transfer_term[t, p, j] == - (self.elec_migration_flux[t, p, j] + self.nonelec_flux[t, p, j]) * (self.cell_width * self.cell_length)
            
        # Add constraints for mass transfer terms (concentrate_channel)
        @self.Constraint(self.flowsheet().config.time,
                         self.config.property_package.phase_list,
                         self.config.property_package.component_list,
                         doc="Mass transfer term for the concentrate channel")
        def eq_mass_transfer_term_concentrate(self, t, p, j):
            return self.diluate_channel.mass_transfer_term[t, p, j] == (self.elec_migration_flux[t, p, j] + self.nonelec_flux[t, p, j]) * (self.cell_width * self.cell_length)
           
    # initialize method
    def initialize(
            blk,
            state_args=None,
            outlvl=idaeslog.NOTSET,
            solver=None,
            optarg=None):
        """
        General wrapper for pressure changer initialization routines

        Keyword Arguments:
            state_args : a dict of arguments to be passed to the property
                         package(s) to provide an initial state for
                         initialization (see documentation of the specific
                         property package) (default = {}).
            outlvl : sets output level of initialization routine
            optarg : solver options dictionary object (default=None)
            solver : str indicating which solver to use during
                     initialization (default = None)

        Returns: None
        """
        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(blk.name, outlvl, tag="unit")
        # Set solver options
        opt = get_solver(solver, optarg)

        # ---------------------------------------------------------------------
        # Initialize diluate_channel block
        flags = blk.diluate_channel.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args,
        )
        init_log.info_high("Initialization Step 1 Complete.")
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # Initialize concentrate_side block
        flags = blk.concentrate_channel.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args,
        )
        init_log.info_high("Initialization Step 2 Complete.")
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # Solve unit
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(blk, tee=slc.tee)
        init_log.info_high(
            "Initialization Step 3 {}.".format(idaeslog.condition(res)))

        # ---------------------------------------------------------------------
        # Release state
        blk.diluate_channel.release_state(flags, outlvl + 1)
        init_log.info(
            "Initialization Complete: {}".format(idaeslog.condition(res))
        )
        blk.concentrate_channel.release_state(flags, outlvl + 1)
        init_log.info(
            "Initialization Complete: {}".format(idaeslog.condition(res))
        )

    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()

        units_meta = self.config.property_package.get_metadata().get_derived_units

        # # TODO: Add scaling factors

    def _get_stream_table_contents(self, time_point=0):
        return create_stream_table_dataframe({"Diluate Channel Inlet": self.inlet_diluate,
                                              "Concentrate Channel Inlet": self.inlet_concentrate,
                                              "Diluate Channel Outlet": self.outlet_diluate,
                                              "Concentrate Channel Outlet": self.outlet_concentrate},
                                                time_point=time_point)
