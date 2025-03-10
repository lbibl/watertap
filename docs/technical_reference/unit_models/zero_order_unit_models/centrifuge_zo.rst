Centrifuge (ZO)
===============

Model Type
----------
This unit model is formulated as a **single-input, double-output** model form.
See documentation for :ref:`single-input, double-output Helper Methods<sido_methods>`.

Electricity Consumption
-----------------------
Electricity consumption is calculated using the **constant_intensity** helper function.
See documentation for :ref:`Helper Methods for Electricity Demand<electricity_methods>`.

Costing Method
--------------
Costing is calculated using the **cost_centrifuge** method in the zero-order costing package.
See documentation for the :ref:`zero-order costing package<zero_order_costing>`.

Additional Variables
--------------------

.. csv-table::
   :header: "Description", "Variable Name", "Units"

   "Dosing rate of polymer", "polymer_dose", ":math:`mg/l`"
   "Consumption rate of polymer", "polymer_demand", ":math:`kg/hr`"

Additional Constraints
----------------------

.. csv-table::
   :header: "Description", "Constraint Name"

   "Polymer demand constraint", "polymer_demand_equation"

.. index::
   pair: watertap.unit_models.zero_order.centrifuge_zo;centrifuge_zo

.. currentmodule:: watertap.unit_models.zero_order.centrifuge_zo

Class Documentation
-------------------

.. automodule:: watertap.unit_models.zero_order.centrifuge_zo
    :members:
    :noindex:
