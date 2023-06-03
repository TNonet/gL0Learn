`fit` function
----------------

.. autofunction:: gL0Learn.fit

FitModels
---------
.. autoclass:: gL0Learn.fitmodel.FitModel
   :members:


`scoring` Functions
-------------------
These functions can be used in combinaiton with :py:class:`gL0Learn.fitmodel.FitModel`


.. autofunction:: gL0Learn.fit


Generating Functions
--------------------
.. autofunction:: gL0Learn.synthetic.generate_synthetic
.. autofunction:: gL0Learn.synthetic.preprocess
.. autofunction:: gL0Learn.synthetic.generate_regression
.. autofunction:: gL0Learn.synthetic.generate_banded_partial_correlation
.. autofunction:: gL0Learn.synthetic.generate_toeplitz_correlation
.. autofunction:: gL0Learn.synthetic.generate_constant_correlation
.. autofunction:: gL0Learn.synthetic.generate_independent

Advanced Details
----------------

For those that have access to a MOSEK liscense and/or interesting in exploring gL0Learn's performance relative to a provable optimal solution you can utilize the tools provided here:

.. autofunction:: gL0Learn.opt.MIO_mosek
.. autofunction:: gL0Learn.opt.mosek_level_values


Penalty Models
--------------
These models are used when setting regularization levels from  :py:meth:`gL0Learn.fit``
.. autofunction:: gL0Learn.Penalty

Bounds Models
-------------
These models are used when setting bounds from  :py:meth:`gL0Learn.fit``
.. autofunction:: gL0Learn.Bounds

Oracle Models
-------------
These models are used in c setting bounds from  :py:meth:`gL0Learn.fit``
.. autofunction:: gL0Learn.Oracle
