==========
spectrum-filter
==========

**spectrum-filter** is command-line-interface to compute the effective spectrum of a polychromatic x-ray beam passing through user-selected filters and illuminating a user-selected scintillator crystal.  The main purpose of this code is to provide matplotlib plots to allow researchers to better understand the beam spectrum and the effect of filters on the beam.


Installation
============

::

    $ git clone https://github.com/aps-7bm/spectrum-filter.git
    $ cd spectrum-filter
    $ python setup.py install

in a prepared virtualenv or as root for system-wide installation.

.. warning:: If your python installation is in a location different from #!/usr/bin/env python please edit the first line of the bin/tomopy file to match yours.


Update
======

**spectrum-filter** is constantly updated to include new features. To update your locally installed version::

    $ cd spectrum-filter
    $ git pull
    $ python setup.py install


Dependencies
============

Numpy, Scipy, Matplotlib, and tabulate

Usage
=====

Specifying filter materials and thicknesses
-------------------------------------------

To specify the filter materials, use the `--filter-materials` option.  Give the filters as a
comma-delimited list of symbols in quotation marks.  Set the thicknesses of the filters with
the `filter-thicknesses` option in the same way.  For a list of the available filters, use::
    $ spectrum-filter materials


Model beam hardening in sample
------------------------------

To model the transmission through various thicknesses of a sample material, use the `sample` option.
For example, to model the transmission through aluminum up to 1 mm thick, use::
    $ spectrum-filter sample --sample-material Al --max-sample-thickness 1000.


Model a set of filters used individually
----------------------------------------

To model a set of filter options that will be used individually, use the `filter-set` option.::
    $ spectrum-filter filter-set --filter-matl-list "Be, Cu, Ge" --filter-thick-list "750, 250, 500"


Model a series of filters used together
---------------------------------------

To model a stack of filters used together, use the `filters` option::
    $ spectrum-filter filters --filter-matl-list "Be, Cu, Ge" --filter-thick-list "750, 250, 500"


Configuration File
------------------

Model parameters are stored in **spectrum-filter.conf**. You can create a template with::

    $ spectrum-filter init

**spectrum-filter.conf** is constantly updated to keep track of the last stored parameters, as initalized by **init** or modified by setting a new option value. 


Help
----
To get help, run::
    $ spectrum-filter -h
