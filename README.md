# surface-self-diffusion
Simple example code demonstrating how to simulate strongly anisotropic surface
self diffusion and solid-state dewetting.

The code in this repository is a simple, self-contained example of the
simulations described in [1]. This example code demonstrates
the solid-state dewetting of a small rectangular prism to the equilibrium shape
dictated by Nickel-like surface energy anisotropy. The evolution is mediated by
surface-self-diffusion with anisotropic diffusivity. The HDF5 files included in
this repository store look-up tables which encode the anisotropic surface
energy and diffusivity. The surface energy values are based on those given in
[Crystalium](http://crystalium.materialsvirtuallab.org/), as calculated in [2].

## Attribution
If you use the code or techniques presented here, please cite both this Github
repository and [1].

## References
[1] M. A. L'Etoile, C. V. Thompson, W. C. Carter, A level-set method for
simulating solid-state dewetting in systems with strong crystalline anisotropy,
**Finish citation when published** 

[2] R. Tran, Z. Xu, B. Radhakrishnan, D. Winston, W. Sun, K.A. Persson, S.P. Ong, Surface energies of elemental crystals, Sci Data 3 (2016) 160080. https://doi.org/10.1038/sdata.2016.80.

