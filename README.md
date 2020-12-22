[![Build Status](https://travis-ci.com/ebranlard/wiz.svg?branch=master)](https://travis-ci.com/ebranlard/wiz)
<a href="https://www.buymeacoffee.com/hTpOQGl" rel="nofollow"><img alt="Donate just a small amount, buy me a coffee!" src="https://warehouse-camo.cmh1.psfhosted.org/1c939ba1227996b87bb03cf029c14821eab9ad91/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4275792532306d6525323061253230636f666665652d79656c6c6f77677265656e2e737667"></a>

# wiz - Wake and induction zone

Wake and induction zone models for wind farm calculations. 

Please note that this repository is still under active development. 

Four main induction models are implemented:
  - "VC": vortex cylinder (VC) model 
  - "VCFF": vortex cylinder model with a far-field (FF) approximation to speed up calculations
  - "VD": vortex dipole model
  - "SS": self-similar model of Troldborg et al.


## QuickStart
Cloning, getting dependencies, installing local package and running tests:
```bash
git clone https://github.com/ebranlard/wiz
cd wiz
python -m pip install --user -r requirements.txt     # install requirements
python -m pip install -e .                           # install as local package
make                                                 # run tests 
```
If the unittests do not run, check the installation process and post an issue. 

If you wish to use FLORIS, continue with the following commands:
```bash
git submodule update --init
cd floris
python -m pip install --user -r requirements.txt     # install requirements
python -m pip install -e .                           # install as local package
```

## Standalone examples for induction-zone / vorticity models
Found in the following folders:
  - `wiz\examples`: the high level scripts `WindTurbine.py` and `WindFarm.py` provide examples to compute the velocity field about a wind turbine or wind farm. Other scripts in this repository use lower level examples
  - `wiz\article_*`: the scripts in this folder reproduce some plots presented in publications by the authors. They can be run as standalone and are also part of the unittests.
  - `wiz\*.py`: unittests are present in most files.

## Examples with FLORIS coupling
Found in the folder `coupling_examples`


## References and how to cite
The latest paper describing this repository is the 
[following](https://onlinelibrary.wiley.com/doi/full/10.1002/we.2546):
```bibtex
@article{Branlard:2020induction,
    title = {Assessing the blockage effect of wind turbines and wind farms using an analytical vortex model},
    author = {E. Branlard and A. R. {Meyer Forsting}},
    journal = {Wind Energy},
    volume = {23},
    number = {11},
    pages = {2068-2086},
    doi = {https://doi.org/10.1002/we.2546},
    year = "2020",
}
```
Older references are given below:
```bibtex
@INPROCEEDINGS{Branlard:2015induction,
    title     = "Using a cylindrical vortex model to assess the induction zone infront of aligned and yawed rotors",
    author    = "E. Branlard and A. {Meyer Forsting}",
    year      = "2015",
    booktitle = "Proceedings of EWEA Offshore 2015 Conference"
}
@article{branlard:2014right,
    title = {Cylindrical vortex wake model: right cylinder},
    author = {E. Branlard and M. Gaunaa},
    journal = {Wind Energy},
    pages = {1-15},
    year = {2014},
    volume = {524},
    number = {1},
    issn = {10954244, 10991824},
    doi = {10.1002/we.1800}
}
```

## Contributing
Any contributions to this project are welcome! If you find this project useful, you can also buy me a coffee (donate a small amount) with the link below:


<a href="https://www.buymeacoffee.com/hTpOQGl" rel="nofollow"><img alt="Donate just a small amount, buy me a coffee!" src="https://warehouse-camo.cmh1.psfhosted.org/1c939ba1227996b87bb03cf029c14821eab9ad91/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4275792532306d6525323061253230636f666665652d79656c6c6f77677265656e2e737667"></a>
