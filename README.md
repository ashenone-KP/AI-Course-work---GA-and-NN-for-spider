# M33174 coursework: GA and neural network for the Webots spider

| Folder | Content |
|---|---|
| [`spider_cw/`](spider_cw/README.md) | **The project.** Part 1: a genetic algorithm that evolves gaits by walking them in Webots physics, with self-collision on. Part 2: a neural network written from scratch in numpy, compared with PyTorch. Start with `spider_cw/README.md`; the design write-ups are in `spider_cw/docs/`. |
| `Spider_world_demo/` | The Webots world and controllers. `ga_trainer` runs the GA live; `gait_player` plays a saved gait. |
| `archive/spider_cw_v1/` | Version 1: a GA scored on a no-physics model of the lecturer's MATLAB spider. Its gaits didn't transfer to Webots, and why is explained in `spider_cw/docs/design.txt`. |

Setup: install Webots R2025a, then run `python -m pip install -r requirements.txt` from this folder.

The module's spec and MATLAB files are not included.
