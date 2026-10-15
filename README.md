GD-PINNs — Implementation Repository

This repository provides a self-contained reference implementation of the GD-PINNs framework, including its core model architectures, feature extraction procedures, manifold construction, transfer strategies, loss formulations, training routines, and evaluation utilities.

Repository Overview

To facilitate the understanding, verification, and adaptation of GD-PINNs, the implementation has been distilled from the original research codebase into a more accessible and unified structure. The original project was developed for large-scale seawater reverse osmosis (SWRO) membrane applications and involved multiple task-specific modules, data-processing procedures, and computational components. Directly releasing this highly distributed project structure would make the underlying methodological logic difficult to inspect and reproduce independently.

Therefore, the present repository reorganizes the essential algorithms into a general-purpose GD-PINNs reference implementation. It preserves the principal computational workflow and methodological components described in the paper while simplifying the engineering-specific organization and dependencies. The objective is to provide a transparent and reusable implementation template rather than an exact replica of the original industrial project environment.

Code Overview

The repository consists of two primary components that illustrate the preprocessing procedures and the GD-PINNs framework.

1. Interpolation and Preprocessing Module

The first component provides an independent demonstration of the interpolation procedure used to reconstruct unstructured finite-element solutions onto a unified Cartesian grid.

It illustrates the transformation of physical fields from irregular computational meshes into structured tensors suitable for spectral analysis and neural-network processing. The implementation includes coordinate nondimensionalization based on a characteristic length scale, Delaunay-based barycentric interpolation as a practical approximation to natural-neighbor interpolation, and topology-preserving coarse-grid extraction using fixed spatial strides.

This module is intended primarily to clarify the preprocessing methodology and facilitate adaptation to different data formats. Users may adjust the interpolation resolution, interpolation scheme, and coarse-grid retention ratio according to the geometric characteristics and numerical requirements of their datasets.

2. GD-PINNs Core Framework

The second component implements the principal methodological pipeline of GD-PINNs, integrating spectral-physical representation, graph-based manifold construction, adaptive anchor selection, operator-geometry transfer, and physics-informed neural-network training.

The implementation includes the following components:

SpectralPhysicalExtractor:** Constructs compact 150-dimensional manifold coordinates using FFT, power spectral density (PSD), and proper orthogonal decomposition (POD) descriptors extracted from reconstructed Cartesian tensors.

GraphManifoldConstructor:** Establishes a k-nearest-neighbor graph in the solution manifold, employs graph attention to characterize anisotropic diffusion, and performs adaptive anchor support set expansion with covariance-based convergence assessment.

OperatorGeometryTransfer:** Implements the decoupled hot-start transfer strategy, including strict one-nearest-neighbor parameter inheritance for the encoder and graph-weighted interpolation for the adapter.

GD_PINNs_Net:** Defines the FiLM-conditioned neural architecture for incorporating geometry-dependent information into physics-informed solution learning.

Training and Evaluation:** Implements the associated physics-informed loss formulations, hybrid Adam and L-BFGS-B optimization, and topology-preserving coarse-grid conditioning through voxel-grid filtering, together with supporting evaluation procedures.

These components are organized to expose the essential algorithmic relationships and facilitate independent examination of the GD-PINNs workflow. While the original project contains additional application-specific engineering modules, the reference implementation concentrates on the generalizable methodology.

Adapting to New Datasets

The reference implementation follows the methodological configuration used in the study. When applying GD-PINNs to new PDE families, geometries, or physical datasets, certain parameters may require adjustment.

The interpolation resolution and coarse-grid retention ratio should be selected according to the spectral complexity and manifold curvature of the target problem, as discussed in Section 4.4 of the paper.

The spectral-physical representation may require adjustments to the FFT low-frequency cutoff, PSD aggregation strategy, or POD energy-retention threshold, depending on the dominant structures of the solution family.

Graph construction parameters, including neighborhood size and attention heads, may be tuned according to the sampling density and continuity of the solution manifold. Similarly, the network architecture, configured with a hidden width of 100 neurons in the provided implementation, may be adapted to the complexity of the target PDE and the available computational resources.

The repository is intended to serve as a methodological reference and a starting point for further research, rather than a configuration that is universally optimal across different physical systems.

Data Availability and Reproducibility

The complete datasets used in the study are not publicly hosted in this repository for two reasons. First, certain data originate from industrial SWRO membrane applications and are subject to confidentiality and usage restrictions. Second, the full datasets are computationally and spatially extensive, making them unsuitable for distribution through a standard source-code repository.

To support methodological transparency, representative small-scale example datasets are included to illustrate the expected data structures, physical quantities, and interaction between preprocessing and model components. These examples support inspection and execution of the reference workflow but do not replace the full-scale datasets required to reproduce all quantitative results reported in the paper.

The complete research datasets may be requested from the corresponding author, subject to applicable confidentiality agreements, data-sharing restrictions, and availability.
