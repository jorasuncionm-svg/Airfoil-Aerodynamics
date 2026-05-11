# 2D Airfoil Aerodynamic Analysis & Inverse Design - Singularity Panel Method 

This repository contains a professional Python implementation of a **2D Singularity Panel Method Solver** developed for potential flow analysis and inverse aerodynamic design. The project models arbitrary 2D profiles (including sharp triangular shapes and curved **NACA 4-digit** geometries) by distributing constant-strength sources ($q_j$) to capture thickness effects and a global uniform vortex strength ($\gamma$) to model lift.

This project was developed within the 3rd-year Aerodynamics curriculum at the *Universidad Politécnica de Madrid (ETSIAE)*.

---

##  Key Features

* **Advanced Singularity Modeling:** Combines constant-source distributions per panel with a global vortex sheet to compute lift, pressure distributions ($C_p$), and pitching moments.
* **Kutta Condition Integration:** Solves the physical circulation around the profile by enforcing equal velocities at the upper and lower trailing edge panels.
* **NACA 4-Digit Generator:** Custom script utilizing cosine distribution clustering to define the geometry of curved airfoils (such as the **NACA 3316**) with high resolution at the leading and trailing edges.
* **Convergence Study Tool:** Automates convergence loops to locate the mathematically optimal number of panels ($N_{opt}$).
* **Inverse Design Optimizer:** Includes a parametric sweep tool based on thin airfoil theory (TPL) to reshape profiles to target aerodynamic criteria.
* **Viscosity Coupling & Verification:** Connects Python predictions with **XFOIL** data to analyze viscous boundary layer effects.

---

##  Physical & Mathematical Background

### 1. No-Penetration Boundary Condition
The potential solver imposes that the flow cannot penetrate the solid boundary at any control point $i$:

$$\sum_{j=1}^{N} A_{ij} \lambda_j + A_{i\Gamma} \gamma = - \vec{V}_{\infty} \cdot \vec{n}_i$$

Where $A_{ij}$ represents the aerodynamic influence coefficients mapping normal velocities induced by sources and vortex elements on the panels.

### 2. Kutta Condition
To close the system with $N+1$ equations, we enforce the physical trailing edge flow behavior:
* **For a basic 3-panel test profile (Triangular):** $\gamma_1 + \gamma_3 = 0$.
* **For N-panels:** Velocities on the upper and lower surfaces are matched at the trailing edge.

---

##  Performance Validation (NACA 3316 at $\alpha = 2.8^\circ$)

Our custom Python solver was strictly validated against inviscid **XFOIL** simulations using $N = 100$ panels:

| Aerodynamic Metric | Python Solver (Ours) | XFOIL (Inviscid Reference) | Relative Error (%) |
| :--- | :---: | :---: | :---: |
| **Lift Coefficient ($C_l$)** | `0.7112` | `0.7153` | **0.57 %** |
| **Pitching Moment ($C_{m, c/4}$)** | `-0.0731` | `-0.0767` | **4.69 %** |

### Numerical Convergence Study
A sweep starting from 40 panels with 20% increases proved that the convergence criterion ($\epsilon_{Cl} < 1.0\%$) was reached at **$N_{opt} = 84$ panels** ($\epsilon = 0.52\%$):

<p align="center">
  <img src="images/convergence_study.png" width="550" alt="Convergence of Cl vs Panel Count">
</p>

---

##  Engineering Challenge: Inverse Design

The goal was to modify the base **NACA 3316** ($f=3.0\%, x_f=30\%, t=16\%$) to achieve a lift increment of **$\Delta C_l = +0.2$ at $\alpha = 0^\circ$** while strictly minimizing pitching moment penalties ($\Delta C_m \approx 0$).

Using **Thin Airfoil Theory (TPL)** to steer a smart parametric sweep, the optimizer outputted the following geometry:
* **Base Profile:** NACA 3316
* **Optimized Profile:** **NACA 4718** ($f=4.7\%, x_f=20\%, t=18\%$)
* **Aerodynamic Gain:** Achieved **$\Delta C_l = +0.1918$** with a negligible moment penalty of **$\Delta C_{m(c/4)} = 0.0164$**!

**Aerodynamic justification:** By shifting the maximum camber location forward ($x_f = 20\%$) and thickening the profile slightly, the suction peak is compressed closer to the quarter-chord point. This reshapes the pressure field to increase lift while keeping the aerodynamic moment in check.
