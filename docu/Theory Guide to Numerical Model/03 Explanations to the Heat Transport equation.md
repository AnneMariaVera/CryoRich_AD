# Explanations to the Heat Transport equation
  
$$\frac{\partial}{\partial t} \left(  C_{eq}T + \rho_i H_f \theta_i\right)+ \nabla \cdot \left(\mathbf{q} C_\ell T \right)- \nabla \cdot \left( \lambda_{eq}(h_m) \nabla T \right) = 0 \tag{HT}
$$
- $C_{eq}=\rho_lC_l\theta_l+\rho_iC_i\theta_i+\rho_aC_a\theta_a+\rho_sC_s(1-n_e)[\frac{kg}{ms^2K}]$ is the equivalent volumetric heat capacity. The following values are set as default in $Jkg^{-1}°C^{-1}$ 
	- $C_l=4180$ 
	- $C_i=2100$
	- $C_a=1000$
	- $C_s=800$
- $\lambda_{eq} = \lambda_{l}^{\theta_l} \lambda_{i}^{\theta_i} \lambda_{a}^{\theta_a} \lambda_{s}^{1-n_e}][\frac{kg\, m }{s^3K}]$ is the equivalent thermal conductivity
- $H_f\,[\frac{J}{kg}]$ is the latent heat of fusion
**Unit check:** each part has the unit $\frac{kg}{ms^3} = \frac{J}{sm^3}$.

## 1. Derivation
The Heat transport equation is derived from an energy balance, where energy can be transported convectively (advection) and conductively (dispersion). After applying the Gauß-Theorem the energy balance equation is given by 
  
$$
\frac{\partial}{\partial t}\int_V E \,dV+\int_{ V}\nabla\cdot f\,d V = \int_V S \,dV.
$$
  
And again for infinitesimally small $V$ we get 
  
$$
\frac{\partial}{\partial t} E +\nabla\cdot f =  S.
$$
  
The energy is given by: 
  
$$
E = E_{sensible}+E_{latent}=C_{eq}T + \rho_i H_f \theta_i
$$
  
and 
  
$$
f = f_{conv}+f_{cond}=\mathbf{q} C_l T-\lambda_{eq}(h_m) \nabla T. 
$$
  
Assuming that there are no internal sources or sinks this results in 
 
$$
  \frac{\partial}{\partial t} \left(  C_{eq}T + \rho_i H_f \theta_i\right)+ \nabla \cdot \left(\mathbf{q} C_\ell T \right)- \nabla \cdot \left( \lambda_{eq}(h_m) \nabla T \right) = 0 \tag{HT}$$
  
Under the assumption of thermal equilibrium (phase change is instantaneous) we get
  
$$
\frac{\partial}{\partial t} \left(  C_{eq}T + \rho_i H_f \theta_i\right)= C_{eq}\frac{\partial T}{\partial t} + \rho_i H_f \frac{\partial \theta_i}{\partial t}=C_{eq}\frac{\partial T}{\partial t} + \rho_i H_f \frac{\partial \theta_i}{\partial T}\frac{\partial T}{\partial t}=\frac{\partial}{\partial t} \left(\left(  C_{eq} + \rho_i H_f \frac{\partial \theta_i}{\partial T}\right)T\right).
$$
