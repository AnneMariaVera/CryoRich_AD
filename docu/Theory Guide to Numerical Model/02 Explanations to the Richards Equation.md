# Explanations to the Richard's Equation
  
$$
\frac{\partial \theta_l}{\partial t}-\nabla \cdot (k_{rel}K_s\nabla(h_c+z))=0\tag{RE}
$$  

Derived from mass balance equation:   

> change of stored mass + net mass flux = total source/sink of mass   

$$
\frac{\partial}{\partial t}\int_V\rho_l\theta_l\,dV+\int_{\partial V}\rho_l\mathbf{q}\,d\partial V = \int_V\rho_l S_0\,dV.
$$   
With the divergence theorem and infinitesimally small control volumes we end up at the Richard's equation:   
$$\frac{\partial (\rho_l\theta_l)}{\partial t}-\nabla \cdot (\rho_l\mathbf{q})=\rho_lS_0.$$    
Applying t. e chain rule to the divergence and time derivative leads to    
$$
\theta_l\frac{\partial \rho_l}{\partial t}+\rho_l\frac{\partial \theta_l}{\partial t}+\mathbf{q}\cdot\nabla \rho_l+\rho_l\nabla \cdot \mathbf{q}=\rho_lS_0.
$$ 
It is reasonable to assume that the density of water does not vary in space  
$$
\frac{\theta_l}{\rho_l}\frac{\partial \rho_l}{\partial t}+\frac{\partial \theta_l}{\partial t}+\nabla \cdot \mathbf{q}=\rho_lS_0.
$$  
We can now also assume that  
$$
\frac{\theta_l}{\rho_l}\frac{\partial \rho_l}{\partial t}\ll\frac{\partial \theta_l}{\partial t}
$$  
which means that the change i.  water content with time is much larger than the change in the density of water.   
We now also neglect the source/sink term and end up at  
$$
\frac{\partial \theta_l}{\partial t}+\nabla \cdot \mathbf{q}=0\tag{1}
$$
The flux is then given by Darcy's law  
$$
\mathbf{q}=-k_{rel}(\theta_l)\frac{K_0}{\mu}\nabla p_{tot}
$$  
$$
p_{tot}=\rho gz+p_c+p_a+p_h
$$  
where $p_a$ is assumed constant through out the entire domain, $p_h$ is the pressure of the overlying water column (0 in unsaturated zone) and $p_c$ is the capillary pressure (0 in saturated zone).   
Note here that  
$$
p=h\rho g \text{ and } K_s=K_0\frac{\rho g}{\mu}  
$$This way we end up at  
$$
\mathbf{q}=-k_{rel}(\theta_l)\frac{K_0}{\mu}\nabla (\rho gz+p_c)=-k_{rel}(\theta_l)K_s\nabla (z+h_{c})
$$
and plugging the flux back into eq. (1)  
$$
\frac{\partial \theta_l}{\partial t}-\nabla \cdot (k_{rel}K_s\nabla(h_c+z))=0\tag{RE}
$$
Which can also be expressed as  
$$
\frac{\partial \theta_l}{\partial h_c}\frac{\partial h_c}{\partial t}-\nabla \cdot (k_{rel}K_s\nabla(h_c+z))=0\tag{RE}
$$

See [Overview Hydraulic Parameters](Overview%20Hydraulic%20Parameters.md) for explanations on $k_{rel}$ and $\frac{\partial \theta_l}{\partial h_c}$.

