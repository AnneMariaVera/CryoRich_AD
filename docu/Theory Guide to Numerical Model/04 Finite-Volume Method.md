# General Derivation 
Assume the governing equation is given by 
  
$$
\frac{\partial m}{\partial t}+\nabla \cdot q=0\tag{RE}
$$
We now look at a single cell $i$ with size $A=\Delta x\Delta z$

$$
\int_A\frac{\partial m}{\partial t}\,dA+\int_A\nabla \cdot q\,dA=0
$$
  
$$
\Leftrightarrow\frac{\partial}{\partial t}\int_Am\,dA+\int_{\partial A}n \cdot q\,d\partial A=0
$$
and for $m$ constant inside the cell and $q$ constant over each face of the boundary we get
  
$$
\Leftrightarrow\frac{\partial m}{\partial t}A+(\Delta z(q_x^l-q_x^r)+\Delta x(q_z^t-q_z^b))=0
$$
Approximating the time derivative with finite differences we end up with 
$$
\Leftrightarrow\frac{ m^{t+1}-m^t}{\Delta t}\Delta z\overline{\Delta x}+CN(\Delta z(q_x^{l,t+1}-q_x^{r,t+1})+\Delta x(q_z^{t,t+1}-q_z^{b,t+1})) + 
$$
$$
+(1-CN)(\Delta z(q_x^{l,t}-q_x^{r,t})+\Delta x(q_z^{t,t}-q_z^{b,t}))=0
$$
and multiplying by $\frac{\Delta t}{\Delta z\overline{\Delta x}}$
$$
\Leftrightarrow m^{t+1}-m^t+CN(\frac{\Delta t}{\overline{\Delta x}}(-q_x^{l,t+1}+q_x^{r,t+1})+\frac{\Delta t\Delta x}{\Delta z\overline{\Delta x}}(q_z^{t,t+1}-q_z^{b,t+1})) + 
$$
$$
+(1-CN)(\frac{\Delta t}{\overline{\Delta x}}(-q_x^{l,t}+q_x^{r,t})+\frac{\Delta t\Delta x}{\Delta z\overline{\Delta x}}(q_z^{t,t}-q_z^{b,t}))=0
$$

# Richards' Equation

For the Richards' equation $m=\theta_l$ and $q = -k_{rel}K_s\nabla(h_c+z)$, then we get (now for CN=1)

$$
\theta_l^{t+1}-\theta_l^t+CN\frac{\Delta t}{\overline{\Delta x}}\left(k_{rel}^{il,t+1}K_s\frac{h_c^{*,t+1}-h_c^{l,t+1}}{\overline{\Delta x}}-k_{rel}^{ir,t+1}K_s\frac{h_c^{r,t+1}-h_c^{*,t+1}}{\overline{\Delta x}}\right)+
$$
$$
+\frac{\Delta t\Delta x}{\Delta z\overline{\Delta x}}\left(k_{rel}^{it,t+1}K_s\frac{h_c^{t,t+1}-h_c^{*,t+1}}{\Delta z}-k_{rel}^{ib,t+1}K_s\frac{h_c^{*,t+1}-h_c^{b,t+1}}{\Delta z}\right)=0 
$$

### Boundary Conditions
Notation remark: $[-1,:]$ is the indexing of matrices in python.  -1 is the index of the top row, as matrices in python are index bottomup. 
**Top**: $q_z[-1,:]=q_{inf}$
**Left, Right, Bottom**: no flow boundary
# Heat Transport
To make the notation easier to read we separate the residual into different parts:
  
$$
R_{storage}=m^{t+1}-m^t
$$ 
  
$$
R_{conv,h}= CN\frac{\Delta t}{\overline{\Delta x}}\left(-q_{conv,x}^{l,t+1}+q_{conv,x}^{r,t+1}\right)+(1-CN)\frac{\Delta t}{\overline{\Delta x}}\left(-q_{conv,x}^{l,t}+q_{conv,x}^{r,t}\right) 
$$
  
$$
R_{conv,v}= CN\frac{\Delta t\Delta x}{\Delta z\overline{\Delta x}}\left(-q_{conv,z}^{b,t+1}+q_{conv,z}^{t,t+1}\right)+(1-CN)\frac{\Delta t\Delta x}{\Delta z\overline{\Delta x}}\left(-q_{conv,z}^{b,t}+q_{conv,z}^{t,t}\right) 
$$
  
$$
R_{cond,h}= CN\frac{\Delta t}{\overline{\Delta x}}\left(-q_{cond,x}^{l,t+1}+q_{cond,x}^{r,t+1}\right)+(1-CN)\frac{\Delta t}{\overline{\Delta x}}\left(-q_{cond,x}^{l,t}+q_{cond,x}^{r,t}\right) 
$$
  
$$
R_{cond,v}= CN\frac{\Delta t\Delta x}{\Delta z\overline{\Delta x}}\left(-q_{cond,z}^{b,t+1}+q_{cond,z}^{t,t+1}\right)+(1-CN)\frac{\Delta t\Delta x}{\Delta z\overline{\Delta x}}\left(-q_{cond,z}^{b,t}+q_{cond,z}^{t,t}\right) 
$$
  
where 
  
$$
m = C_{eff}T+\rho_i H_f \theta_i
$$
  
$$
q_{conv} = \mathbf{q} C_l T 
$$
  
$$
q_{cond} = -\lambda_{eq}(h_m) \nabla T
$$
  
### Boundary Conditions
**Top**: $T[-1,:]=T_{surface}$, implemented through convective and conductive BC  
**Bottom, Left, Right**: $q_{cond}=0$ and $q_{conv}=0$
