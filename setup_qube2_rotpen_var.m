%% Load Model
qube2_rotpen_param_var;
% Set open-loop state-space model of rotary single-inverted pendulum (SIP)
rotpen_ABCD_eqns_ip;
% Display matrices
VarA = A
VarB = B

Q = diag([10 10 0 0])

R = [0.1]

[K,S,P] = lqr(VarA,VarB,Q,R)

Ts = 0.01
[VAd,VBd] = c2d(VarA,VarB,Ts)
Cd= eye(4)
Dd= zeros(4,1)

