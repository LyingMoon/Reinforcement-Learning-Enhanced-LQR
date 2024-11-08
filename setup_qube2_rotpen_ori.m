%% Load Model
qube2_rotpen_param;
% Set open-loop state-space model of rotary single-inverted pendulum (SIP)
rotpen_ABCD_eqns_ip;
% Display matrices
A
B

Q = diag([10 10 0 0])

R = [0.1]

[K,S,P] = lqr(A,B,Q,R)

Ts = 0.01
[Ad,Bd] = c2d(A,B,Ts)
Cd= eye(4)
Dd= zeros(4,1)

