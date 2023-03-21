%%LID DRIVEN CAVITY FLOW EXAMPLE
%%The code is based on the SIMPLE algorithm

clc
clear all
close all
format longG

%% GRID SIZE AND OTHER PARAMETERS
%i runs along x-direction and j runs along y-direction 

imax=30;                        %grid size in x-direction 
jmax=30;                        %grid size in y-direction 
max_iteration=60000000; 
maxRes = 1000;
iteration = 1;
mu = 0.01;                      %viscosity
rho = 1;                        %density
% velocity= 1;                     %velocity = lid velocity
% velocity= 1.5;                     %velocity = lid velocity
% velocity= 2;                     %velocity = lid velocity
% velocity= 2.5;                     %velocity = lid velocity
% velocity= 3;                     %velocity = lid velocity
% velocity= 3.5;                     %velocity = lid velocity
% velocity=  4;                     %velocity = lid velocity
% velocity= 4.5;                     %velocity = lid velocity
% velocity= 5;                     %velocity = lid velocity
velocity= 5.5;                     %velocity = lid velocity
% velocity= 6;                     %velocity = lid velocity
% velocity= 6.5;                     %velocity = lid velocity
% velocity= 7;                     %velocity = lid velocity
% velocity= 7.5;                     %velocity = lid velocity
% velocity= 8;                     %velocity = lid velocity
% velocity= 8.5;                     %velocity = lid velocity
% velocity= 9;                     %velocity = lid velocity
% velocity= 9.5;                     %velocity = lid velocity
% velocity= 10;                     %velocity = lid velocity

dx=1/(imax-1);					%dx,dy cell sizes along x and y directions
dy=1/(jmax-1); 
x=dx/2:dx:1-dx/2; 
y=0:dy:1; 
alphaP = 0.0008;                   %pressure under-relaxation
alphaU = .994;                   %velocity under-relaxation

tol = 1e-5;

%   u_star, v_star are Intermediate velocities
%   u and v = Final velocities

%	Flow Reynolds number
Re = rho*velocity*1/mu;

%Variable declaration
p   = zeros(imax,jmax);             %   p = Pressure
p_star   = zeros(imax,jmax);        
p_prime = zeros(imax,jmax);         %   pressure correction 
rhsp = zeros(imax,jmax);            %   Right hand side vector of pressure correction equation
divergence = zeros(imax,jmax); 

%Vertical velocity
v_star = zeros(imax,jmax+1);
vold   = zeros(imax,jmax+1);
vRes   = zeros(imax,jmax+1);
v      = zeros(imax,jmax+1);
d_v    = zeros(imax,jmax+1);    %velocity orrection coefficient

% Horizontal Velocity -----------
u_star = zeros(imax+1,jmax);
uold   = zeros(imax+1,jmax);
uRes   = zeros(imax+1,jmax);
u      = zeros(imax+1,jmax);
d_u    = zeros(imax+1,jmax);  %velocity orrection coefficient

%Boundary condition 
%Lid velocity (Top wall is moving with 1m/s)
u_star(1:imax+1,jmax)=velocity;
u(1:imax+1,jmax)=velocity;

%% ---------- iterations -------------------
while ( (iteration <= max_iteration) && (maxRes > tol) ) 
    iteration = iteration + 1;
    [u_star,d_u] = u_momentum(imax,jmax,dx,dy,rho,mu,u,v,p_star,velocity,alphaU);       %%Solve u-momentum equation for intermediate velocity u_star 
    [v_star,d_v] = v_momentum(imax,jmax,dx,dy,rho,mu,u,v,p_star,alphaU);                 %%Solve v-momentum equation for intermediate velocity v_star
    uold = u;
    vold = v; 
    [rhsp] = get_rhs(imax,jmax,dx,dy,rho,u_star,v_star);                                 %%Calculate rhs vector of the Pressure Poisson matrix 
    [Ap] = get_coeff_mat_modified(imax,jmax,dx,dy,rho,d_u,d_v);                          %%Form the Pressure Poisson coefficient matrix 
    [p,p_prime] = pres_correct(imax,jmax,rhsp,Ap,p_star,alphaP);                         %%Solve pressure correction implicitly and update pressure
    [u,v] = updateVelocity(imax,jmax,u_star,v_star,p_prime,d_u,d_v,velocity);            %%Update velocity based on pressure correction
    [divergence]=checkDivergenceFree(imax,jmax,dx,dy,u,v);                               %%check if velocity field is divergence free
    p_star = p;                                                                          %%use p as p_star for the next iteration
    
    %find maximum residual in the domain
    vRes = abs(v - vold);
    uRes = abs(u - uold);
    maxRes_u = max(max(uRes));
    maxRes_v = max(max(vRes));
    maxRes = max(maxRes_u, maxRes_v);
    It(1,iteration) = iteration;
    Res(1,iteration) = maxRes;                                                                       %%Check for convergence 
    disp(['It = ',int2str(iteration),'; Res = ',num2str(maxRes)])
    if (maxRes > 2)
        disp('not going to converge!');
        break;
    end

%     write out variables at each time step to use for
%     ML-Open-Lid-Driven-Flow
    p_star_3d(:,:,iteration)  = p(1:30,:); 
    u_star_3d(:,:,iteration)  = u(1:30,:);
    v_star_3d(:,:,iteration)  = v(1:30,:);
    p_prime_3d(:,:,iteration) = p_prime(1:30,:);
    
end

%   Declare arrays for post processing steps
psi = zeros(imax,jmax);
w = zeros(imax,jmax);

%   Calculating Vorticity
for i=2 : imax-1
    for j = 2:jmax-1
        uy(i,j) = ( u(i,j) - u(i,j-1))/( y(j) - y(j-1));
        vx(i,j) = ( v(i,j) - v(i-1,j))/(x(i) - x(i-1));
        w       = vx - uy;
    end
end

%   Calculating the streamlines
for i = 2:imax-1
    for j = 2:jmax-1
        psi(i,j) = u(i,j)*( y(j+1) - y(j) + psi(i,j-1));
    end
end



%% plot


disp(['Total Iterations = ',int2str(iteration)])

% figure 
% contourf(x,y,u(2:imax,:)',50, 'edgecolor','none');colormap jet
% colorbar;
% axis([0 1 0 1]); 
% title('steady Ux'); 

% U VELOCITY CONTOUR PLOTS
FigHandle_01 = figure('Position', [100, 150, 390, 290]);
contourf(x,y,u(2:imax,:)',50, 'edgecolor','none');
colormap jet
colorbar;
hold on
contour(x,x,u(2:imax,2:imax)',8,'edgecolor','g');
axis([0 1 0 1]);
title(sprintf('U Velocity plot RE = %d',Re))
saveas(FigHandle_01,sprintf('U Velocity plot RE_%d',Re),'png');

%% STREAMLINE CONTOUR PLOT
FigHandle_02 = figure('Position', [100, 150, 390, 290]);
contourf(x,y,psi(2:imax,:)',10, 'edgecolor','w');
colormap jet
colorbar;
hold on
contour(x,y,psi(2:imax,:)',10,'edgecolor','r');
axis([0 1 0 1]);
title(sprintf('Streamline plot RE = %d',Re))
saveas(FigHandle_02,sprintf('Streamline plot RE_%d',Re),'png');

%% RESIDUAL PLOT
FigHandle_03 = figure('Position', [100, 150, 390, 290]);
semilogy(It,Res)
%axis([0 1 0 1]);
title(sprintf('Residual plot RE = %d',Re))
saveas(FigHandle_03,sprintf('Residual plot RE_%d',Re),'png');


%% VORTICITY CONTOUR PLOT
FigHandle_04 = figure('Position', [100, 150, 390, 290]);
contourf(x,y(:,1:imax-1),w(1:imax-1,:)',100, 'edgecolor','none');
colormap jet
colorbar;
hold on
contour(x,y(:,1:imax-1),w(1:imax-1,:)',35,'edgecolor','k');
axis([0 1 0 1]);
title(sprintf('Vorticity plot RE = %d',Re))
saveas(FigHandle_04,sprintf('Vorticity plot RE_%d',Re),'png');

FigHandle_02 = figure('Position', [100, 150, 390, 290]);
filename = sprintf('U_vel_Re_%d.gif',Re);
for n = 2:500:size(u_star_3d,3)
    contourf(x,y,squeeze(u_star_3d(2:imax,:,n))', 20);
    colormap jet
    colorbar;
%     hold on
%     contour(x,x,squeeze(u_star_3d(2:imax,2:imax,n))',20,'edgecolor','k');
%     hold off
    axis([0 1 0 1]);
    title(sprintf('U Velocity plot RE = %d at iter= %d',Re, n))
    drawnow
    frame = getframe(1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    if n == 2;
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf, "DelayTime", 0.001);
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append', "DelayTime", 0.001);
    end
end

save(sprintf('re_%d_base_aP_0p001_aU_0p994',Re), '-v7.3')