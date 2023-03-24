function save_3D_matrix_as_gif(filename, matrix, delaytime)

if(nargin<2 || nargin>3)
    error('incorrect number of input arguments')
end

if nargin==2
    delaytime = 0.1;
end

% adjust matrix to have entries between 1 and 256
% first make range between 0 and 1
u_star_3d = u_star_3d - min(min(min(u_star_3d)));
u_star_3d = u_star_3d/(max(max(max(u_star_3d))));
% adjust range to be between 1 and 256
u_star_3d = u_star_3d*255 + 1;
%     [X,Y] = meshgrid(x,y);
% the first slice is all zeros and imwrite doesn't like that
FigHandle_02 = figure('Position', [100, 150, 390, 290]);
contourf(x,y,squeeze(u_star_3d(2:imax,:,end))',10);
colormap jet
colorbar;
hold on
contour(x,x,squeeze(u_star_3d(2:imax,2:imax,end))',10,'edgecolor','g');
axis([0 1 0 1]);
title(sprintf('U Velocity plot RE = %d at iter= %d',Re, iteration))


imwrite(uint8(255 * mat2gray(c)),filename,'gif', 'WriteMode','overwrite','DelayTime',delaytime,'LoopCount',Inf);
for ii = 3:size(u_star_3d,3)
    c = contourf(X,Y,squeeze(u_star_3d(2:imax,:,ii)'));
    imwrite(uint8(255 * mat2gray(c)),filename,'gif', 'WriteMode','append','DelayTime',delaytime);
end

end

FigHandle_02 = figure('Position', [100, 150, 390, 290]);
filename = sprintf('U_vel_Re_%d.gif',Re);
for n = 2:10:size(u_star_3d,3)
    contourf(x,y,squeeze(u_star_3d(2:imax,:,n))',10);
    colormap jet
    colorbar;
%     hold on
%     contour(x,x,squeeze(u_star_3d(2:imax,2:imax,n))',10,'edgecolor','g');
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