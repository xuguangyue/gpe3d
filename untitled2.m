clear
clc

[x,y,z,den]=load3d('./real3d-den');
% [x,y,z,den]=load3d('./N=6000_omg=1.7/data/real3d-den');
[x2,y2,z2]=meshgrid(y, z, x);

xslice = 0;                               % define the cross sections to view
yslice = -18;
zslice = 0;

figure
slice(x2, y2 ,z2, den, xslice, yslice, zslice)    % display the slices
ylim([-40 0])
view(-34,24)
shading interp

[fo,vo] = isosurface(x2,y2,z2,den,0.0001);               % isosurface for the outside of the volume
[fe,ve,ce] = isocaps(x2,y2,z2,den,0.0001);               % isocaps for the end caps of the volume

figure
p1 = patch('Faces', fo, 'Vertices', vo);       % draw the outside of the volume
p1.FaceColor = 'red';
p1.EdgeColor = 'none';

p2 = patch('Faces', fe, 'Vertices', ve, ...    % draw the end caps of the volume
   'FaceVertexCData', ce);
p2.FaceColor = 'interp';
p2.EdgeColor = 'none';

view(-40,24)
daspect([1 1 1])                             % set the axes aspect ratio
colormap(gray(100))
box on

camlight(40,40)                                % create two lights 
camlight(-20,-10)
lighting gouraud
