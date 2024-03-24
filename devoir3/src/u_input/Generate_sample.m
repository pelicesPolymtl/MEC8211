%% Create a 2D fiber structure and export it in tiff format to be used in the LBM code
function [d_equivalent]= Generate_sample(seed,filename,mean_d,std_d,poro,nx,dx)
%
%INPUT VARIABLES:
%
% SEED: integer representing the seed for initializing the random
% generator. If seed=0, automatic seed generation. If you want to reproduce
% the same fiber structure, use the same seed (fibers will be located at the same place).
%
% MEAN_D: contains the mean fiber to be used
%
% STD_D: contains the standard deviation of the fiber diameters
%
% PORO: estimated porosity of the fiber structure to be generated
%
% NX: domain lateral size in grid cell
%
% OUTPUT VARIABLE:
%
% D_EQUIVALENT: equivalent diameter to be used to represent the fiber size distribution

% initialize seed for random generator
if (seed==0)
    rng('shuffle');  % random seed
else
    rng(seed);  % for reproducibility
end

dx=dx*1e6; % convertir dx en micron

% Determine distribution of fibers
[nb_fiber,dist_d,poro_eff,d_equivalent]=distribution_of_fiber(mean_d,std_d,poro,nx,dx);

poro=poro_eff;

circle = ones(nb_fiber,3);   %declaring an array to store the shape data (x,y,r)

poremat = zeros(nx,nx);

% positioning fibers
fiber_count = 1; %counter
circle(fiber_count,1) = rand()*nx*dx;   %x-coordinate of centre
circle(fiber_count,2) = rand()*nx*dx;   %y-coordinate of centre
circle(fiber_count,3) = dist_d(fiber_count);  %fiber diameter

%Checking for overlapping with previous fiber (circle)
while (fiber_count < nb_fiber)

    flag = 0;
    di = dist_d(fiber_count);
    xi = rand()*nx*dx;
    yi = rand()*nx*dx;
    %Overlap check in all possible 9 directions for periodicity reason
    for i = 1:fiber_count
        if  (xi - circle(i,1))^2 + (yi - circle(i,2))^2 < (di + circle(i,3))^2 ||...
                (xi - circle(i,1) + nx*dx)^2 + (yi - circle(i,2))^2 < (di + circle(i,3))^2 ||...
                (xi - circle(i,1) - nx*dx)^2 + (yi - circle(i,2))^2 < (di + circle(i,3))^2 ||...
                (xi - circle(i,1))^2 + (yi - circle(i,2) + nx*dx)^2 < (di + circle(i,3))^2 ||...
                (xi - circle(i,1))^2 + (yi - circle(i,2) - nx*dx)^2 < (di + circle(i,3))^2 ||...
                (xi - circle(i,1) + nx*dx)^2 + (yi - circle(i,2) + nx*dx)^2 < (di + circle(i,3))^2 ||...
                (xi - circle(i,1) + nx*dx)^2 + (yi - circle(i,2) - nx*dx)^2 < (di + circle(i,3))^2 ||...
                (xi - circle(i,1) - nx*dx)^2 + (yi - circle(i,2) + nx*dx)^2 < (di + circle(i,3))^2 ||...
                (xi - circle(i,1) - nx*dx)^2 + (yi - circle(i,2) - nx*dx)^2 < (di + circle(i,3))^2

            flag = 1;
            break;
        end
    end

    if(flag == 1)
        continue
    end

    fiber_count = fiber_count + 1;
    circle(fiber_count,1) = xi;
    circle(fiber_count,2) = yi;
    circle(fiber_count,3) = di;
end

%filling in the cell grids
for i = 1:nx
    for j = 1:nx
        px = (0.5 + (i - 1))*dx;
        py = (0.5 + (j - 1))*dx;
        for k = 1:nb_fiber
            %checking if a grid cell belongs to a circle or its periodic image
            if (px - circle(k,1))^2 + (py - circle(k,2))^2 < (circle(k,3)/2)^2 || ...
                    (px - (circle(k,1) + nx*dx))^2 + (py - circle(k,2))^2 < (circle(k,3)/2)^2 ||...
                    (px - (circle(k,1) - nx*dx))^2 + (py - circle(k,2))^2 < (circle(k,3)/2)^2 ||...
                    (px - circle(k,1))^2 + (py - (circle(k,2) - nx*dx))^2  < (circle(k,3)/2)^2 ||...
                    (px - circle(k,1))^2 + (py - (circle(k,2) + nx*dx))^2  < (circle(k,3)/2)^2 ||...
                    (px - (circle(k,1) + nx*dx))^2 + ((py - (circle(k,2) + nx*dx))^2) < (circle(k,3)/2)^2 ||...
                    (px - (circle(k,1) + nx*dx))^2 + ((py - (circle(k,2) - nx*dx))^2) < (circle(k,3)/2)^2 ||...
                    (px - (circle(k,1) - nx*dx))^2 + ((py - (circle(k,2) + nx*dx))^2) < (circle(k,3)/2)^2 ||...
                    (px - (circle(k,1) - nx*dx))^2 + ((py - (circle(k,2) - nx*dx))^2) < (circle(k,3)/2)^2
                %poremat(i,j) = 0;
                poremat(i,j) = 1;

                break
            end
        end
    end
end

number_of_fibres=fiber_count;

imwrite(logical(poremat),filename,'tiff');

data=imread(filename);
imshow(data);

dx=dx*1e-6; % reconvertir dx en m

end

function [nb_fiber,dist_d,poro_eff,d_equivalent]=distribution_of_fiber(mean_d,std_d,poro,nx,dx)

dist=normrnd(mean_d,std_d,[1,10000]);
nb_fiber=1;
poro_eff=1-sum(dist(1:nb_fiber).^2/4*pi)/(nx*dx)^2;

while poro_eff >= poro
    poro_eff_old=poro_eff;
    nb_fiber=nb_fiber+1;
    poro_eff=1-sum(dist(1:nb_fiber).^2/4*pi)/(nx*dx)^2;
end
if (abs(poro_eff-poro)>abs(poro_eff_old-poro))
    nb_fiber=nb_fiber-1;
    poro_eff=poro_eff_old;
end

dist_d=sort(dist(1:nb_fiber),'descend');

d_equivalent=(sum(dist_d.^2)/sum(dist_d));

end
