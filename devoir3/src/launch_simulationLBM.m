%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% launch_simulationLBM.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code pour MEC8211
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Auteurs :
%     - M. Valid
%     - Justin BELZILE
%     -
%     -
% Date : 20/03/2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MATLAB script to launch a fiber structure generation and the corresponding LBM simulation
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear all; close all;

% Same seed as team #
seed=7;

% Fiber diameter data
mean_fiber_d= 12.5 ; % in microns
std_d= 2.85 ; % in microns

% Porosity data
poro= 0.9 ;
poro_std = 7.5e-3;

% Pressure loss
deltaP= 0.1 ; % pressure drop in Pa

% Grid
%NX= 100 ;
%dx= 2e-6 ; % grid size in m
NX = [50 100 200 400];
dx = 2e-6*100./NX;

filename = {'fiber_mat_50.tiff','fiber_mat_100.tiff','fiber_mat_200.tiff','fiber_mat_400.tiff'} ;

%% Lognormal
%% Create a lognormal distribution object
%pd = makedist('Lognormal', 'mu', 0, 'sigma', 1);
%
%% Generate a data sample
%sample_data = random(pd, 1000, 1); % Generate 1000 samples
%
%% Calculate parameters from the sample
%sample_mean = mean(sample_data);
%sample_std_dev = std(sample_data);

% Loop for each dx and NX
for i=1:4
    %fprintf("i = : %d",i);
    % generation of the fiber structure
    [d_equivalent]=Generate_sample(seed,filename{i},mean_fiber_d,std_d,poro,NX(i),dx(i));

    % calculation of the flow field and the permeability from Darcy Law
    LBM(filename{i},NX(i),deltaP,dx(i),d_equivalent)

end
