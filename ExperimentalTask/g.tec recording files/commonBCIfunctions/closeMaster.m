fclose all;
close all;

clear
% cgshut
%stop_cogent
clc

reset(RandStream.getGlobalStream,sum(100*clock)); seed_state = rng;