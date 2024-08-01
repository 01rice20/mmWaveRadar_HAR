clc;clear;close all;

data = ['/mnt/Fantom/Micro_Doppler Activity data/Xethrue Data/05_Walking Towards the radar/*.dat']';
out = '/mnt/Fantom/Micro_Doppler Activity data/twards_no_MTI/';
if ~exist(out, 'dir')
       mkdir(out)
end
files = dir(data);

for i = 1:length(files)
      disp([int2str(i) '/' int2str(length(files))]);
      fname = [files(i).folder '/' files(i).name];
      fout = [out files(i).name(1:end-4) '.png'];
      RDC = RDC_extract_xethru(fname);
      RDC2MD_xethru(RDC, fout);  
        
end





