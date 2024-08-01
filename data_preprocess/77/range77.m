% away 04060029_1574627273_Raw_0.bin
% towards 04050029_1574626078_Raw_0.bin

function [ ] = range77(fNameIn, picname)
    fileID = fopen(fNameIn, 'r'); % open file
    Data = fread(fileID, 'int16');% DCA1000 should read in two's complement data
    fclose(fileID); % close file
    % Data length: 65536000
    N = length(Data)
    numADCBits = 16; % number of ADC bits per sample
    numLanes = 4; 
    fstart = 77.1799e9; % Start Frequency
    fstop = 77.9474e9; % Stop Frequency
    fc = (fstart+fstop)/2; % Center Frequency
    c = physconst('LightSpeed'); % Speed of light
    lambda = c/fc; % Lambda
    SweepTime = 40e-3; % Time for 1 frame=sweep
    NTS = 256; % Number of time samples per sweep                               
    NPpF = 128; % Number of pulses per frame
    NoF = 500; % Number of frames
    Bw = fstop - fstart; % Bandwidth=767.5e6
    sampleRate = 10e6; % Smpling Rate
    Tc = numADCBits/sampleRate
    dT = SweepTime/NPpF;
    prf = 1/dT; 
    timeAxis = [1:NPpF*NoF]*dT ; % Time Axis
    fprintf(' dT, Tc = [%d %d] \n',dT, Tc);

    % index = 1 : 1 : N;
    % rangeAxis = (index - 1) * (c*Tc*sampleRate) / (2*Bw*N); 

    % reshape and combine real and imaginary parts of complex number
    Data = reshape(Data, numLanes*2, []);
    Data = Data(1:4,:) + sqrt(-1)*Data(5:8,:);                                  
    Data = Data.';
    Np = floor(size(Data(:,1),1)/NTS); % #of pulses
    
    clearvars fileID dataArray ans;

%% IQ Correction
rawData = zeros(NTS,Np,numLanes);
fftRawData = zeros(NTS,Np,numLanes);

for ii = 1:4
    Colmn = floor(length(Data(:,1))/NTS);
    rawData(:,:,ii) = reshape(Data(:,ii),NTS,Colmn);
    fftRawData(:,:,ii) = fftshift(fft(rawData(:,:,ii)),1);
    rp((1:NTS/2),:,ii) = fftRawData(((NTS/2+1):NTS),:,ii); %range profile,color space
end

%% MTI Filter
[m,n]=size(rp(:,:,1));
ns = size(rp,2)+4;
h=[1 -2 3 -2 1]';
rngpro=zeros(m,ns);
for k=1:m
    rngpro(k,:)=conv(h,rp(k,:,1));
end

bin_indl = 3;
bin_indu = 40;
newrngpro = rngpro(bin_indl:bin_indu, :);
fprintf(' size(xy) at location 12 is [%d %d] \n',size(newrngpro, 1), size(newrngpro, 2));


max_distance = 10; % max distance set as 10 m
rangeAxis = linspace(0, max_distance, size(newrngpro, 1)); % Change to distance axis
fig = figure('units', 'normalized', 'outerposition', [0 0 1 1], 'Visible','off');
colormap(jet(256));
imagesc(timeAxis, rangeAxis, 20*log10(abs(newrngpro)));
axis xy
ylim([0 4]);
clim = get(gca,'CLim');
set(gca, 'CLim', clim(2)+[-40,0]);
% xlabel('Time[s]', 'FontSize', 16);
% ylabel('Velocity[m/s]', 'FontSize', 16)
set(gca, 'FontSize', 1)
set(gca,'xtick',[])
set(gca,'ytick',[])
tightfig(gcf);
saveas(gca, picname)
fclose('all');
close all

end