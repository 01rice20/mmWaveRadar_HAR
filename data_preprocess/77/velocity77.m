% away 04060029_1574627273_Raw_0.bin
% towards 04050029_1574626078_Raw_0.bin

function [ ] = velocity77(fNameIn, picname)
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
    % fprintf("Tc: %f\n", Tc)
    dT = SweepTime/NPpF;
    prf = 1/dT; 
    timeAxis = [1:NPpF*NoF]*dT ; % Time Axis
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

MD.TimeWindowLength = 200;
MD.OverlapFactor = 0.95;
MD.OverlapLength = round(MD.TimeWindowLength*MD.OverlapFactor);
MD.Pad_Factor = 4;
MD.FFTPoints = MD.Pad_Factor*MD.TimeWindowLength;
MD.DopplerBin = prf/(MD.FFTPoints);
MD.DopplerAxis = -prf/2:MD.DopplerBin:prf/2-MD.DopplerBin;
MD.WholeDuration = size(rngpro, 2)/prf;
MD.NumSegments = floor((size(rngpro,2)-MD.TimeWindowLength)/floor(MD.TimeWindowLength*(1-MD.OverlapFactor)));

bin_indl = 3;
bin_indu = 40;

Data_spec_MTI2=0;
Data_spec2=0;
for RBin=bin_indl:1:bin_indu
    Data_MTI_temp = fftshift(spectrogram(rngpro(RBin,:),MD.TimeWindowLength,MD.OverlapLength,MD.FFTPoints),1);
    Data_spec_MTI2=Data_spec_MTI2+abs(Data_MTI_temp);                                
    Data_temp = fftshift(spectrogram(rngpro(RBin,:),MD.TimeWindowLength,MD.OverlapLength,MD.FFTPoints),1);
    Data_spec2=Data_spec2+abs(Data_temp);
end

% fprintf('size(rngpro) is %s\n', mat2str(size(rngpro)))
MD.TimeAxis = linspace(0,MD.WholeDuration, size(Data_spec_MTI2, 2));
Data_spec_MTI2 = flipud(Data_spec_MTI2);


fig = figure('units','normalized','outerposition',[0 0 1 1],'Visible','off');
colormap(jet(256));
imagesc(MD.TimeAxis, MD.DopplerAxis.*c/2/fc, 20*log10(abs(Data_spec_MTI2))); 
axis xy;
ylim([-1 1]);

clim = get(gca,'CLim');
set(gca, 'CLim', clim(2)+[-40,0]);
% xlabel('Time[s]', 'FontSize', 16);
% ylabel('Velocity[m/s]', 'FontSize', 16)
set(gca, 'FontSize', 1)
set(gca,'xtick',[])
set(gca,'ytick',[])
% axis tight;
tightfig(gcf);
saveas(gca, picname)
fclose('all');
close all

end