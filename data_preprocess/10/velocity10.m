% towards 06050030_1574456837.dat
% away 06060029_1574114829.dat

function [] = velocity10(filename, fOut)
    fid = fopen(filename, 'rb');
    ctr = 0;
    while(1)

        ContentId=fread(fid,1,'uint32');
        Info=fread(fid,1,'uint32');
    
        ctr=ctr+1; 
        Length=fread(fid,1,'uint32');
        Data=fread(fid,182,'float');
        
        if feof(fid)
            break
        end
            
        Datastream(:,ctr)=Data;
        data_length(ctr)=Length;

    end 

for n=1:size(Datastream,2)
        Data=Datastream(:,n);
        iq_mat(:,n) = Data(1:end/2) + 1i*Data(end/2 + 1:end);
end

% MTI Filter
[b,a]=butter(4, 0.01, 'high'); %  4th order is 24dB/octave slope, 6dB/octave per order of n
                                 % [B,A] = butter(N,Wn, 'high') where N filter order, b (numerator), a (denominator), ...
                                 % highpass, Wn is cutoff freq (half the sample rate)
[h,fl]=freqz(b,a,size(iq_mat,2));
for k=1:size(iq_mat,1)
    Data_RTI_complex_MTIFilt(k,:)=filter(b,a,iq_mat(k,:));
end

%% Parameters
Bw = 1.5e9;
NTS = size(iq_mat,1); % 128 in ancortek
Np = size(iq_mat,2); % # of pulses 20k in ancortek
Tsweep = 0.001954; % 1e-3 in ancortek
fc = 8.748e9; % Center frequency, 2.5e10 in ancortek
c = physconst('LightSpeed'); % Speed of light
record_length= Np*Tsweep; % length of recording in s
fs=NTS/Tsweep; % sampling frequency ADC - 128000 in ancortek
nc=record_length/Tsweep; % number of chirps 20k in ancortek

MD.PRF = 1/Tsweep;
MD.TimeWindowLength = 200;
MD.OverlapFactor = 0.95;
MD.OverlapLength = round(MD.TimeWindowLength*MD.OverlapFactor);
MD.Pad_Factor = 4;
MD.FFTPoints = MD.Pad_Factor*MD.TimeWindowLength;
MD.DopplerBin=MD.PRF/(MD.FFTPoints);

MD.WholeDuration = size(Data_RTI_complex_MTIFilt, 2)/MD.PRF;
MD.NumSegments = floor((size(Data_RTI_complex_MTIFilt, 2) - MD.TimeWindowLength)/floor(MD.TimeWindowLength*(1-MD.OverlapFactor)));

bin_indl = 3;
bin_indu = 64;

Data_spec_MTI2=0;
for RBin=bin_indl:1:bin_indu
    Data_MTI_temp = fftshift(spectrogram(Data_RTI_complex_MTIFilt(RBin, :), MD.TimeWindowLength, MD.OverlapLength, MD.FFTPoints), 1);
    Data_spec_MTI2 = Data_spec_MTI2 + abs(Data_MTI_temp);                                
end
% Data_spec_MTI2 = flipud(Data_spec_MTI2);

MD.TimeAxis = linspace(0,MD.WholeDuration, size(Data_spec_MTI2, 2));
MD.DopplerAxis = -MD.PRF/2:MD.DopplerBin:MD.PRF/2 - MD.DopplerBin;

%% micro-Doppler Spectrogram
fig = figure('units','normalized','outerposition',[0 0 1 1],'Visible','off');
colormap(jet(256));
imagesc(MD.TimeAxis, MD.DopplerAxis.*c/2/fc, 20*log10(abs(Data_spec_MTI2))); 
axis xy;
ylim([-4 4]);

clim = get(gca,'CLim');
set(gca, 'CLim', clim(2)+[-40,0]);
% xlabel('Time[s]', 'FontSize', 16);
% ylabel('Velocity[m/s]', 'FontSize', 16)
set(gca, 'FontSize', 1)
set(gca,'xtick',[])
set(gca,'ytick',[])
% axis tight;
tightfig(gcf);
saveas(gca, fOut)
fclose('all');
close all

end