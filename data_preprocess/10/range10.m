% away 06060029_1574114829.dat
% towards 06050030_1574456837.dat

function [] = range10(filename, fOut)

% Open file.
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

%% Parameters
Bw = 1.5e9;
PRF=512;
c = physconst('LightSpeed'); % Speed of light
frame_start=0.3; %0.3858
frame_stop=20;
%Generate range vector
bin_length = 8 * 1.5e8/23.328e9; % range_decimation_factor * (c/2) / fs.
range_vector = (frame_start-1e-5):bin_length:(frame_stop+1e-5); % +-1e-5 to account for float precision.

for n=1:size(Datastream,2)
        Data=Datastream(:,n);
        iq_mat(:,n) = Data(1:end/2) + 1i*Data(end/2 + 1:end);
end

NTS = size(iq_mat,1); % 128 in ancortek
Np = size(iq_mat,2); % #of pulses 20k in ancortek

Tsweep = 0.001954; % 1e-3 in ancortek
record_length= Np*Tsweep; % length of recording in s
nc=record_length/Tsweep; % number of chirps 20k in ancortek
fprintf('record_length, nc = [%d %d]\n', record_length, nc);

Rmax = c*NTS/(4*Bw); % maximum range
RANGE_FFT_SIZE = NTS*2;

% Data_time=reshape(iq_mat, [NTS nc]);
% tmp = fftshift(fft(Data_time),1);
% Data_range(1:NTS/2,:) = tmp(NTS/2+1:NTS,:);

%% MTI Filter
[b,a]=butter(4, 0.01, 'high'); %  4th order is 24dB/octave slope, 6dB/octave per order of n
                                 % [B,A] = butter(N,Wn, 'high') where N filter order, b (numerator), a (denominator), ...
                                 % highpass, Wn is cutoff freq (half the sample rate)
[h,fl]=freqz(b,a,size(iq_mat,2));
for k=1:size(iq_mat,1)
    Data_RTI_complex_MTIFilt(k,:)=filter(b,a,iq_mat(k,:));
end

bin_indl = 3;
bin_indu = 60;
str_index = 100;
end_index = 500;
Data_range_MTI_selected = Data_RTI_complex_MTIFilt(bin_indl:bin_indu, str_index:end-end_index);

% fs = 23.328e9;
% cutoff_frequency = 10e9; % 截止頻率（以 Hz 為單位）
% filter_order = 4; % 濾波器階數
% design_method = 'butter'; % 使用 Butterworth 濾波器
% lpf = designfilt('lowpass', 'FilterOrder', filter_order, 'CutoffFrequency', cutoff_frequency, 'DesignMethod', design_method);
% Data_range_MTI_selected = filter(lpf, Data_range_MTI_selected);

TimeAxis=(1:size(Data_range_MTI_selected,2))/PRF;
RangeAxis=linspace(0, Rmax, RANGE_FFT_SIZE/2);

%% micro-Doppler Spectrogram
fig = figure('units','normalized','outerposition',[0 0 1 1], 'Visible', 'off');
colormap(jet(256));
imagesc(TimeAxis,RangeAxis,20*log10(abs(Data_range_MTI_selected)));
axis xy
ylim([0 5]);

clim = get(gca,'CLim');
set(gca, 'CLim', clim(2)+[-40,0]);
% xlabel('Time[s]', 'FontSize', 16);
% ylabel('Velocity[m/s]', 'FontSize', 16)
set(gca, 'FontSize', 1)
set(gca,'xtick',[])
set(gca,'ytick',[])
axis tight;
tightfig(gcf);
saveas(gca, fOut)
fclose('all');
close all
end
