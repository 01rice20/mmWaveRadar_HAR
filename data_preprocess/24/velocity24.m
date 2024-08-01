% away 01060010_1573837135.dat
% towards 01050010_1573836822.dat

function [ ] = velocity24(fNameIn, picname)

    fileID = fopen(fNameIn, 'r');
    dataArray = textscan(fileID, '%f');
    fclose(fileID);
    radarData = dataArray{1};
    clearvars fileID dataArray ans;
    fc = radarData(1); % Center frequency: 
    % fprintf("fc: %f\n", fc)
    c = physconst('LightSpeed'); % Speed of light
    Tsweep = radarData(2); % Sweep time in ms
    Tsweep=Tsweep/1000; %then in sec
    NTS = radarData(3); % Number of time samples per sweep
    Bw = radarData(4); % FMCW Bandwidth. For FSK, it is frequency step;
    % For CW, it is 0.
    Data = radarData(5:end); % raw data in I+j*Q format
    fs = NTS/Tsweep; % sampling frequency ADC
    record_length = length(Data)/NTS*Tsweep; % length of recording in s
    nc = record_length/Tsweep; % number of chirps
    %% Reshape data into chirps and do range FFT (1st FFT)
    Data_time = reshape(Data, [NTS nc]);
    %Part taken from Ancortek code for FFT and IIR filtering
    tmp = fftshift(fft(Data_time), 1);
    Data_range(1:NTS/2, :) = tmp(NTS/2+1:NTS, :);
    % Data_range = tmp;
    
    % IIR Notch filter
    ns = oddnumber(size(Data_range, 2)) - 1;
    Data_range_MTI = zeros(size(Data_range, 1), ns);
    [b,a] = butter(4, 0.01, 'high');
    % [h, f1] = freqz(b, a, ns);
    for k = 1:size(Data_range, 1)
      Data_range_MTI(k, :) = filter(b, a, Data_range(k, :));
    end
    
    % No MTI
    %Data_range_MTI=Data_range;
    
    %% Spectrogram processing for 2nd FFT to get Doppler
    % This selects the range bins where we want to calculate the spectrogram
    % %Parameters for spectrograms
    MD.PRF = 1/Tsweep;
    MD.TimeWindowLength = 200;
    MD.OverlapFactor = 0.95;
    MD.OverlapLength = round(MD.TimeWindowLength*MD.OverlapFactor);
    MD.Pad_Factor = 4;
    MD.FFTPoints = MD.Pad_Factor*MD.TimeWindowLength;
    MD.DopplerBin=MD.PRF/(MD.FFTPoints);
    MD.DopplerAxis = -MD.PRF/2:MD.DopplerBin:MD.PRF/2 - MD.DopplerBin;
    MD.WholeDuration = size(Data_range_MTI, 2)/MD.PRF;
    MD.NumSegments = floor((size(Data_range_MTI, 2) - MD.TimeWindowLength)/floor(MD.TimeWindowLength*(1-MD.OverlapFactor)));

    bin_indl = 3;
    bin_indu = 64;
    cut_index = 20;

    Data_spec_MTI2=0;
    Data_spec2=0;
    for RBin=bin_indl:1:bin_indu
        Data_MTI_temp = fftshift(spectrogram(Data_range_MTI(RBin, :), MD.TimeWindowLength, MD.OverlapLength, MD.FFTPoints), 1);
        Data_spec_MTI2 = Data_spec_MTI2 + abs(Data_MTI_temp);                                
        Data_temp = fftshift(spectrogram(Data_range(RBin, :), MD.TimeWindowLength, MD.OverlapLength, MD.FFTPoints), 1);
        Data_spec2=Data_spec2 + abs(Data_temp);
    end

    MD.TimeAxis = linspace(0,MD.WholeDuration, size(Data_spec_MTI2(:, cut_index:end), 2));
    Data_spec_MTI2 = flipud(Data_spec_MTI2);
    % fprintf('size(A) is %s\n', mat2str(size(Data_spec_MTI2(:, cut_index:end))))
    fig= figure('visible','off','units','normalized');
    % fig = figure('units','normalized','outerposition',[0 0 1 1],'Visible','off');
    colormap(jet(256)); %xlim([1 9])
    imagesc(MD.TimeAxis, MD.DopplerAxis.*c/2/fc, 20*log10(abs(Data_spec_MTI2(:, cut_index:end)))); 
    axis xy

    ylim([-4 4]);
    
    clim = get(gca,'CLim');
    set(gca, 'CLim', clim(2)+[-40,0]);
    % xlabel('Time[s]', 'FontSize', 16);
    % ylabel('Velocity[m/s]', 'FontSize', 16)
    set(gca,'xtick',[])
    set(gca,'ytick',[])
    set(gca, 'FontSize', 1)
    axis tight;
    tightfig(gcf);
    saveas(gca, picname)
    fclose('all');
    close all
    
    end
    % fprintf('size(A) is %s\n', mat2str(size(A)))