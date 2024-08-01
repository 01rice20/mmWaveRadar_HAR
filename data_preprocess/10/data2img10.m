% towards 06050030_1574456837.dat
% away 06060029_1574114829.dat

function [] = data2img10(filename, fOut)
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
        
% frame_start=0.3; %0.3858
% frame_stop=20;

% %Generate range vector
% bin_length = 8 * 1.5e8/23.328e9; % range_decimation_factor * (c/2) / fs.
% range_vector = (frame_start-1e-5):bin_length:(frame_stop+1e-5); % +-1e-5 to account for float precision.

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

PRF=512;
TimeAxis=(1:size(iq_mat,2))/PRF;
DopplerAxis=linspace(-PRF/2,PRF/2,size(iq_mat,2));

nfft = 2^12;window = 150;noverlap = 130;shift = window - noverlap; 
sx = myspecgramnew_10(sum(Data_RTI_complex_MTIFilt(4:90,:)),window,nfft,shift); %18 front, 27 corner
sx1 = fftshift(sx,1);
sx_scaled = sx1(1408:3688,:);
sx2 = abs(sx1);

%% micro-Doppler Spectrogram
fig = figure('visible','off');
colormap(jet(256));  
imagesc(TimeAxis, DopplerAxis, 20*log10(sx2./max(max(sx2))));

axis xy

caxis([-35 0])
axis([0 max(TimeAxis) -PRF/2 PRF/2])
set(gca,'xtick',[],'ytick',[])

%% Save
frame = frame2im(getframe(gca));
imwrite(frame, fOut);

close all
fclose('all');
end