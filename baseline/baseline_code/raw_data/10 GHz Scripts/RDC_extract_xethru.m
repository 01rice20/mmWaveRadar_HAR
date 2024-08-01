function [iq_mat] = RDC_extract_xethru(filename)

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
    
frame_start=0.3; %0.3858
frame_stop=20;

%Generate range vector
bin_length = 8 * 1.5e8/23.328e9; % range_decimation_factor * (c/2) / fs.
range_vector = (frame_start-1e-5):bin_length:(frame_stop+1e-5); % +-1e-5 to account for float precision.

for n=1:size(Datastream,2)
        Data=Datastream(:,n);
        iq_mat(:,n) = Data(1:end/2) + 1i*Data(end/2 + 1:end);
end

close all
end
