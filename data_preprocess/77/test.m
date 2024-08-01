% away 04060029_1574627273_Raw_0.bin
% towards 04050029_1574626078_Raw_0.bin

function [ ] = test(fNameIn, picname)
    fileID = fopen(fNameIn, 'r'); % open file
    Data = fread(fileID, 'int16');% DCA1000 should read in two's complement data
    fclose(fileID); % close file
    % Data length: 65536000
    N = length(Data);
    fprintf("N: %d\n", N);
    disp(Data(1000:1100));

end