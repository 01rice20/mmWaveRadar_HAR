clear; clc; close all; 

main = '/media/rspl-admin/Seagate Backup Plus Drive/100sign_ASL Fall 2020/';
pos = 'Front/';

sub_folds = dir(main);
subjects = sub_folds(3:end);
subs = {subjects.name};
seqPerRecord = [2 0 1 5 0 5 2 1 0 1 1 1 1] ; 

mDout = '/media/rspl-admin/Seagate Backup Plus Drive/100sign_ASL Fall 2020/OUTPUT/xethru/';

for s = 7:7
        if seqPerRecord(s) == 0
                continue
        end
        datapath = [main subs{s} '/10ghz/' pos '*.dat'];
        files = dir(datapath);
        mdfolder = [mDout subs{s} '/' pos];
        
        if ~exist(mdfolder, 'dir')
                mkdir(mdfolder)
        end
        
        filenames2 = {files.name};
%         for z = 1:length(filenames2)
%                 temp{1,z} = filenames2{z}(1:end-10);
%         end
%         uniqs = unique(temp);
        uniqs = unique(filenames2);
        for j = 1:length(uniqs)
                match = strfind(filenames2,uniqs{j}); % find matches
                idx = find(~cellfun(@isempty,match)); % find non-empty indices
                RDC = [];
                % concat RDCs with same names
                for r = 1:length(idx)
                        fname = fullfile(files(idx(r)).folder,files(idx(r)).name);
                        [temp2] = RDC_extract_xethru(fname);
                        RDC = [RDC temp2];
                end
                % divide into sub RDCs
                numChirps = floor(size(RDC,2)/seqPerRecord(s));
                for r =1:seqPerRecord(s)
                        tic
                        msg = ['Processing: Subject ''' subs{s} ''', File: ' int2str(j) ' of ' int2str(length(uniqs)) ', Part ' ...
                                num2str(r) '/' num2str(seqPerRecord(s))];   % loading message
                        disp(msg);
                        subRDC = RDC(:,(r-1)*numChirps+1:r*numChirps,:);
                        fname2 =  filenames2(j);
                        mD_Out = [mdfolder uniqs{j}(1:end-4) '_' num2str(r) '.png'];
                        
                        RDC2MD_xethru(subRDC, mD_Out);
                        
                        toc
                end
                
        end
end