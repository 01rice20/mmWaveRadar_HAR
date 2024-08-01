% folder_path = '../../../dataset/rawdata/24ghz/19_Scissors_gait';
% file_list = dir(fullfile(folder_path, '*.dat'));
% for i = 1:length(file_list)
%     current_file_path = fullfile(folder_path, file_list(i).name);
%     match_result = regexp(current_file_path, '(\d{8}_\d{10})\', 'match');

%     if ~isempty(match_result)
%         extracted_part = match_result{1};
%         new_string = ['./velocity/19/' extracted_part '_velocity.png'];
%         disp(new_string);
%     else
%         disp('No match found');
%     end
%     velocity24(current_file_path, new_string);
% end
mainFolder = '../../../dataset/rawdata/24ghz';
contents = dir(mainFolder);

for i = 1:length(contents)
    if contents(i).isdir && ~strcmp(contents(i).name, '.') && ~strcmp(contents(i).name, '..')
        folder_path = fullfile(mainFolder, contents(i).name);
        filename = contents(i).name
        % fprintf("file name: %s\n", contents(i).name)
        file_list = dir(fullfile(folder_path, '*.dat'));
        for x = 1:length(file_list)
            current_file_path = fullfile(folder_path, file_list(x).name);
            match_result = regexp(current_file_path, '(\d{8}_\d{10})', 'match');

            if ~isempty(match_result)
                extracted_part = match_result{1};
                new_string = ['./velocity_24/' filename '/' extracted_part '_velocity.png'];
                disp(new_string);
            else
                disp('No match found');
            end
            velocity24(current_file_path, new_string);
        end
    end
end