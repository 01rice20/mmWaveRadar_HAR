folder_path = '../../../dataset/rawdata/77ghz/19_Scissors_gait/'

% fprintf("file name: %s\n", contents(i).name)
file_list = dir(fullfile(folder_path, '*.bin'));
for x = 1:length(file_list)
    current_file_path = fullfile(folder_path, file_list(x).name);
    match_result = regexp(current_file_path, '(\d{8}_\d+_Raw_\d+)', 'match');

    if ~isempty(match_result)
        extracted_part = match_result{1};
        new_string = ['./range/' filename '/' extracted_part '_range.png'];
        disp(new_string);
    else
        disp('No match found');
    end
    range77(current_file_path, new_string);
end
