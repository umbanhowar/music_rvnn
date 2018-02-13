function [] = write_ESC(ESC, filepath, shingling_type, thresh)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    % Conversions to sixteenth note and python zero indexing
    ESC(:, 1) = 4 * ESC(:, 1) - 1;

    full_path = strcat(filepath, '-', shingling_type, '-', thresh, '-ESC');
    csvwrite(full_path, ESC);
end

