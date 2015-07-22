clc; close all

desc_size = 16;

% train desc
fileID = fopen('train_descriptors.txt','r');
values = fscanf(fileID,'%f');
train_descriptors = reshape(values, [desc_size, numel(values) / desc_size])'; % n x 16
% %train labels
fileID = fopen('train_labels.txt','r');
train_labels = fscanf(fileID,'%f');

% test desc
% fileID = fopen('test_descriptors.txt','r');
% values = fscanf(fileID,'%f');
% test_descriptors = reshape(values, [desc_size, numel(values) / desc_size])'; % n x 16
% % test labels
% fileID = fopen('test_labels.txt','r');
% test_labels = fscanf(fileID,'%f');
% %predicted labels
% fileID = fopen('predicted_labels.txt','r');
% predicted_labels = fscanf(fileID,'%f');
fclose('all');


%%
% USED TOOLBOX - http://lvdmaaten.github.io/drtoolbox/
NUM_DIM = 3; % num dimensions map to (2 or 3)
METHOD = 'tSNE'; %'PCA';
[mapped_train, mapping_train] = compute_mapping(train_descriptors, METHOD, NUM_DIM);


% display
dx = 0.1; dy = 0.1; dz = 0.1;
b = num2str(train_labels); c = cellstr(b(1:3:end, :));
if (NUM_DIM == 2)
    figure, text(mapped_train(1:3:end,1)+dx, mapped_train(1:3:end,2)+dy, c, '\fontsize{1} text'); hold on;
    colormap(jet),scatter( mapped_train(:,1), mapped_train(:,2), 30, train_labels, 'd', 'filled');

elseif (NUM_DIM == 3)    
    figure, colormap(jet),scatter3(mapped_train(:,1), mapped_train(:,2), mapped_train(:,3), 20, train_labels, 'd', 'filled');
    
end