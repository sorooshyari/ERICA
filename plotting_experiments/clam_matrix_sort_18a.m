

%k-value mismatch, k = 6
%GaussianMix1_k_value_mismatch_k_set_to6_100_dim  

A = readmatrix('D:\Donoho\clustering_1\code_my_BS_1\synthetic_data\Gaussian_mixture_1_k_value_mismatch\k_set_to_6\100_dim\Siamak_formed_CLAM_aligned_clusters.csv'); 


A(1,:) = []; 
A(:,1) = []; 

sum_c = sum(A, 2);        %sum across all rows of matrix A

[sorted_rows_A, cluster_index] = sort(A, 2, 'descend');  
 
%writematrix(sorted_rows_A, 'sorted_rows_1.csv')
%writematrix(cluster_index, 'sorted_rows_indices_1.csv') 

%writematrix(sum_c, 'sum_all_rows.csv') 

%------------------form sorted matrix with corresponding index matrix of the points' assignmnets to clusters (end)----------------------

offset = 600;

zeros_for_offset = zeros(1, offset); 

%------------------------------- Evaluating points that were assigned most number of times to cluster 1 (begin) ---------------------------------------
rows_with_first_entry_1 = cluster_index(:,1) == 1;    % Finding rows where the first entry is 1
indices_rows_with_first_entry_1 = find(rows_with_first_entry_1 == 1);    % Finding indices where A equals 1, note: this is a logical variable 0 or 1

print_num_datapoints_most_assigned_to_cluster1 = length(indices_rows_with_first_entry_1)

% Extracting the submatrix
submatrix = sorted_rows_A(rows_with_first_entry_1, :);
submatrix_indices = cluster_index(rows_with_first_entry_1, :);

neighbors_of_cluster_1 = submatrix;

normalized_wrt_cluster_1 = A(indices_rows_with_first_entry_1,:)./sum_c(indices_rows_with_first_entry_1);

y1 = [zeros_for_offset  normalized_wrt_cluster_1(:,2)'  zeros_for_offset  normalized_wrt_cluster_1(:,3)'  zeros_for_offset  normalized_wrt_cluster_1(:,4)'  zeros_for_offset  normalized_wrt_cluster_1(:,5)'   zeros_for_offset  normalized_wrt_cluster_1(:,6)'    zeros_for_offset];

x1= 1:length(y1);
%zeros(1, length(y1))

figure(1)
scatter(x1, 100*y1)
xlim([1 max(x1)])
ylim([0.01 50])
%xlabel('x-axis')
title('Cluster 1 assignments')
ylabel('% assigned to cluster')
xticks([1700  3800  6100  13000  17000])
xticklabels({'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6'})

y1_line1 = length(zeros_for_offset) + length(normalized_wrt_cluster_1(:,1)') +  length(zeros_for_offset)/2;
vline(y1_line1) 

y1_line2 = y1_line1 + length(normalized_wrt_cluster_1(:,2)') +  length(zeros_for_offset);
vline(y1_line2) 

y1_line3 = y1_line2 + length(normalized_wrt_cluster_1(:,3)') +  length(zeros_for_offset);
vline(y1_line3) 

y1_line4 = y1_line3 + length(normalized_wrt_cluster_1(:,4)') +  length(zeros_for_offset);
vline(y1_line4) 
%------------------------------- Evaluating points that were assigned most number of times to cluster 1 (end) ---------------------------------------

%------------------------------- Evaluating points that were assigned most number of times to cluster 2 (begin) ---------------------------------------
rows_with_first_entry_2 = cluster_index(:,1) == 2;    % Finding rows where the first entry is 2
indices_rows_with_first_entry_2 = find(rows_with_first_entry_2 == 1);    % Finding indices where A equals 1, note: this is a logical variable 0 or 1

print_num_datapoints_most_assigned_to_cluster2 = length(indices_rows_with_first_entry_2)

% Extracting the submatrix
submatrix2 = sorted_rows_A(rows_with_first_entry_2, :);
submatrix2_indices = cluster_index(rows_with_first_entry_2, :);

neighbors_of_cluster_2 = submatrix2;

normalized_wrt_cluster_2 = A(indices_rows_with_first_entry_2,:)./sum_c(indices_rows_with_first_entry_2);

y2 = [zeros_for_offset  normalized_wrt_cluster_2(:,1)'  zeros_for_offset  normalized_wrt_cluster_2(:,3)'  zeros_for_offset  normalized_wrt_cluster_2(:,4)'   zeros_for_offset  normalized_wrt_cluster_2(:,5)'   zeros_for_offset  normalized_wrt_cluster_2(:,6)'   zeros_for_offset];

x2= 1:length(y2);
%zeros(1, length(y2))

figure(2)
scatter(x2, 100*y2)
xlim([1 max(x2)])
ylim([0.01 50])
%xlabel('x-axis')
title('Cluster 2 assignments')
ylabel('% assigned to cluster')
xticks([2100  5660  9160  12800  16200])
xticklabels({'Cluster 1', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6'})
%------------------------------- Evaluating points that were assigned most number of times to cluster 2 (end) ---------------------------------------

%------------------------------- Evaluating points that were assigned most number of times to cluster 3 (begin) ---------------------------------------
rows_with_first_entry_3 = cluster_index(:,1) == 3;    % Finding rows where the first entry is 3
indices_rows_with_first_entry_3 = find(rows_with_first_entry_3 == 1);    % Finding indices where A equals 1, note: this is a logical variable 0 or 1

print_num_datapoints_most_assigned_to_cluster3 = length(indices_rows_with_first_entry_3)

% Extracting the submatrix
submatrix3 = sorted_rows_A(rows_with_first_entry_3, :);
submatrix3_indices = cluster_index(rows_with_first_entry_3, :);

neighbors_of_cluster_3 = submatrix3;

normalized_wrt_cluster_3 = A(indices_rows_with_first_entry_3,:)./sum_c(indices_rows_with_first_entry_3);

y3 = [zeros_for_offset  normalized_wrt_cluster_3(:,1)'  zeros_for_offset  normalized_wrt_cluster_3(:,2)'  zeros_for_offset  normalized_wrt_cluster_3(:,4)'   zeros_for_offset   normalized_wrt_cluster_3(:,5)'   zeros_for_offset  normalized_wrt_cluster_3(:,6)'   zeros_for_offset];

x3= 1:length(y3);
%zeros(1, length(y3))

figure(3)
scatter(x3, 100*y3)
xlim([1 max(x3)])
ylim([0.01 50])
%xlabel('x-axis')
title('Cluster 3 assignments')
ylabel('% assigned to cluster')
xticks([1475  3788  6100  8400  10500])
xticklabels({'Cluster 1', 'Cluster 2', 'Cluster 4', 'Cluster 5', 'Cluster 6'})
%------------------------------- Evaluating points that were assigned most number of times to cluster 3 (end) ---------------------------------------


%------------------------------- Evaluating points that were assigned most number of times to cluster 4 (begin) ---------------------------------------
rows_with_first_entry_4 = cluster_index(:,1) == 4;    % Finding rows where the first entry is 4
indices_rows_with_first_entry_4 = find(rows_with_first_entry_4 == 1);    % Finding indices where A equals 1, note: this is a logical variable 0 or 1

print_num_datapoints_most_assigned_to_cluster4 = length(indices_rows_with_first_entry_4)

% Extracting the submatrix
submatrix4 = sorted_rows_A(rows_with_first_entry_4, :);
submatrix4_indices = cluster_index(rows_with_first_entry_4, :);

neighbors_of_cluster_4 = submatrix4;

normalized_wrt_cluster_4 = A(indices_rows_with_first_entry_4,:)./sum_c(indices_rows_with_first_entry_4);

y4 = [zeros_for_offset  normalized_wrt_cluster_4(:,1)'  zeros_for_offset  normalized_wrt_cluster_4(:,2)'  zeros_for_offset  normalized_wrt_cluster_4(:,3)'   zeros_for_offset  normalized_wrt_cluster_4(:,5)'   zeros_for_offset  normalized_wrt_cluster_4(:,6)'   zeros_for_offset];

x4= 1:length(y4);
%zeros(1, length(y4))

figure(4)
scatter(x4, 100*y4)
xlim([1 max(x4)])
ylim([0.01 50])
%xlabel('x-axis')
title('Cluster 4 assignments')
ylabel('% assigned to cluster')
xticks([1900  5640  9050  12750  16100])
xticklabels({'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 5', 'Cluster 6'})
%------------------------------- Evaluating points that were assigned most number of times to cluster 4 (end) ---------------------------------------


%------------------------------- Evaluating points that were assigned most number of times to cluster 5 (begin) ---------------------------------------
rows_with_first_entry_5 = cluster_index(:,1) == 5;    % Finding rows where the first entry is 5
indices_rows_with_first_entry_5 = find(rows_with_first_entry_5 == 1);    % Finding indices where A equals 1, note: this is a logical variable 0 or 1

print_num_datapoints_most_assigned_to_cluster5 = length(indices_rows_with_first_entry_5)

% Extracting the submatrix
submatrix5 = sorted_rows_A(rows_with_first_entry_5, :);
submatrix5_indices = cluster_index(rows_with_first_entry_5, :);

neighbors_of_cluster_5 = submatrix5;

normalized_wrt_cluster_5 = A(indices_rows_with_first_entry_5,:)./sum_c(indices_rows_with_first_entry_5);

y5 = [zeros_for_offset  normalized_wrt_cluster_5(:,1)'  zeros_for_offset  normalized_wrt_cluster_5(:,2)'  zeros_for_offset  normalized_wrt_cluster_5(:,3)'   zeros_for_offset  normalized_wrt_cluster_5(:,4)'   zeros_for_offset  normalized_wrt_cluster_5(:,6)'   zeros_for_offset];

x5= 1:length(y5);
%zeros(1, length(y5))

figure(5)
scatter(x5, 100*y5)
xlim([1 max(x5)])
ylim([0.01 50])
%xlabel('x-axis')
title('Cluster 5 assignments')
ylabel('% assigned to cluster')
xticks([750  1570  2420  3293  4140])
xticklabels({'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 6'})
%------------------------------- Evaluating points that were assigned most number of times to cluster 5 (end) ---------------------------------------

%------------------------------- Evaluating points that were assigned most number of times to cluster 6 (begin) ---------------------------------------
rows_with_first_entry_6 = cluster_index(:,1) == 6;    % Finding rows where the first entry is 6
indices_rows_with_first_entry_6 = find(rows_with_first_entry_6 == 1);    % Finding indices where A equals 1, note: this is a logical variable 0 or 1

print_num_datapoints_most_assigned_to_cluster6 = length(indices_rows_with_first_entry_6)

% Extracting the submatrix
submatrix6 = sorted_rows_A(rows_with_first_entry_6, :);
submatrix6_indices = cluster_index(rows_with_first_entry_6, :);

neighbors_of_cluster_6 = submatrix6;

normalized_wrt_cluster_6 = A(indices_rows_with_first_entry_6,:)./sum_c(indices_rows_with_first_entry_6);

y6 = [zeros_for_offset  normalized_wrt_cluster_6(:,1)'  zeros_for_offset  normalized_wrt_cluster_6(:,2)'  zeros_for_offset  normalized_wrt_cluster_6(:,3)'  zeros_for_offset  normalized_wrt_cluster_6(:,4)'   zeros_for_offset  normalized_wrt_cluster_6(:,5)'   zeros_for_offset];

x6= 1:length(y6);
%zeros(1, length(y6))

figure(6)
scatter(x6, 100*y6)
xlim([1 max(x6)])
ylim([0.01 50])
%xlabel('x-axis')
title('Cluster 6 assignments')
ylabel('% assigned to cluster')
xticks([705  1550  2370  3180  3972])
xticklabels({'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'})
%------------------------------- Evaluating points that were assigned most number of times to cluster 6 (end) ---------------------------------------

mean_normalized_wrt_cluster1_assign_to_cluster1 = 100 * mean(normalized_wrt_cluster_1(:,1))
mean_normalized_wrt_cluster1_assign_to_cluster2 = 100 * mean(normalized_wrt_cluster_1(:,2))
mean_normalized_wrt_cluster1_assign_to_cluster3 = 100 * mean(normalized_wrt_cluster_1(:,3))
mean_normalized_wrt_cluster1_assign_to_cluster4 = 100 * mean(normalized_wrt_cluster_1(:,4))
mean_normalized_wrt_cluster1_assign_to_cluster5 = 100 * mean(normalized_wrt_cluster_1(:,5))
mean_normalized_wrt_cluster1_assign_to_cluster6 = 100 * mean(normalized_wrt_cluster_1(:,6))

var_normalized_wrt_cluster1_assign_to_cluster1 = (100 * 100) * var(normalized_wrt_cluster_1(:,1))
var_normalized_wrt_cluster1_assign_to_cluster2 = (100 * 100) * var(normalized_wrt_cluster_1(:,2))
var_normalized_wrt_cluster1_assign_to_cluster3 = (100 * 100) * var(normalized_wrt_cluster_1(:,3))
var_normalized_wrt_cluster1_assign_to_cluster4 = (100 * 100) * var(normalized_wrt_cluster_1(:,4))
var_normalized_wrt_cluster1_assign_to_cluster5 = (100 * 100) * var(normalized_wrt_cluster_1(:,5))
var_normalized_wrt_cluster1_assign_to_cluster6 = (100 * 100) * var(normalized_wrt_cluster_1(:,6))

mean_normalized_wrt_cluster2_assign_to_cluster1 = 100 * mean(normalized_wrt_cluster_2(:,1))
mean_normalized_wrt_cluster2_assign_to_cluster2 = 100 * mean(normalized_wrt_cluster_2(:,2))
mean_normalized_wrt_cluster2_assign_to_cluster3 = 100 * mean(normalized_wrt_cluster_2(:,3))
mean_normalized_wrt_cluster2_assign_to_cluster4 = 100 * mean(normalized_wrt_cluster_2(:,4))
mean_normalized_wrt_cluster2_assign_to_cluster5 = 100 * mean(normalized_wrt_cluster_2(:,5))
mean_normalized_wrt_cluster2_assign_to_cluster6 = 100 * mean(normalized_wrt_cluster_2(:,6))

var_normalized_wrt_cluster2_assign_to_cluster1 = (100 * 100) * var(normalized_wrt_cluster_2(:,1))
var_normalized_wrt_cluster2_assign_to_cluster2 = (100 * 100) * var(normalized_wrt_cluster_2(:,2))
var_normalized_wrt_cluster2_assign_to_cluster3 = (100 * 100) * var(normalized_wrt_cluster_2(:,3))
var_normalized_wrt_cluster2_assign_to_cluster4 = (100 * 100) * var(normalized_wrt_cluster_2(:,4))
var_normalized_wrt_cluster2_assign_to_cluster5 = (100 * 100) * var(normalized_wrt_cluster_2(:,5))
var_normalized_wrt_cluster2_assign_to_cluster6 = (100 * 100) * var(normalized_wrt_cluster_2(:,6))

mean_normalized_wrt_cluster3_assign_to_cluster1 = 100 * mean(normalized_wrt_cluster_3(:,1))
mean_normalized_wrt_cluster3_assign_to_cluster2 = 100 * mean(normalized_wrt_cluster_3(:,2))
mean_normalized_wrt_cluster3_assign_to_cluster3 = 100 * mean(normalized_wrt_cluster_3(:,3))
mean_normalized_wrt_cluster3_assign_to_cluster4 = 100 * mean(normalized_wrt_cluster_3(:,4))
mean_normalized_wrt_cluster3_assign_to_cluster5 = 100 * mean(normalized_wrt_cluster_3(:,5))
mean_normalized_wrt_cluster3_assign_to_cluster6 = 100 * mean(normalized_wrt_cluster_3(:,6))

var_normalized_wrt_cluster3_assign_to_cluster1 = (100 * 100) * var(normalized_wrt_cluster_3(:,1))
var_normalized_wrt_cluster3_assign_to_cluster2 = (100 * 100) * var(normalized_wrt_cluster_3(:,2))
var_normalized_wrt_cluster3_assign_to_cluster3 = (100 * 100) * var(normalized_wrt_cluster_3(:,3))
var_normalized_wrt_cluster3_assign_to_cluster4 = (100 * 100) * var(normalized_wrt_cluster_3(:,4))
var_normalized_wrt_cluster3_assign_to_cluster5 = (100 * 100) * var(normalized_wrt_cluster_3(:,5))
var_normalized_wrt_cluster3_assign_to_cluster6 = (100 * 100) * var(normalized_wrt_cluster_3(:,6))

mean_normalized_wrt_cluster4_assign_to_cluster1 = 100 * mean(normalized_wrt_cluster_4(:,1))
mean_normalized_wrt_cluster4_assign_to_cluster2 = 100 * mean(normalized_wrt_cluster_4(:,2))
mean_normalized_wrt_cluster4_assign_to_cluster3 = 100 * mean(normalized_wrt_cluster_4(:,3))
mean_normalized_wrt_cluster4_assign_to_cluster4 = 100 * mean(normalized_wrt_cluster_4(:,4))
mean_normalized_wrt_cluster4_assign_to_cluster5 = 100 * mean(normalized_wrt_cluster_4(:,5))
mean_normalized_wrt_cluster4_assign_to_cluster6 = 100 * mean(normalized_wrt_cluster_4(:,6))

var_normalized_wrt_cluster4_assign_to_cluster1 = (100 * 100) * var(normalized_wrt_cluster_4(:,1))
var_normalized_wrt_cluster4_assign_to_cluster2 = (100 * 100) * var(normalized_wrt_cluster_4(:,2))
var_normalized_wrt_cluster4_assign_to_cluster3 = (100 * 100) * var(normalized_wrt_cluster_4(:,3))
var_normalized_wrt_cluster4_assign_to_cluster4 = (100 * 100) * var(normalized_wrt_cluster_4(:,4))
var_normalized_wrt_cluster4_assign_to_cluster5 = (100 * 100) * var(normalized_wrt_cluster_4(:,5))
var_normalized_wrt_cluster4_assign_to_cluster6 = (100 * 100) * var(normalized_wrt_cluster_4(:,6))

mean_normalized_wrt_cluster5_assign_to_cluster1 = 100 * mean(normalized_wrt_cluster_5(:,1))
mean_normalized_wrt_cluster5_assign_to_cluster2 = 100 * mean(normalized_wrt_cluster_5(:,2))
mean_normalized_wrt_cluster5_assign_to_cluster3 = 100 * mean(normalized_wrt_cluster_5(:,3))
mean_normalized_wrt_cluster5_assign_to_cluster4 = 100 * mean(normalized_wrt_cluster_5(:,4))
mean_normalized_wrt_cluster5_assign_to_cluster5 = 100 * mean(normalized_wrt_cluster_5(:,5))
mean_normalized_wrt_cluster5_assign_to_cluster6 = 100 * mean(normalized_wrt_cluster_5(:,6))

var_normalized_wrt_cluster5_assign_to_cluster1 = (100 * 100) * var(normalized_wrt_cluster_5(:,1))
var_normalized_wrt_cluster5_assign_to_cluster2 = (100 * 100) * var(normalized_wrt_cluster_5(:,2))
var_normalized_wrt_cluster5_assign_to_cluster3 = (100 * 100) * var(normalized_wrt_cluster_5(:,3))
var_normalized_wrt_cluster5_assign_to_cluster4 = (100 * 100) * var(normalized_wrt_cluster_5(:,4))
var_normalized_wrt_cluster5_assign_to_cluster5 = (100 * 100) * var(normalized_wrt_cluster_5(:,5))
var_normalized_wrt_cluster5_assign_to_cluster6 = (100 * 100) * var(normalized_wrt_cluster_5(:,6))

mean_normalized_wrt_cluster6_assign_to_cluster1 = 100 * mean(normalized_wrt_cluster_6(:,1))
mean_normalized_wrt_cluster6_assign_to_cluster2 = 100 * mean(normalized_wrt_cluster_6(:,2))
mean_normalized_wrt_cluster6_assign_to_cluster3 = 100 * mean(normalized_wrt_cluster_6(:,3))
mean_normalized_wrt_cluster6_assign_to_cluster4 = 100 * mean(normalized_wrt_cluster_6(:,4))
mean_normalized_wrt_cluster6_assign_to_cluster5 = 100 * mean(normalized_wrt_cluster_6(:,5))
mean_normalized_wrt_cluster6_assign_to_cluster6 = 100 * mean(normalized_wrt_cluster_6(:,6))

var_normalized_wrt_cluster6_assign_to_cluster1 = (100 * 100) * var(normalized_wrt_cluster_6(:,1))
var_normalized_wrt_cluster6_assign_to_cluster2 = (100 * 100) * var(normalized_wrt_cluster_6(:,2))
var_normalized_wrt_cluster6_assign_to_cluster3 = (100 * 100) * var(normalized_wrt_cluster_6(:,3))
var_normalized_wrt_cluster6_assign_to_cluster4 = (100 * 100) * var(normalized_wrt_cluster_6(:,4))
var_normalized_wrt_cluster6_assign_to_cluster5 = (100 * 100) * var(normalized_wrt_cluster_6(:,5))
var_normalized_wrt_cluster6_assign_to_cluster6 = (100 * 100) * var(normalized_wrt_cluster_6(:,6))

