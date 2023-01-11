clear;
clc;
close all;

load("mnist.mat", "digits_train", "digits_test", "labels_test", "labels_train");
[x, y, z] = size(digits_test);
resh = cast(reshape(digits_test, x^2, 1, z),"double");

mean_digit = zeros(x^2, 10);
cov_digit = zeros (x^2,x^2, 10);
count_digit = zeros(10,1);

for i=[1:1:z]
    count_digit(labels_test(i,1)+1,1)=count_digit(labels_test(i,1)+1,1)+1;
    mean_digit(:, labels_test(i,1)+1) = mean_digit(:, labels_test(i,1)+1) +resh(:,1,i);
end

for i=[1:10]
   mean_digit(:, i) = mean_digit(:,i)./count_digit(i,1);
end

for i=[1:1:z]
   cov_digit(:,:,(labels_test(i,1)+1))=cov_digit(:,:,(labels_test(i,1)+1))+((resh(:,1,i)-mean_digit(:,labels_test(i,1)+1))*transpose(resh(:,1,i)-mean_digit(:,labels_test(i,1)+1)))./(count_digit(labels_test(i,1)+1,1));
end
V = zeros(x^2, x^2, 10);
D = zeros(x^2, x^2, 10);
for i=[1:1:10]
    [V(:,:,i), D(:,:,i)] = eig(cov_digit(:,:,i));
end
sorted_d = zeros(x^2, 10);
max = zeros(10,1);
for i=[1:1:10]
    m=0;
    for j= [1:1:x^2]
        if(m<D(j,j, i))
            m=D(j,j, i);
            max(i,1)=j;
        end
        sorted_d(j, i)=D(j,j, i);
    end
end

for i=[1:1:10]
    sorted_d(:, i) = sort(sorted_d(:, i));
end
count = zeros(10,1);
for i =[1:1:10]
    for j = [1:1:x^2]
        if sorted_d(j,i)>sorted_d(1,i)/500
            count(i,1)=count(i,1)+1;
        end
    end
end
for i=[1:1:10]
    figure();
    plot ( sorted_d(:, i));
end
dr1=zeros(x^2, 10);
dr2=zeros(x^2, 10);
for i=[1:1:10]
    dr1(:, i) = mean_digit(:, i) - V(:, max(i),i).*D(max(i), max(i),i)^0.5;
    dr2(:, i) = mean_digit(:, i) + V(:, max(i),i).*D(max(i), max(i),i)^0.5;
end

im1 = reshape(dr1, 28, 28, 10);
im3 = reshape(dr2, 28, 28, 10);

im2 = reshape(mean_digit, 28, 28, 10);
for i = [1:1:10]
    figure(10*i+1)
    axis equal;
    pos1= [0.1 0.3 0.25 0.25];
    han1=subplot('position', pos1);
    axis equal;
    pos2= [0.4 0.3 0.25 0.25];
    imagesc(im1(:,:, i));
    han2=subplot('position', pos2);
    axis square;
    pos3= [0.7 0.3 0.25 0.25];
    imagesc(im2(:,:, i));
    han3=subplot('position', pos3);
    axis square;
    imagesc(im3(:,:, i));
end