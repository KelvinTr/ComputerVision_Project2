clear; close all;

I=imread('retina1.jpg'); 
J(:,:)=I(:,:,2);

%% Matched Filtering Edge Detection

N = 9;                  % Filter Size
y = -(N-1)/2:(N-1)/2;   % Filter Range
sigma = 4;              
S = 1;

% Gaussian Filter Construction
gaus = -1*(1/(sqrt(2*pi*(sigma^2)))) * exp(-1*(y.^2)/(2*(sigma^2)));            % Gaussian Function
m0 = mean(gaus);                                                                % Average of Gaussian
gaus = gaus - m0;                                                               % Shifting Gaussian by the mean
gaus_mat = repmat(gaus, S, 1);                                                  
gaus = [zeros(5, N+10); zeros(S, 5), gaus_mat, zeros(S, 5); zeros(5, N+10)];    % Zero-padding and Gaussian Construction

% Gaussian Filter Bank Construction
gaus_group=cell(12,1);
for i = 1:12
    gaus_group{i} = imrotate(gaus, (i-1)*15, 'bicubic', 'crop');
end
gaus_img=cell(12,1);

% Guassian Filters Convolve with Input Image
for i = 1:12
    gaus_img{i} = conv2(J,gaus_group{i});
    gaus_img{i} = gaus_img{i}(8:626, 9:808);
end

% Convolved Images Merge
final_img = zeros(size(gaus_img{1}));
for i = 1:size(gaus_img{1}, 1)
    for j = 1:size(gaus_img{1},2)
        for k = 1:12
            if final_img(i,j) < gaus_img{k}(i,j)
                final_img(i,j) = gaus_img{k}(i,j);
            end
        end
    end
end

% Length Filtering
level = graythresh(final_img);
final_img = imbinarize(final_img,level);
L = bwlabel(final_img, 8);
for i = 1:max(max(L))
    [r,c] = find(L==i);
    for j = 1:size(r,1)
        if size(r) < 200
            final_img(r(j),c(j)) = 0;
        end
    end
end
imshow(I); figure;
imshow(J); figure;
imshow(final_img); figure;

%% Canny Edge Detection
Low = 0.093549; 
High = 0.09364; 
sigmaC = 1.41;   
BW = edge(J, "canny", [Low High], sigmaC);

% Length Filtering
L1 = bwlabel(BW, 8);
for i = 1:max(max(L1))
    [r,c] = find(L1==i);
    for j = 1:size(r,1)
        if size(r) < 80
            BW(r(j),c(j)) = 0;
        end
    end
end
imshow(BW); figure;

%% Laplacian of Gaussian Edge Detection
Thresh = 0.00026;
sigmaL = 2.98;
BW2 = edge(J, "log", Thresh, sigmaL);

% Length Filtering
L2 = bwlabel(BW2, 8);
for i = 1:max(max(L2))
    [r,c] = find(L2==i);
    for j = 1:size(r,1)
        if size(r) < 80
            BW2(r(j),c(j)) = 0;
        end
    end
end

imshow(BW2);

