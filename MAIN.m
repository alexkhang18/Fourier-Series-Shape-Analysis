clc
close all
clear all

%% Fourier fit

% Fourier series evaluated at these theta values
thetas = linspace(0,2*pi,10000);

% iterates through fossil images
for k = 1:12
    
    % Imports image, finds boundary, converts boundary to x,y coordinates
    BWoutline = imread(strcat(num2str(k),'.png'));
    BWoutline = rgb2gray(BWoutline);
    BWoutline = BWoutline > 0;
    BWoutline = imfill(BWoutline,'holes');
    B = bwboundaries(BWoutline);
    x = B{1,1}(:,1);
    y = B{1,1}(:,2);
    points = [x,y];
    points = unique(points,'rows','stable');
    points = [points(:,2),(size(BWoutline,1)-points(:,1))];

    % Centers and rotates shapes so that longest axis is on x-axis
    points = points - mean(points);
    R = pca(points);
    points = (R*points')';

    % Ensures counterclockwise winding 
    tf = ispolycw(points(:,1),points(:,2));
    if tf 
        points = flip(points,1);
    end

    % re-orders points such that point furthest in the x-axis is first
    indx_ = find(points(:,1) == max(points(:,1)));
    points = [points(indx_:end,:);points(1:indx_-1,:)];

    % Rotates shape back to original pose
    points = (inv(R)*points')';

    % finds arc-length of distance around closed shape
    dt = diff(points);
    dis = zeros(length(dt),1);
    for i = 1:length(dt)
        if i == 1
            dis(i) = norm(dt(i,:));
        else
            dis(i) = dis(i-1) + norm(dt(i,:));
        end
    end
    
    % prepares arc-length data for Fourier series fit
    dis = [0;dis]; % first point is zero arc-length
    dis = normalize(dis,'range')*2*pi;
    
    % Fourier series fit
    points_x = points(:,1);
    points_y = points(:,2);
    [a] = fourier_series_fit(dis,points_x,25);
    [b] = fourier_series_fit(dis,points_y,25);
    
%% normalization 

    A = a(2:2:end);
    B = a(3:2:end);
    C = b(2:2:end);
    D = b(3:2:end);

    theta_ = 0.5*atan((2*(A(1)*B(1)+C(1)*D(1)))/(A(1)^2+C(1)^2-B(1)^2-D(1)^2));

    a_ = A(1)*cos(theta_)+B(1)*sin(theta_);
    c_ = C(1)*cos(theta_)+D(1)*sin(theta_);
    E = sqrt(a_^2+c_^2);

    a = a/E;
    b = b/E;

    % saves normalized variables
    save(strcat(num2str(k),'a_coeff.mat'),'a')
    save(strcat(num2str(k),'b_coeff.mat'),'b')

end

%% PCA analysis and Eigenshape generation 

% loads in saved varialbes and concatenates into one variable called
% f_coeffs
for i = 1:12

    a = load(strcat(num2str(i),'a_coeff.mat'));
    a = a.a;

    b = load(strcat(num2str(i),'b_coeff.mat'));
    b = b.b;

    f_coeffs(i,:) = [a',b'];

end

% Principal component analysis 
A = cov(f_coeffs) ;   % co-variance matrix 
[V,D] = eig(A) ;  % Get Eigenvalues and Eigenvectors 
Eig = diag(D) ; % concatenates Eigenvalues
[val,idx] = sort(Eig,'descend') ; % sorts Eigenvalues
PV = Eig(idx); % sorts Eigenvalues
PC = V(:,idx); % sorts Eigenvectors 

% Computes PCA scores
score = f_coeffs*PC;
score2 = mean(score);
score2_std = std(score);

% Plotting of Eigenshapes from -2*STD to 2*STD on PC axis 1-3
z = 1;
for k = 1:3 % iterates for PC axes 1-3

    % starts at -2*STD
    score2 = mean(score);
    score2_std = std(score);
    score2(k) = score2(k) - 2*score2_std(k);

    for i = 1:5 % Iterates from -2*STD to 2*STD
    
        % Adds STD at each iteration 
        if i > 1
            score2(k) = score2(k) + score2_std(k);
        end
        
        % Computes recovered coefficients 
        PC_inv = inv(PC);
        rec_coeff = score2*PC_inv;
        a_ = rec_coeff(1:length(rec_coeff)/2);
        b_ = rec_coeff(length(rec_coeff)/2+1:end);
        
        % Evaluate Fourier Series with recovered coefficients 
        [x_pred] = fourier_series_evaluate(a_,thetas);
        [y_pred] = fourier_series_evaluate(b_,thetas);
    
        % Superimposed plots 
        figure (k)
        plot(x_pred,y_pred) ; hold on;
        xlim([-2 2])
        ylim([-2 2])
        % xlim([-700 700])
        % ylim([-700 700])
        set(gca,'XTick',[],'YTick',[])
        legend('-2','-1','0','1','2','location','northeastoutside')
        title(strcat('Axis',num2str(k)))

        % Plots shapes at PC Axis 
        figure(10)
        subplot(3,5,z)
            plot(x_pred,y_pred) ; hold on;
        xlim([-2 2])
        ylim([-2 2])
        % xlim([-700 700])
        % ylim([-700 700])
        set(gca,'XTick',[],'YTick',[])
        title(strcat('Axis',num2str(k),',Shape',num2str(i)))
        z = z+1;
    
    end

end

