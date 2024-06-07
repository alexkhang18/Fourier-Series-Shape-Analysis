function [x_pred] = fourier_series_evaluate(a_,x)
%FOURIER_SERIES_EVALUATE.M This function evaluates a Fourier series given the
%Fourier series coefficients a_ and the parameterized value x.

% Input variables
%x - Parameterization variable. In practice, this is the arc length at a
%point of interest. 

% a_ - Fourier series coefficients.

% Output variables
% x_pred - Predicted variable. In practice, this is a fitted or predicted
% value of the x- or y- component of a point of interest. 

% concatenates Fourier series coefficients into a*cos and b*sin
a0 = a_(1);
a = a_(2:2:end);
b = a_(3:2:end);

% pre-allocation of variables
x_pred = zeros(length(x),1);

% evaluates x_pred
for i = 1:length(x)
    for j = 1:length(a)
        if j == 1
            x_pred(i) = a0 + a(j)*cos(j*x(i)) + b(j)*sin(j*x(i));
        else
            x_pred(i) = x_pred(i) + a(j)*cos(j*x(i)) + b(j)*sin(j*x(i));
        end
    end
end

end