function [h a1 a2 a3 z2 z3] = hypothesis(X,hprev,Wxh,Whh,Why)

% Useful values
m = size(X, 1);
output_layer_size = size(Why, 1);

h = zeros(output_layer_size, 1);

a1 = X;

%calculate the z2 using the Wxh*a1 + Whh*hprev
z2x = a1 * Wxh';
z2h = hprev' * Whh';
z2 = z2x .+ z2h;

%calculate the activation of hiddenlayer a2
a2 = sigmoid(z2);

%add Bias to layer 2
a2sizem = size(a2,1);
a2 = [ones(a2sizem,1) a2];

z3 = a2 * Why';


a3 = sigmoid(z3);
h = a3';
fprintf('\nHypothesis done::\n');
%pause

% =========================================================================


end
