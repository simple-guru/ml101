function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   output_layer_size, ...
                                   X, y, lambda,sequenceLength)

%Note that the X is sequence of xi* that constitute one data sequence in RNN 



%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

m = size(X, 1);
X = [ones(m, 1) X];     
    
fprintf('\nSize(X)\n');
size(X)
fprintf('\nSize(y)\n');
size(y)

fprintf('\nWeights size Wxh Whh Why\n');





Wxh = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
size(Wxh)

Whh = reshape(nn_params(1 + hidden_layer_size * (input_layer_size + 1):
			hidden_layer_size * (input_layer_size + 1) +((hidden_layer_size * (hidden_layer_size + 1)))), ...
                 hidden_layer_size, (hidden_layer_size + 1));
size(Whh)
Why = reshape(nn_params(1 + hidden_layer_size * (input_layer_size + 1) +((hidden_layer_size * (hidden_layer_size + 1))):
			end), ...
                 output_layer_size, (hidden_layer_size + 1));
size(Why)

% Setup some useful variables


%pause


% You need to return the following variables correctly 
J = 0;
Wxhgrad = zeros(size(Wxh));
Whhgrad = zeros(size(Whh));
Whygrad = zeros(size(Why));

read = 1;

temp = zeros(output_layer_size,1);
Delta2x = zeros(size(Wxhgrad));
Delta2h = zeros(size(Whhgrad));
Delta3 = zeros(size(Whygrad));




while read<=m
fprintf('\nReading i/p seq from X index ::%d to %d\n',read,read + sequenceLength);

%Xseq = reshape(X(read:input_layer_size * (input_layer_size + 1)),sequenceLength,(input_layer_size + 1));
%yseq = reshape(y(read:output_layer_size * (output_layer_size)),sequenceLength,(output_layer_size));

Xseq = X(read:(read+sequenceLength-1),:);
yseq = y(read:(read+sequenceLength-1),:);

a2hdnprev = zeros(hidden_layer_size+1,1);


fprintf('\nXseq size::\n');
size(Xseq)
fprintf('\nyseq size::\n');
size(yseq)

%pause
for i=1:size(Xseq,1) 

[h, a1, a2, a3, z2, z3] = hypothesis(Xseq(i,:),a2hdnprev,Wxh,Whh,Why);


[delta2x delta2h delta3] = Delta(a1,z2,a2,Why,z3,a3,yseq(i,:));

Delta3 = Delta3 + (delta3' * a2);
Delta2h = Delta2h + (delta2h * a2hdnprev');
Delta2x = Delta2x + (delta2x * a1);




temp += (yseq(i) .* log(h)) + ((1 - yseq(i)) .* log(1- h));
fprintf('\ntemp size::\n');
size(temp)

a2hdnprev = a2';

end;

read+=sequenceLength;

%fprintf('press to continue the sequence reading');
%pause
end

fprintf('Done Forward and Backward Propogation.\n press to continue\n ');
%pause
J = (-1/m * sum(temp))



Delta2x(:,1)=0;
Delta2h(:,1)=0;
Delta3(:,1)=0;


Wxhgrad=Delta2x;
Whhgrad=Delta2h;
Whygrad=Delta3;

% Unroll gradients
grad = [Wxhgrad(:) ; Whhgrad(:) ; Whygrad(:)];

end;
