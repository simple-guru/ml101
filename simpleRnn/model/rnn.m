%% Initialization
clear ; close all; clc

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

%#You need to create or import input data X
%#X = importInputData(textFilename,sequenceLength);

%#You need to create or import input data y
%#y = importOutputData(textFilename,sequenceLength);

%load('ex4data1.mat');



%ABCDE
%BCDEA
%CDEAB

X = [
	1 0 0 0 0;0 1 0 0 0;0 0 1 0 0; 0 0 0 1 0;0 0 0 0 1;%ABCDE
	0 1 0 0 0;0 0 1 0 0; 0 0 0 1 0;0 0 0 0 1;1 0 0 0 0;%BCDEA
	0 0 1 0 0; 0 0 0 1 0;0 0 0 0 1;1 0 0 0 0;0 1 0 0 0;%CDEAB
    ];

y = [
 	0 1 0 0 0;0 0 1 0 0; 0 0 0 1 0;0 0 0 0 1;1 0 0 0 0;%BCDEA
	0 0 1 0 0; 0 0 0 1 0;0 0 0 0 1;1 0 0 0 0;0 1 0 0 0;%CDEAB
	1 0 0 0 0;0 1 0 0 0;0 0 1 0 0; 0 0 0 1 0;0 0 0 0 1;%ABCDE
    ];



m = size(X, 1);
vocabSize = 5;


%% Setup the parameters you will use for this exercise
input_layer_size  = vocabSize;  % VocabSize
hidden_layer_size = 10;   % 25 hidden units
output_layer_size = vocabSize;          % VocabSize
sequenceLength = 5;	  % sequence length of each data xi* in i/p	

initial_Wxh = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Whh = randInitializeWeights(hidden_layer_size, hidden_layer_size);
initial_Why = randInitializeWeights(hidden_layer_size, output_layer_size);

% Unroll parameters
initial_nn_params = [initial_Wxh(:) ; initial_Whh(:) ; initial_Why(:)];
fprintf('initial_nn_params size::');
size(initial_nn_params)
save initialweights.txt initial_nn_params -ascii

%Load the weights into variables Theta1 and Theta2
%Load('ex4weights.mat');
% Unroll parameters 
%nn_params = [Theta1(:) ; Theta2(:)];



%% ================ Part 3: Compute Cost (Feedforward) ================
%  To the neural network, you should first start by implementing the
%  feedforward part of the neural network that returns the cost only. You
%  should complete the code in nnCostFunction.m to return cost. After
%  implementing the feedforward to compute the cost, you can verify that
%  your implementation is correct by verifying that you get the same cost
%  as us for the fixed debugging parameters.
%
%  We suggest implementing the feedforward cost *without* regularization
%  first so that it will be easier for you to debug. Later, in part 4, you
%  will get to implement the regularized cost.
%

fprintf('\nTest Feedforward Using Neural Network ...\n')

% Weight regularization parameter (we set this to 0 here).
lambda = 0;

J = nnCostFunction(initial_nn_params, input_layer_size, hidden_layer_size, ...
                   output_layer_size, X, y, lambda,sequenceLength);


fprintf('\nTraining Neural Network... \n')
pause
%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 50);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized

costFunction = @(p) nnCostFunction(initial_nn_params,input_layer_size, hidden_layer_size, output_layer_size, X, y, lambda,sequenceLength);



% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

fprintf('nn_params size::');
size(nn_params)
save finalweights.txt nn_params -ascii

pause
% Obtain Theta1 and Theta2 back from nn_params
Wxh = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Whh = reshape(nn_params((1 + (hidden_layer_size * (hidden_layer_size + 1))):end), ...
                 hidden_layer_size, (hidden_layer_size + 1));

Why = reshape(nn_params((1 + (hidden_layer_size * (output_layer_size + 1))):end), ...
                 output_layer_size, (hidden_layer_size + 1));


fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================= Part 10: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

