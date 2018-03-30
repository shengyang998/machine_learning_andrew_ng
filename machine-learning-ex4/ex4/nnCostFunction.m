function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
%% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
% ====================== INCORRECT IMPLEMENTATION ======================
% h = hypothesis(X, Theta1, Theta2); % 5000 by 10
% [~, I] = max(h, [], 2);
% h = I; % 5000 by 1
% for i=1:num_labels
%     J = J+cee(h, (y == i));
% end
% J = J/m;

% ====================== CORRECTED VERSION ======================
labels = zeros(size(y, 1), num_labels); % map y to labels with 5000 by 10
for i=1:num_labels
    labels(:, i) = (y==i);
end

h = hypothesis(X, Theta1, Theta2); % feedforward and get h(x): 5000 by 10
J = sum(sum(cee(h, labels)))/m;

%% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% ====================== INCORRECT IMPLEMENTATION ======================
% Delta1 = 0;
% Delta2 = 0;
% z_2 = [ones(size(z_2, 1), 1) z_2]; % 5000 by 26
% for i=1:m
%     delta_3 = h(i, :) - labels(i, :); % 1 by 10
%     delta_2 = delta_3 * Theta2 .* sigmoidGradient(z_2(i, :)); % 1 by 26
%     delta_2 = delta_2(2:end);
%     Delta1 = Delta1 + delta_2 .* a_2(i, :);
%     Delta2 = Delta2 + delta_3 .* a_3(i, :);
% end
% Theta1_grad = 1/m * Delta1;
% Theta2_grad = 1/m * Delta2;
% ====================== CORRECTED VERSION ======================
% first time implementation, with for loop. next time, you can do most of
% it when calc hypothesis(X)
for i=1:m
    a1 = [1 X(i, :)]; % 1 by 401
    z2 = Theta1 * a1'; % 25 by 1
    a2 = sigmoid(z2);
    a2 = [1; a2];
    z3 = Theta2 * a2; % 10 by 1
    a3 = sigmoid(z3);
    
    delta3 = a3 - labels(i, :)'; % 10 by 1
    
    z2 = [1; z2];
    delta2 = Theta2' * delta3 .* sigmoidGradient(z2); % 26 by 1
    delta2 = delta2(2: end); % 25 by 1
    
    Theta2_grad = Theta2_grad + delta3 * a2'; % 10 by 26 -> 260
    Theta1_grad = Theta1_grad + delta2 * a1; % 25 by 401 -> 10025
end
Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;
%% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% regularized cost function
t1 = Theta1.*Theta1;
t1(:, 1) = zeros(size(t1, 1), 1);
t2 = Theta2.*Theta2;
t2(:, 1) = zeros(size(t2, 1), 1);
reg = (lambda/(2*m)) * (sum(sum(t1)) + sum(sum(t2)));
J = J + reg;
% regularized gradient
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda/m) * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda/m) * Theta2(:, 2:end);
%% =========================================================================
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
