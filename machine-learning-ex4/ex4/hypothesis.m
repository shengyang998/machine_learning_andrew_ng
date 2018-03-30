function h = hypothesis(x, theta1, theta2)
x = [ones(size(x, 1), 1) x]; % 5000 by 400
t = sigmoid(x*theta1'); % 5000 by 25
t = [ones(size(t, 1), 1) t];
h = sigmoid(t*theta2'); % 5000 by 10
end