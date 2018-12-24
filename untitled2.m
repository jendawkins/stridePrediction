
n = 100;
theta = 3;

q0 = (1/n)*((2*theta)-sqrt(theta));
q1 = (sqrt(theta)/n)*((2*sqrt(theta))-1);
q2 = (theta - sqrt(theta) + n)/n;

disp(q0)
disp(q1)
disp(q2)

1 - (.5)*exp(1.16/2) + .5*exp(-1.16/2);