clc; clear;
% Class 1
% Region 1
theta = linspace(0,2*pi,10);
rx = rand(1,10);
ry = rand(1,10);
x = rx.*cos(theta)-2;
y = ry.*sin(theta)+1.5;

X = [x' y'];

% Region 2
rx = 0.2*rand(1,10);
ry = 2*rand(1,10);
x  = rx.*cos(theta);
y  = ry.*sin(theta)+2;

X = [X; x' y'];

% Region 3
rx = rand(1,10);
ry = rand(1,10);
x = rx.*cos(theta)+2;
y = ry.*sin(theta)+1.5;

X = [X; x' y'];

% Class 2
% Region 1
rx = rand(1,10);
ry = rand(1,10);
x = rx.*cos(theta)-2;
y = ry.*sin(theta)-0.5;

X = [X; x' y'];

% Region 2
rx = rand(1,10);
ry = rand(1,10);
x = rx.*cos(theta)  ;
y = ry.*sin(theta)-1;

X = [X; x' y'];

% Region 3
rx = rand(1,10);
ry = rand(1,10);
x = rx.*cos(theta)+2;
y = ry.*sin(theta)-0.5;

X = [X; x' y'];

% Clustering
[idx,C] = kmeans(X,6);
X1 = X(idx==1,:);
X2 = X(idx==2,:);
X3 = X(idx==3,:);
X4 = X(idx==4,:);
X5 = X(idx==5,:);
X6 = X(idx==6,:);

scatter(X(:,1),X(:,2)); axis equal; hold on;

plot(C(:,1),C(:,2),'x');
plot(X1(:,1),X1(:,2),'bx');