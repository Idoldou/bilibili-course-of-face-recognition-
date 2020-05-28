# Exercise

MNIST dataset \([http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)\) is a large collection of real handwritten digits, widely used in computer vision and machine learning research. The dataset contains 60,000 training grey-level images of 28x28 pixels for 10 classes of digits \(“0”, “1”, “2”, … “9”\). Each class has 6000 samples. A few examples are shown below. Along with the training samples, there are also matching labels for each of the training samples. The dataset also has a test set and respective labels of 1000 images. You may download its original format or Matlab .mat format \(available from BB. You could load it use “d=load\(‘mnist.mat’\); then d.trainX is a 60000x784 matrix, ie. training patterns, d.trainY contains their lables, d.testX test patterns, and d.testY labels of test patterns\). Each image is expressed as a 784-dimensional column vector \(28x28 =784\). Corresponding label is 0, or 1, …, or 9.

![example](.gitbook/assets/image%20%287%29.png)

Extend the single perceptron to a single layer of 10 perceptrons, each for each class of digits. You will need to use the sigmoid activation function instead of the sign function, shown below and use the following weight updating equation,

![](.gitbook/assets/image%20%286%29.png)

You may like to derive the above weight updating equation if you wish. Apply the weight updating equation to train the single layer of perceptrons. You may use 10% of the training set to start with \(i.e. randomly sampled or selected 600 images per class\), once working then expanding to the entire training set. Target or desired output for 10 perceptron can be formed based on the label of a training image. For example, if a training image is of digit ‘0’, then the target output is \[1, 0, 0, 0, 0, 0, 0, 0, 0, 0\]T . For a training image of digit ‘6’, the target output is \[0, 0, 0, 0, 0, 0, 1, 0, 0, 0\]T . This is the so-called 1-of-c coding scheme. You can set a fixed number of iterations or epochs \(for each epoch all training patterns are used once, usually in random order\), in case you set a training error rate as the stopping criterion but it may never be satisfied.

After training, apply the network to the test dataset. Report performance in terms accuracy or error rate for both training set and test set. Discuss influence of the learning rate to the performance.

```text
% Single Layer Perceptrons
% Classifying digital numbers 0-9 
clc;
clear;
d=load('mnist.mat');
% d.trainX = d.trainX(1:6000,:);
% d.trainY = d.trainY(:,1:6000);
train = double(d.trainX)/256;
labeltrain = double(d.trainY);
test = double(d.testX)/256;
labeltest = double(d.testY);
w = randn(784,10)/100;
b = randn(1,10)/100;
y = zeros(60000,10);
d = zeros(60000,10);
a = 0.1;
for i = 1:60000
d(i,labeltrain(i)+1) = 1;
end
for i = 1:50
  for k = 1:60000y(k,:) = 1./(1+exp(-train(k,:)*w-b));
  w = w+((a*(d(k,:)-y(k,:)).*y(k,:).*(1-y(k,:)))'*train(k,:))';
  b = b+(a*(d(k,:)-y(k,:)).*y(k,:).*(1-y(k,:)));
  end
  end
for k= 1:60000
[max_m,index] = max(y(k,:));
r(1,k)=index-1;
end
errorate_train= sum(r~=labeltrain)/60000;
for k = 1:10000
m(k,:) = 1./(1+exp(-test(k,:)*w-b));
[max_m,index] = max(m(k,:));
e(1,k)=index-1;
enderrorrate_test = sum(e~=labeltest)/10000
```

Then apply MLP to the MNIST data. Details of MLP can be found in the lecture notes. You may use 10-20 hidden nodes in the hidden layer, while the output layer should have 10 nodes representing 10 classes. After training, apply the MLP to the test dataset. Report performance in terms accuracy or error rate for both training set and test set. You may like to explore and discuss the influence of the following parameters;

 • learning rate 

• number of hidden nodes

 • stopping rule

```text
clc;
clear;
d=load('mnist.mat');
% d.trainX =d.trainX(1:6000,:);
% d.trainY = d.trainY(:,1:6000);
train = double(d.trainX)/256;
labeltrain = double(d.trainY);
test = double(d.testX)/256;
labeltest = double(d.testY);
wh = randn(784,15)/100;
bh = randn(1,15)/100;
wo = randn(15,10)/100;
bo = randn(1,10)/100;
v = zeros(60000,15);
y = zeros(60000,10);
d = zeros(60000,10);
a = 0.1;
for i = 1:60000
d(i,labeltrain(i)+1) = 1;
end
for i = 1:50
  for k = 1:60000
  v(k,:) = 1./(1+exp(-train(k,:)*wh-bh));
  y(k,:) = 1./(1+exp(-v(k,:)*wo-bo));
  wo = wo+((a*(d(k,:)-y(k,:)).*y(k,:).*(1-y(k,:)))'*v(k,:))';
  bo = bo+(a*(d(k,:)-y(k,:)).*y(k,:).*(1-y(k,:)));
  wh = wh+((a*(wo*((d(k,:)-y(k,:)).*y(k,:).*(1-y(k,:)))')'.*v(k,:).*(1-v(k,:)))'*train(k,:))';
  bh = bh+(a*(wo*((d(k,:)-y(k,:)).*y(k,:).*(1-y(k,:)))')'.*v(k,:).*(1-v(k,:)));
  end
  end
for k= 1:60000
[max_m,index] = max(y(k,:));
r(1,k)=index-1;
end
errorate_train = sum(r~=labeltrain)/60000;
for k = 1:10000
n(k,:) = 1./(1+exp(-test(k,:)*wh-bh));
m(k,:) = 1./(1+exp(-n(k,:)*wo-bo));
[max_m,index] = max(m(k,:));e(1,k)=index-1;
end
errorrate_test = sum(e~=labeltest)/10000;
```

