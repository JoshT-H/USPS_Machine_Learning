%%Coursework of <name> <knumber>, Feb 2021%%
% rename this file to k12345678.m for submission, using your k number
%%%%%%%%%%%%%
%% initialization
clear; close all; clc; format longg;

load 'USPS_dataset9296.mat' X t; % loads 9296 handwritten 16x16 images X dim(X)=[9296x256] and the lables t in [0:9] dim(t)=[9298x1]
[Ntot,D] =      size(X);         % Ntot = number of total dataset samples. D =256=input dimension

% Anonymous functions as inlines
show_vec_as_image16x16 =    @(row_vec)      imshow(-(reshape(row_vec,16,16)).');    % shows the image of a row vector with 256 elements. For matching purposes, a negation and rotation are needed.
sigmoid =                   @(x)            1./(1+exp(-x));                         % overwrites the existing sigmoid, in order to avoid the toolbox
LSsolver =                  @(Xmat,tvec)    ( Xmat.' * Xmat ) \ Xmat.' * tvec;      % Least Square solver

PLOT_DATASET =  1;      % For visualization. Familiarise yourself by running with 1. When submiting the file, set back to 0
if PLOT_DATASET
    figure(8); sgtitle('First 24 samples and labels from USPS data set');
    for n=1:4*6
        subplot(4,6,n);
        show_vec_as_image16x16(X(n,:));
        title(['t_{',num2str(n),'}=',num2str(t(n)),'   x_{',num2str(n),'}=']);
    end
end
%% Section 0: Seperation of Data
% Code here initialization code that manipulations the data sets
 Ones_Index = find(t==1); % Finds all the Values that are equal to 1 I.e Index from the label
 Zero_Index = find(t==0); % Finds all the Values that are equal to 0 I.e Index from the label (Row Value) 

% Ivec =Image Vector % In we know the index positions for all the t labels (i.e
% outputs) we can use this to find all the images positions in the x matrix
% i.e if a at row 3, t = 1 in the t matrix , at row 3 in the x (image
% matrix) is the corresponding hand drawn image for t = 1. 

IVec_1 = X(Ones_Index,:); % Creates a matrtix of only digital Images containg Ones 
IVec_0 = X(Zero_Index,:); % Creates a matrtix of only digital Images containg Zeros 
 
IVal_1= IVec_1; % Makes a copy of the above matrix for manipulation later on. 
IVal_0= IVec_0;

% Next we find the height of the matrix above (i.e the matrix above contains all the images.
% The "height" function runs the number of rows in that matrix which is equal to the number of images in said matrix. 
% The "height" function seems to be version dependent so was replaced by
% the "size" function.

% Rows_1 = round(height(IVec_1)*0.7);
% Rows_0 = round(height(IVec_0)*0.7);

Rows_1 = round(size(IVec_1,1)*0.7); % Obtain a value for 70% of the rows (i.e 888)
Rows_0 = round(size(IVec_0,1)*0.7);% (i.e 1086)

% Next we create a vector from 1 to 70% of the total number of images.
A = linspace(1,Rows_1,Rows_1);% Linespace creates a vector starting at 1 and ending at 888:
B = linspace(1,Rows_0,Rows_0); % Linespace "" at 1 and endind a 1086 


ImTraining_1 = IVec_1(A,:);% From the matrix contain all the ones we add the first 70% of these values to a new matrix which is the training data of ones
ImTraining_0 = IVec_0(B,:);% From the matrix contain all the zeros we add the first 70% of these values to a new matrix which is the training data of zeros

IVal_1(A,:) = []; % We then use the copy of the matrix containing all the ones that was created above and remove the first 70% of the rows
IVal_0(B,:)= []; 


% Now that the training images have been seperated we need to seperate the prediction values i.e output t  
% This bits a bit redundant as you can probabliy just create a vector equal
% to the height of the image vector that contains only ones or zeros. 

% First create a subvector of t which only contains the t outputs for ones outputs  
t_1 =  t(Ones_Index,:);
% Next create a vector of the first 70% of the one outputs i.e training predictions 
t_Training_1 = t_1(A,:);
% Repeat the method above
tVal_1 = t_1; % Create a copy of the predictions containg all the t = 1 outputs
tVal_1(A,:) =[]; % Remove the first 70%

% Repeat the proccess for the Zeros 
% Seperating the first creating a sub vector of t which only contains the t outputs for zero outputs  
t_0 =  t(Zero_Index,:);
t_Training_0 = t_0(B,:);
tVal_0 = t_0;
tVal_0(B,:)= [];

% Now we combine the vectors containing all the first 70% of the t= 1s and
% t = 0s into one training matrix and combine the remain 30% of the t
% predictions 

% Training/Validation Ouputs for both t=0 and t=1
t_training = [t_Training_0;t_Training_1];
tVal =[tVal_0;tVal_1];

%Training/Validation Inputs
x_training = [ImTraining_0;ImTraining_1];
xVal = [IVal_0;IVal_1];


%% Section 1: Supervised Learning Using Least Squares
% dim(theta)=257:
N_training=length(x_training); % 1974 training images in training data set

% Each column of the 1 x 257 vector is a feature 
ux_ERM = [ones(1,N_training)',x_training];% Feature Vector 1974 x 257; Where the first column of the feature vector is a row of ones

% "The optimal parameter vector effectively "summarizes" the traing set via 
% a linear combination of the feature vectors of the data points with 
% coefficients given by the correspoding targets" [Chapter 4]
thERM=LSsolver(ux_ERM,t_training);

N_validation = length(xVal);% 846 validation images in validation data set
ux_val_ERM = [ones(1,N_validation)',xVal];% Validation Feature Vector 846 x 257

% Generating the predictions t_hat by mutiplying the feature vector u(x) by the
% learning paramters theta (summarizes the relationship to go from x to t) 

% Week 4.2: Problem 4: LDRERM t= X*thERM 
t_model_prediction = ux_ERM*thERM; % Prediction i.e t_hat % Plot with training Data X 

% Note: The prediction were generated using the training set, not the validation
% set as advised during tutorial session.  

% 2: Calculate and Display the training loss and the validation loss:

% Caculating the Training Loss (Chapter 4: slide 35/109)
% Training LD(Thetha) = 1/N* Sum(n->N) (tn - thethaT*u(xn))^2
% Where tn is the true labels of the training data (i.e the real outputs of
% the training data)

% The quadractic Training Loss is the (true_label (t) of an input (x) minus
% the model_prediction (t) for the same input (x)) squared, for all 
% input-label pairs,  divided by the total number of data points
% t_model_predictions = ux_LR*thERM 
LDRERM=1/N_training*norm(t_training-ux_ERM*thERM)^2; % Derived from Week 4.2: Problem 

traininglossLS_257 = LDRERM % Loss for Logistic Regression under quadratic loss for training set

% CLoss for Logistic Regression under quadratic loss for validation set
validationlossLS_257 = 1/N_validation *norm(tVal-ux_val_ERM*thERM)^2

% 3: Consider a vector of feature x(un)=[1,xT]T where xT contians the first
% 9 features. 

% dim(theta)=10:
ux_9_ERM = ux_ERM; % Make a copy of the Feature Vector u(x)  
% width function originally used ,however, it is not included in the list
% of functions below which may lead to compatability issues.
% C = linspace(11,width(ux_LR),247);
C = linspace(11,size(ux_ERM,2),247); % Generate a vector from 11 to 257
ux_9_ERM(:,C) = []; % Remove columns  11 onwards not 9 because XVal = [1,246] we need to inlude the vector of ones at the start

thERM_9=LSsolver(ux_9_ERM,t_training); % Generating Model paramaters for first 9 features. 

% Note: Once again using training data not validation data as advised
% during tutorial sessions.
t_predictions_9 = ux_9_ERM * thERM_9;% Generating model prediction t, using the first 9 feature vector, and the first 9 learning parameters. 

% Validation Loss for 9 entry feature vector: 
ux_9_val_ERM = ux_val_ERM;
ux_9_val_ERM(:,C) = [];
traininglossLS_10 = 1/N_training*norm(t_training-ux_9_ERM*thERM_9)^2
validationlossLS_10 = 1/N_validation*norm(tVal-ux_9_val_ERM*thERM_9)^2

figure(1); % Ploting ERM Predictions
x_axis = linspace(1,1974,1974); % Generating a vector equal to the number of data points
plot(x_axis,t_predictions_9,'-r','LineWidth',0.5); % Plotting predictions using 9 entry feature vector (i.e 1 x 10)
hold on
plot(x_axis,t_model_prediction,'g','LineWidth',1);% Plotting predictions using 256 entry feature vector (i.e 1 x 257)
hold on
plot(x_axis,t_training,'--k','LineWidth',1); % Plotting predictions using 256 entry feature vector (i.e 1 x 257)
hold on
xlabel('$N$','Interpreter','latex');
ylabel('t','Interpreter','latex')
title('Section 1: ERM, with quadratic loss','Interpreter','latex','FontSize',12); grid minor;
legend('$\hat{t} (x_{n}|\theta^{ERM})$','$$\hat{t} (x_{n}|\theta^{ERM})= (\theta^{ERM})^{T}u(x_{n}) $$','$t_{n}$','Interpreter','latex','FontSize',12,'Location','southeast');

% 4: Comparision between Red and Green Line
display(strvcat({'Section1: The prediction with the longer and shorter feature vectors',...
         'is different because the shorter feature vector aims to capture the key properties',...
         'of the distribution with fewer model parameters. Using only nine key features, the model',...
         'is obviously unable to capture the distribution of the data as well as the longer feature vector.',...
         'Obtaining fewer correct predictions overall, in addition to higher validation losses and training losses',...
         'across the data set. The shorter feature vector is still, however, able to map the distribution of the data',...
         'relatively well. Therefore it can be concluded that not all the features in the longer feature vector are required',...
         'to truly map the distribution of the data for the purpose of classification.'}))

%% Section 2: Supervised Learning Using Logistic Regression 
%1: Impliment Logistic Regression
I = 50;% Number of Iterations % 50
gamma = 0.14;% Learning Rate % 
S = 20; % Mini Batch Size % 20
u_x_LR = [ones(1,1974)',x_training]; % Vector of Features for Logistic Reggression
%th_LR = ones(257,1); % Initalize Vector of ones for thetha
th_LR = 0.1.*ones(257,1); % Initalize Vector of ones for thetha
training_logloss_LR_50 = zeros(I,1); % Creating a variable to store total training log-loss for 50 interations
validation_logloss_LR_50 = zeros(I,1);% Creating a variable to store total validation log-loss for 50 interations
training_log_loss= zeros(1974,1);% Creating a variable to store total training log-loss for each data point
validation_training_log_loss= zeros(846,1);% Creating a variable to store total validation log-loss for each data point


Val_logloss_LR_50_test = zeros(I,1); % Creating a variable to store total training log-loss for 50 interations
Val_log_loss_test= zeros(1974,1);

display(strvcat({'Section 2: I have chosen S=32 and gamma = 0.14 because smaller batch, ', ...
                 'sizes update more frequently, which allows the algorithm to explore more,', ...
                 'at a faster rate, especially with larger learning rates. Furthermore, smaller', ...
                 'batch sizes also result in lower generalisation error, and inject more ', ...
                 '"noise" into the estimate, which is a form of regularisation. Typically, a ', ...
                 'small batch size ranges  from 1 to 100; a batch size of 32 is,  a good', ...
                 'default value[1]. The second hyper-parameter is the learning, rate gamma.', ...
                 'If gamma is set too high the average loss will increase; the optimal ', ...
                 'learning rate is usually  close to (by a factor of 2) to the largest learning ', ...
                 'rate that does not cause divergence of the training criterion[1]. A heuristic', ...
                 'approach for setting the learning rate is to initially select a large learning ', ...
                 'rate, if the training criterion diverges, try again with a rate 3 times smaller ', ...
                 'until no divergence is observed[1]. A number of analytical methods exist ', ...
                 'for the selection of the learning rate; the Murone-Robbins condition ', ...
                 'states that convergence to a stationary point is guaranteed if the learning ', ...
                 'rate is selected such that gamma(i) = 1/i^(alpha) for 0.5< alpha<1.', ...
                 'Following the Murone-Robbins condition, a variable gamma for 50 interation would vary from 0.02 to 0.5', ...
                 'therefore, a fixed learning rate of 0.2 is selected, as it performed well within the aforemtioned range.', ...  
                  }))
% Week 8.1 :Problem 3
for i = 1:I
    
    %Monroe-Robbins Criterion for Variable Learning Rate
    %gamma = 1/(i^0.5)
    
    % Performing Stocastic Gradient Decent
    ind = mnrnd(1,1/1974*ones(1974,1),S)*[1:1974]'; %Generate one random index
    g=zeros(257,1);% Initialising 
    for s=1:S
       % Calcaulating the Sum the Mean Error x the feature vector
        g=g-1/S*(1/(1+exp(-th_LR'*u_x_LR(ind(s),:)'))-t_training(ind(s)))*u_x_LR(ind(s),:)';
    end
    % Obtaining the next iterate via SGD
    th_LR=th_LR+gamma*g;
    
    %Extra
    %dv = zeros(1, 1974) ;% Initiallising Decision Variables(i.e thetha^T.u(x))
    %p = zeros(1, 1974) ;% Predictions (i.e sigmoid(thetha^T.u(x))
    
    %1:Plot the training log-loss as a function of the number of iteration:
    for j = 1:1974  
    % Extra: Plotting Soft Predictor vs. Decision Variable:
    %dv(j) = u_x_LR(j,:)*th;% Decision Variables for iteration i: theta^T*u(x)
    %p(j) = sigmoid(u_x_LR(j,:)*th);% Soft Predictor for iteration i: p(t=1|x,thetha) = sigmoid(theta^T*u(x)) 
    % Chapter 7: Page 41
    %training_log_loss(j) = log(1+exp((-(2*t_training(j)-1).*(u_x_LR(j,:)*th_LR))));% Log-Loss for Each data point 
    training_log_loss(j) = -log(sigmoid ((2*t_training(j)-1).*u_x_LR(j,:)*th_LR ));  % Using Sigmoid Function        
    end 

    % Extra:Plotting Soft Predictor vs. Decision Variable:
    %clf(figure(9))
    %figure(9); hold on; title('Section 2: Soft-Prediction vs. Decision Variable');
    %plot(dv, p,'*', 'LineWidth',2);
    %xlabel('$\sigma (\theta^{T} u(x))$', 'Interpreter', 'latex', 'FontSize',12);
    %ylabel('$\theta^{T} u(x)$', 'Interpreter', 'latex', 'FontSize',12);
    
    % Extra: Plotting the Soft Prediction Vs. Data Point
    %clf(figure(10))
    %figure(10); hold on; title('Section 2: Soft-Prediction vs. Data Point');
    %plot(linspace(1,1974,1974), p, '*', 'LineWidth',2);
    %xlabel('$N$', 'Interpreter', 'latex', 'FontSize',12)
    %ylabel('$\sigma (\theta^{T} u(x))$', 'Interpreter', 'latex', 'FontSize',12);
    
    training_logloss_LR_50(i) = 1/1974*sum(training_log_loss); % Computing Training Log-Loss for 50 iterations
    %2:Plot the validation log-loss as a function of the number of iteration:
    for j = 1:846  
    validation_training_log_loss(j) = -log(sigmoid((2*tVal(j)-1).*ux_val_ERM(j,:)*th_LR ) ); % Using Sigmoid Function
    %validation_training_log_loss(j) = log(1+exp((-(2*tVal(j)-1).*(ux_val_ERM(j,:)*th_LR))));% Validation Log-Loss for Each data point
    end
    validation_logloss_LR_50(i) = 1/846*sum(validation_training_log_loss); %Computing Validation Log-Loss for 50 iteration
    
end 
% Ploting both the training and validation log loss as a function of
% iterations 
figure(2); hold on; title('Section 2: Logistic Regression, Log-Loss');
plot([1:1:I],training_logloss_LR_50,'-k', 'LineWidth',2);
hold on 
plot([1:1:I],validation_logloss_LR_50,'-r', 'LineWidth',2);
xlabel('N', 'Interpreter', 'latex', 'FontSize',12);
ylabel('Log-Loss', 'Interpreter', 'latex', 'FontSize',12);
legend('Training Loss','Validation Loss','Interpreter','latex','FontSize',12);


%% Section 3: Unsupervised Learning Using PCA 
N = 1974; % Number of Data Points in Data-Set 
% Performing PCA: Week 10.1: Problem 2
m = mean(x_training); % Emperical Mean of the data  
%x_training_mean = x_training; % - m;
x_training_mean = x_training - ones(N,1)*m; % Removing the emperical Mean from the data
% x_training_mean = x_training; % Better recounstruction results were achived without the removal of the mean 
[W D] = eig(1/N*x_training_mean'*x_training_mean); % Computing the Eigvalues and Eigen Vectors

w1 = W(:,1); % Extracting the first three eigen vectors which corresponsed to the first three principle components
w2 = W(:,2);
w3 = W(:,3);

% Displaying the Principle Components as 16x16 pixel Images:
%imv1 = reshape(w1, [16, 16])';
%imv2 = reshape(w2, [16, 16])';
%imv3 = reshape(w3, [16, 16])';

%1: Plot the first three principle components as grey scale images
% Displaying the Model Parameters as Gray Scale Images
figure(3);sgtitle('Section 3: PCA most significant Eigen vectors');

%subplot(1,3,1); imagesc(imv1); colormap(gray); subtitle('Most significant');
%subplot(1,3,2); imagesc(imv2); colormap(gray); subtitle('Second significant');
%subplot(1,3,3); imagesc(imv3); colormap(gray); subtitle('Third significant');

subplot(1,3,1); show_vec_as_image16x16(10*w1); title('Most significant'); % Images Scaled to Increase 
subplot(1,3,2); show_vec_as_image16x16(10*w2); title('Second significant');
subplot(1,3,3); show_vec_as_image16x16(10*w3); title('Third significant');

%2: Reconstuct the first image of the training set using PCA
% Reconstruction of the first image from the training set using the first three principle components 
z(1) = w1'*x_training(1,:)'; % with first principle componenet
z(2) = w2'*x_training(1,:)'; % with second principle componenet
z(3) = w3'*x_training(1,:)'; % with third principle componenet
im1reconstv = z(1)*w1 ; % Reconstructing image M=1
im2reconstv = z(1)*w1 +z(2)*w2; % Reconstructing image M=2
im3reconstv = z(1)*w1 +z(2)*w2+ z(3)*w3 ; % Reconstructing image M=2

% im1reconst = reshape(im1reconstv,[16,16])';% Reconstructed image
% img1 = reshape(x_training(1,:), [16,16])'; % Original Image
img1 = x_training(1,:);

figure(4);sgtitle('Section 3: Estimating using PCA, M = number of significant components');
%subplot(1,2,1); imagesc(img1); colormap(gray); subtitle('First Training Image ');
%subplot(1,2,2); imagesc(im1reconst); colormap(gray); subtitle('Reconstructed Image');

subplot(2,2,1); show_vec_as_image16x16(img1);          title('First training set image');
subplot(2,2,2); show_vec_as_image16x16(im1reconstv);   title('Reconstracting using M=1 most significant components');
subplot(2,2,3); show_vec_as_image16x16(im2reconstv);   title('Reconstracting using M=2');
subplot(2,2,4); show_vec_as_image16x16(im3reconstv);   title('Reconstracting using M=3');


%3: Plot the contibutions of the first three components 
% Contribution of Principle Components for all the images 
z0(1,:) = w1'*x_training(1:1086,:)'; % Images of Zeros
z0(2,:) = w2'*x_training(1:1086,:)';
z0(3,:) = w3'*x_training(1:1086,:)';

figure(5); sgtitle('Significant PCA components over all training set');
plot3(z0(1,:),z0(2,:),z0(3,:),'o', 'MarkerSize', 4);% Plotting zeros as "o"
hold on 
z1(1,:) = w1'*x_training(1087:1974,:)';% Images of Ones as "x" 
z1(2,:) = w2'*x_training(1087:1974,:)';
z1(3,:) = w3'*x_training(1087:1974,:)';
% 3 Dimeninsional Plot
plot3(z1(1,:),z1(2,:),z1(3,:),'x', 'MarkerSize', 4);
xlabel('$z_1$', 'Interpreter', 'latex', 'FontSize',12);
ylabel('$z_2$', 'Interpreter', 'latex', 'FontSize',12);
zlabel('$z_3$', 'Interpreter', 'latex', 'FontSize',12);
legend('Zeros','Ones','Interpreter','latex','FontSize',12);


%% Section 4: Unsupervised Learning and Supervised Learning
z_PCA=[z0';z1'];
u_x_PCA = [ones(1, 1974)',z_PCA(:,1),z_PCA(:,2)]; % Constructing three-dimensional feature vector 
th_LR_PCA = 0.1.*ones(3,1); % Applying ERM Solver
for i=1:I
    ind=mnrnd(1,1/N*ones(N,1),S)*[1:N]'; %generate S random indices
    g=zeros(3,1);
    for s=1:S
        g=g-1/S*(1/(1+exp(-th_LR_PCA'*u_x_PCA(ind(s),:)'))-t_training(ind(s)))*u_x_PCA(ind(s),:)';
    end
    th_LR_PCA=th_LR_PCA+gamma*g;
    training_loss_PCA_LR(i)= 1/N*sum(log(1+exp(-(2*t_training-1).*(u_x_PCA*th_LR_PCA))));
    
end
figure(6); hold on; title('Section 5: Logistic Regression');
% code for plot here
plot([1:I],training_loss_PCA_LR,'LineWidth',2)
xlabel('N', 'Interpreter', 'latex', 'FontSize',12);
ylabel('Training Log-Loss', 'Interpreter', 'latex', 'FontSize',12);
% complete the insight:

display(strvcat({'Section 4: Comparing with the solution in Section 2, I conclude that, ', ...
                 'comparatively, using PCA to construct the feature vector, before the ', ...
                 'application of logistic regression resulted in a lower initial training log-', ...
                 'loss.  Using the PCA feature vector, however, also resulted in a lower ', ...
                 'convergence rate; the PCA-Logistic Regression composite algorithm ', ...
                 'required far more iterations to achieve comparable levels of performance ', ...
                 'in terms of training log-loss.  It should be noted that the empirical mean ', ...
                 'of the data was subtracted from said data to center it around the origin; a ', ...
                 'key part of the inductive bias of PCA. However, in terms of image ', ...
                 'reconstruction in this instance, simply applying PCA without removal of ', ...
                 'the empirical mean resulted in more recognizable images. Furthermore, ', ...
                 'without the removal of the empirical mean, faster convergence rates, and ', ...
                 'lower initial log-losses were observed. The initial theta value,  mini-batch ', ...
                 'size, and learning rate have been optimised for logistic regression in ', ...
                 'section 2, using the same parameters on the PCA-Logistic Regression ', ...
                 'composite algorithm would intuitively result in suboptimal performance as  ', ...
                 'hyperparameters are generally non-transferable and have not been ', ...
                 'optimised to fit the application; the hyperparameters must therefore be ', ... 
                 'optimised for the hybrid algorithm to achieve comparable levels of ', ... 
                 'performance. PCA is an unsupervised learning technique wherein the ', ... 
                 'algorithm is given no prior labels for the data, the addition of logistic ', ... 
                 'regression effectively makes the following algorithm a semi-supervised ', ... 
                 'learning problem.  Unsupervised problems generalize relatively well, ', ... 
                 'however in this instance as the data is already labeled a fully supervised ', ... 
                 'solution may be more appropriate .', ... 
                  }));

%% References
% [1] Bengio, Y., 2012. Practical recommendations for gradient-based training of deep architectures. In Neural networks: Tricks of the trade (pp. 437-478). Springer, Berlin, Heidelberg.
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A list of functions you may want to familiarise yourself with. Check the help for full details.

% + - * / ^ 						% basic operators
% : 								% array indexing
% * 								% matrix mult
% .* .^ ./							% element-wise operators
% x.' (or x' when x is surely real)	% transpose
% [A;B] [A,B] 						% array concatenation
% A(row,:) 							% array slicing
% round()
% exp()
% log()
% svd()
% max()
% sqrt()
% sum()
% ones()
% zeros()
% length()
% randn()
% randperm()
% figure()
% plot()
% plot3()
% title()
% legend()
% xlabel(),ylabel(),zlabel()
% hold on;
% grid minor;

