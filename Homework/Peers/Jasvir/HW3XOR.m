% Pocket Algorithms and Perceptron Learning on XOR
% XOR TruthTable
%x1 x2  y
%0  0   0
%1  0   1
%0  1   1
%1  1   0
clc;
hold off;
clf
clear
% visualizing AND function
axis([-1 3 -1 3]) % setting the axis limit
hold on
plot(0,0,'r*') % plot (0,0) with Red circle
plot(0,1,'go')% plot (0,1) with Green circle
plot(1,0,'go')% plot (1,0) with Green circle
plot(1,1,'r*')% plot (1,1) with Red circle

x0=[1 1 1 1]; % this is bias and it is always ON
x1=[0 1 0 1]; % input 1
x2=[0 0 1 1]; % input 2
target= [0 1 1 0]; % output
x=[x0;x1;x2]; %concatenating input data including the bias
% now we have four different data points (0,0),(0,1),(1,0) and (1,1)
disp('input data=');
disp(x); %display data points
disp('target values=');
disp(target);%display target values

w=1-2*rand(3,1); % initializing the weights vector

disp('Initial random weight vector is=');
disp(w);

w_pocket=w; %Pocket weight vector; default set to same as initial PLA weight

disp('Initial pocket weight vector is=');
disp(w);

g=w'*x;% activation function
disp('activation function=');
disp(g);

m=-w(2)/w(3); % slope of the line
b=-w(1)/w(3); % intercept
%now setting the graph between -1 and 1
px1=-1;
px2=1;

%As per equation y=mx+b, finding py1 and py2
py1=m*px1+b;
py2=m*px2+b;
plot([px1,px2],[py1,py2],'r');% plotting a graph in Red
max_iterations = 500;
n=length(g);
pw_in_sample_error = get_pocket_weight_in_sample_error(x,w_pocket, target);
disp('pocket weight in-sample error');
disp(pw_in_sample_error);
run=0;%count the number of true decisions during training
 converge = 0;
 while converge==0 
     for epoch=1:max_iterations %At each iteration, we will pocket the weights vector which gives minimum error
         sample_error_sum = 0;
             for i=1:n % looping through each row
                 g = w'*x; % activation function
                 if g(i) >= 0
                     y_hat = 1;
                 else
                     y_hat = 0;
                 end
%                  if isequal(y_hat,target(i))
%                       run=run+1;                   
%                  end 
                 if(target(i)~= y_hat)  % if target output is not equal to predicted output
                     converge = 0; % not converged yet
                         if(target(i) == 1 && y_hat == 0) 
                             % case 1: if target output is 1 but predicted output is 0, then update the weight by w=w+x
                             larray = x*0.01;
                             w = w + larray(:,i); %w= w+x(:,i); %  new weight = old weight + all rows of ith column
                             g= w'*x; %activation function
                             %sample_error_sum = sample_error_sum + 1;
                               break;
                          elseif (target(i) == 0 && y_hat ==1)
                             % case 2: if target output is 0 but predicted output is 1, then update the weight by w=w-x
                             larray = x*0.01;
                             w = w - larray(:,i);  %w= w-x(:,i); %  new weight = old weight - all rows of ith column
                             g= w'*x; 
                            %activation function
                             %sample_error_sum = sample_error_sum + 1;
                               break;
                         end
                    else
                          % if target output is equal to predicted output, then it is converged             
                         converge = 1;
                 end
             end
             g_logic = double(g>=0);
             
             sample_error_sum = sum(target ~= g_logic);
             if(epoch > 1)  
                 in_sample_error = sample_error_sum/n;
                 %disp('new in-sample error=');
                 %disp(in_sample_error);
                 if in_sample_error < pw_in_sample_error
                     disp('old pw act');
                     disp(w_pocket' * x);
                     disp('old error');
                     disp(pw_in_sample_error);
                     disp('next minimum in-Sample error:')
                     disp(in_sample_error);
                     w_pocket = w;
                     pw_in_sample_error = in_sample_error;
                     disp('new pocket weight vector=');
                     disp(w_pocket'*x);
                 end
             end 
          if(epoch == max_iterations)
              converge=1;
              disp('reached maximun number of iterations');
          end
     end
      
 end 

g= w_pocket' *x;% activation function
disp('activation function=');
disp(g);
disp(pw_in_sample_error)
disp('final pocket weight:');
disp(w_pocket);
m=-w_pocket(2)/w_pocket(3);% new slope of the line
b=-w_pocket(1)/w_pocket(3);% new intercept
%As per equation y=mx+b, finding py1 and py2
py1=m*px1+b;
py2=m*px2+b;
plot([px1,px2],[py1,py2],'b'); %plotting final graph with new y1 and y2
% blue line is the new classification line

function pw_in_sample_error = get_pocket_weight_in_sample_error(x, w_pocket, target)
    sample_error_sum = 0;
    n = length(x);
    for i=1:n % looping through each column
             g = w_pocket'*x; % activation function
             if g(i) >= 0
                 y_hat = 1;
             else
                 y_hat = 0;
             end
             if(target(i)~= y_hat)
                 sample_error_sum = sample_error_sum+1;
             end
    end
    pw_in_sample_error = sample_error_sum/n;
end