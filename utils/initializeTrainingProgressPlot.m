function [plotter,trainTransPlotter,valTransPlotter, ...
  trainRotPlotter, valRotPlotter] = initializeTrainingProgressPlot()
% Plot the loss, training accuracy, and validation accuracy.
figure

% Loss plot
subplot(3,1,1)
plotter = animatedline;
xlabel("Iteration")
ylabel("Loss")

% Trans mae plot
subplot(3,1,2)
trainTransPlotter = animatedline('Color','b');
valTransPlotter = animatedline('Color','g');
legend('Training Translation MAE','Validation Translation MAE','Location','northwest');
xlabel("Iteration")
ylabel("MAE")

% Rot mae plot
subplot(3,1,3)
trainRotPlotter = animatedline('Color','b');
valRotPlotter = animatedline('Color','g');
legend('Training Rotation MAE','Validation Rotation MAE','Location','northwest');
xlabel("Iteration")
ylabel("MAE")
end