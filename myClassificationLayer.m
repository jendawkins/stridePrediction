classdef myClassificationLayer < nnet.layer.ClassificationLayer
        
    properties
        % (Optional) Layer properties.

        % Layer properties go here.
        Weights
    end
 
    methods
        function layer = myClassificationLayer(Weights, name)           
            % (Optional) Create a myClassificationLayer.

            % Layer constructor function goes here.
            layer.Weights = Weights;
            layer.Name = name;
            layer.Description = 'Custom Cross Entropy';
        end

        function loss = forwardLoss(layer, Y, T)
            % Return the loss between the predictions Y and the 
            % training targets T.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     ? Predictions made by network
            %         T     ? Training targets
            %
            % Output:
            %         loss  - Loss between Y and T

            loss = -sum(T.*log(Y.*layer.Weights));
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % Backward propagate the derivative of the loss function.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     ? Predictions made by network
            %         T     ? Training targets
            %
            % Output:
            %         dLdY  - Derivative of the loss with respect to the predictions Y
            dLdY = -(T./layer.Weights*Y)/N;
            % Layer backward loss function goes here.
        end
    end
end