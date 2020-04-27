classdef gsLayer < nnet.layer.Layer
    % A simple Matlab implementation of GraphSage.
    % Copyright (c) 2020, Chong WU All rights reserved.
    % 
    % Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
    % 
    % Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    % 
    % Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    % 
    % Neither the name of City University of Hong Kong nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
    % 
    % THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    %
    % Reference:
    % Inductive Representation Learning on Large Graphs. W.L. Hamilton, R. Ying, and J. Leskovec arXiv:1706.02216 [cs.SI], 2017. 
    properties
        % (Optional) Layer properties.
        adjMat
        featMat
        embed_dim
        feat_dim
        numsample
        % Layer properties go here.
    end
    
    properties (Learnable)
        % Layer learnable parameters
            
        % Scaling coefficient
        Alpha
        
    end
    
    methods
        function layer = gsLayer(adjMat, featMat,embed_dim, feat_dim, numsample, name) 
            % layer = gsLayer(adjMat, featMat, embed_dim, feat_dim, agg, name) creates a GSA layer
            % with numChannels channels, adjacent matrix, feature matrix,feat_dim, embed dim, aggregator and specifies the layer name.

            % Set layer name.
            layer.Name = name;
            
            % Get adjacent matrix
            layer.adjMat = adjMat;
            
            % Get feature matrix
            layer.featMat = featMat;
            
            % Set embedding dim
            layer.embed_dim = embed_dim;
            
            % Set feature dim
            layer.feat_dim = feat_dim;       
           
            % Set sample rate
            layer.numsample = numsample;   
            
            % Set layer description.
            layer.Description = "GraphSageAggregation Layer with " + embed_dim + " embedding dimensions";
        
            % Initialize scaling coefficient.
            layer.Alpha = rand([layer.embed_dim layer.feat_dim*2  1]); 
        end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
            nodes = X;
            
            neighbors = layer.adjMat(nodes,:);
            
            for i = 1:length(neighbors)
                if length(neighbors{i}) > layer.numsample
                    ind = randperm(length(neighbors{i}),layer.numsample);
                    neighbors{i} = neighbors{i}(ind);
                end
                
            end
            
            neighbors_all = cell(size(neighbors,1),1);
            neigh_feats = dlarray(zeros(size(layer.featMat,2), size(neighbors,1),1,'like',single(1)));
            self_feats = dlarray(zeros(size(layer.featMat,2), size(neighbors,1),1,'like',single(1)));

            for i = 1:size(neighbors,1)
                temp = [];
                for j = 1:length(neighbors{i})
                    temp = union(temp,layer.adjMat{neighbors{i}(j),:});
                end
                neighbors_all{i} = temp;
                
                dlY = layer.featMat(neighbors_all{i},:);
                
                temp = mean(dlY,1);
                neigh_feats(:,i,1) = temp;
            end
            
            self_feats(:,:,1) = layer.featMat(nodes,:)';
            
            combined = cat(1,self_feats,neigh_feats);
            Z = relu(layer.Alpha*combined);
            
        end
    end
end