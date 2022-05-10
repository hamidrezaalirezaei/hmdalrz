% function [C, epsilon, gamma] = pso (ClusteringData)

%% Problem Definition

CostFunction=@main;   % Cost Function

nVar=3;                % Number of Decision Variables

VarSize=[1 nVar];       % Size of Decision Variables Matrix

VarMin= 100;            % Lower Bound of Decision Variables
VarMax= 1000;            % Upper Bound of Decision Variables

VelMax=(VarMax-VarMin)/20;

%% PSO Parameters

MaxIt=50;          % Maximum Number of Iterations

nPop=50;            % Swarm Size

% w=1;                % Inertia Weight
% wdamp=0.99;         % Inertia Weight Damping Ratio
% c1=2;               % Personal Learning Coefficient
% c2=2;               % Global Learning Coefficient

phi1=2.05;
phi2=2.05;
phi=phi1+phi2;
chi=2/(phi-2+sqrt(phi^2-4*phi));

w=chi;              % Inertia Weight
wdamp=1;            % Inertia Weight Damping Ratio
c1=chi*phi1;        % Personal Learning Coefficient
c2=chi*phi2;        % Global Learning Coefficient

%% Initialization

empty_particle.Position=[];
empty_particle.Velocity=[];
empty_particle.Cost=[];
empty_particle.Best.Position=[];
empty_particle.Best.Cost=[];

particle=repmat(empty_particle,nPop,1);

GlobalBest.Cost=inf;

for i=1:nPop
    Cmin=0.1;
    Cmax=20000;
    C=unifrnd(Cmin,Cmax);
    epsilonmin=1e-3;
    epsilonmax=1;
    epsilon=unifrnd(epsilonmin,epsilonmax);
    gammamin=1e-3;
     gammamax=500;
     gamma=unifrnd( gammamin, gammamax);
    particle(i).Position=[C, epsilon, gamma];
%     particle(i).Position=unifrnd(VarMin,VarMax,VarSize);
    particle(i).Velocity=zeros(VarSize);
    
    particle(i).Cost=CostFunction( particle(i).Position);
    
    particle(i).Best.Position=particle(i).Position;
    particle(i).Best.Cost=particle(i).Cost;
    
    if particle(i).Best.Cost<GlobalBest.Cost
        GlobalBest=particle(i).Best;
    end
    
end

BestCost=zeros(MaxIt,1);


%% PSO Main Loop

for it=1:MaxIt
    
    for i=1:nPop
        
        particle(i).Velocity=w*particle(i).Velocity ...
                            +c1*rand*(particle(i).Best.Position-particle(i).Position) ...
                            +c2*rand*(GlobalBest.Position-particle(i).Position);
        
        particle(i).Velocity=min(max(particle(i).Velocity,-VelMax),+VelMax);
        
        particle(i).Position=particle(i).Position+particle(i).Velocity;
        
        flag=(particle(i).Position<VarMin | particle(i).Position>VarMax);
        particle(i).Velocity(flag)=-particle(i).Velocity(flag);
        
        particle(i).Position=min(max(particle(i).Position,VarMin),VarMax);
        
        particle(i).Cost=CostFunction( particle(i).Position);
        
        if particle(i).Cost<particle(i).Best.Cost
            
            particle(i).Best.Position=particle(i).Position;
            particle(i).Best.Cost=particle(i).Cost;
            
            if particle(i).Best.Cost<GlobalBest.Cost
                GlobalBest=particle(i).Best;
            end
            
        end
        
    end
    
    BestCost(it)=GlobalBest.Cost;
    
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
    w=w*wdamp;
    
end

%% Results
epsilon = GlobalBest.Position (2)
C = GlobalBest.Position (1)
gamma = GlobalBest.Position (3)
RMSE = GlobalBest.Cost

figure;
plot(BestCost,'LineWidth',2);
xlabel('Iteration');
ylabel('Best Cost');
