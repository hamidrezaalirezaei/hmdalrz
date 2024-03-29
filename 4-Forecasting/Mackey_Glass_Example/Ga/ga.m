clc;
clear;
close all;

%% Problem Definition

global NFE;
NFE=0;

CostFunction=@choatic_MG;     % Cost Function

nVar=3;             % Number of Decision Variables

VarSize=[1 nVar];   % Decision Variables Matrix Size

Cmin=1e-0;
Cmax=1e+3;

epsilonmin=1e-4;
epsilonmax=1e-1;

arg1min=1e-2;
arg1max=1e+1;

VarMin = [Cmin, epsilonmin, arg1min];
VarMax = [Cmax, epsilonmax, arg1max];
% VarMin=-10;         % Lower Bound of Variables
% VarMax= 10;         % Upper Bound of Variables


%% GA Parameters

MaxIt=100;      % Maximum Number of Iterations

nPop=50;        % Population Size

pc=0.8;                 % Crossover Percentage
nc=2*round(pc*nPop/2);  % Number of Offsprings (Parnets)

pm=0.3;                 % Mutation Percentage
nm=round(pm*nPop);      % Number of Mutants

gamma=0.05;

mu=0.02;         % Mutation Rate

ANSWER=questdlg('Choose selection method:','Genetic Algorithm',...
    'Roulette Wheel','Tournament','Random','Roulette Wheel');

UseRouletteWheelSelection=strcmp(ANSWER,'Roulette Wheel');
UseTournamentSelection=strcmp(ANSWER,'Tournament');
UseRandomSelection=strcmp(ANSWER,'Random');

if UseRouletteWheelSelection
    beta=8;         % Selection Pressure
end

if UseTournamentSelection
    TournamentSize=3;   % Tournamnet Size
end

pause(0.1);

%% Initialization

empty_individual.Position=[];
empty_individual.Cost=[];

pop=repmat(empty_individual,nPop,1);
tic
for i=1:nPop
%     Cmin=1e-0;
%     Cmax=1e+2;
    C=unifrnd(Cmin,Cmax);
%     epsilonmin=1e-3;
%     epsilonmax=1e-2;
    epsilon=unifrnd(epsilonmin,epsilonmax);
%     sigma1min=1e+0;
%      sigma1max=1e+1;
     arg1=unifrnd( arg1min, arg1max);
    % Initialize Position
    pop(i).Position=[C,epsilon,arg1];
    
    % Evaluation
    pop(i).Cost=CostFunction(pop(i).Position);
    
end

% Sort Population
Costs=[pop.Cost];
[Costs, SortOrder]=sort(Costs);
pop=pop(SortOrder);

% Store Best Solution
BestSol=pop(1);

% Array to Hold Best Cost Values
BestCost=zeros(MaxIt,1);

% Store Cost
WorstCost=pop(end).Cost;

% Array to Hold Number of Function Evaluations
nfe=zeros(MaxIt,1);


%% Main Loop

for it=1:MaxIt
    
    % Calculate Selection Probabilities
    if UseRouletteWheelSelection
        P=exp(-beta*Costs/WorstCost);
        P=P/sum(P);
    end
    
    % Crossover
    popc=repmat(empty_individual,nc/2,2);
    for k=1:nc/2
        
        % Select Parents Indices
        if UseRouletteWheelSelection
            i1=RouletteWheelSelection(P);
            i2=RouletteWheelSelection(P);
        end
        if UseTournamentSelection
            i1=TournamentSelection(pop,TournamentSize);
            i2=TournamentSelection(pop,TournamentSize);
        end
        if UseRandomSelection
            i1=randi([1 nPop]);
            i2=randi([1 nPop]);
        end

        % Select Parents
        p1=pop(i1);
        p2=pop(i2);
        
        % Apply Crossover
        [popc(k,1).Position popc(k,2).Position]=...
            Crossover(p1.Position,p2.Position,gamma,VarMin,VarMax);
        
        % Evaluate Offsprings
        popc(k,1).Cost=CostFunction(popc(k,1).Position);
        popc(k,2).Cost=CostFunction(popc(k,2).Position);
        
    end
    popc=popc(:);
    
    
    % Mutation
    popm=repmat(empty_individual,nm,1);
    for k=1:nm
        
        % Select Parent
        i=randi([1 nPop]);
        p=pop(i);
        
        % Apply Mutation
        popm(k).Position=Mutate(p.Position,mu,VarMin,VarMax);
        
        % Evaluate Mutant
        popm(k).Cost=CostFunction(popm(k).Position);
        
    end
    
    % Create Merged Population
    pop=[pop
         popc
         popm];
     
    % Sort Population
    Costs=[pop.Cost];
    [Costs, SortOrder]=sort(Costs);
    pop=pop(SortOrder);
    
    % Update Worst Cost
    WorstCost=max(WorstCost,pop(end).Cost);
    
    % Truncation
    pop=pop(1:nPop);
    Costs=Costs(1:nPop);
    
    % Store Best Solution Ever Found
    BestSol=pop(1);
    
    % Store Best Cost Ever Found
    BestCost(it)=BestSol.Cost;
    
    % Store NFE
    nfe(it)=NFE;
    
    % Show Iteration Information
    disp(['Iteration ' num2str(it) ': NFE = ' num2str(nfe(it)) ', Best Cost = ' num2str(BestCost(it))]);
    
end
toc
%% Results
epsilon = BestSol.Position (2)
C = BestSol.Position (1)
arg1 = BestSol.Position (3)
MAPE = BestSol.Cost

toc
figure;
semilogy(nfe,BestCost,'LineWidth',2);
xlabel('NFE');
ylabel('Cost');
