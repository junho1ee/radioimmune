%
% Cancer research: Mathematical Modeling of Cancer Immunotherapy
%
% Cancer Immunology on SFRT
% Simulation on Jake's Patient (Miller Sarah)
%


%=========================================================================
%[1] Tumor Immunology parameter 
%
RS_T = [0.214 0.0214]; %Radiosensitivity of Tumor, [alpha, beta]
RS_L = [0.182, 0.143];   %Radiosensitivity of T-Cell/Dendrite Cells, [alpha, beta]
rho=0.10;    %T-cell production by live Tumor
Psi=4.60;    %T-cell production by Radiation damaged Tumor cells
mu =0.03;   %0.217 (Tpot=3.2d), 0.187 (3.7d), 0.03=Td of 23days, 0.05=13.9days
lambda_T = 1-exp(-log(2)/15);  %Decay constant of doomed cell clearing [day], 15 for mice, 17 for human
lambda_DC= 1-exp(-log(2)/15.0);%Recovery constant (>15 days, maybe 30 day) of Dendrite cell;
lambda_Ln= 1-exp(-log(2)/12.0);%Decay constant (~15 day) of T-cell;
omega=0.0040; %Primary immune trigger
gamma=0.0000; %Secondary 0.009 (Fig2) 0 or 0.009 (Fig 3), 0.0314 (Fig 4), 0.128, 0.0314 (Fig 5)
%gamma=0.0009; %Secondary 0.009 (Fig2) 0 or 0.009 (Fig 3), 0.0314 (Fig 4),
%0.128, 0.0314 (Fig 5), It is better but paper assumed no secondary.
r = 5;        %r is normalization factor, gamma/r is secondary immune effect
k = 0.50;     %Down regulate immune cell (primary immune effect)

% %if less down regulation is assumed.=> Volume effect really help to fit
% the data
% rho=0.054;    %T-cell production by live Tumor
% Psi=1.20;    %T-cell production by Radiation damaged Tumor cells
% k = 0.1;     %Down regulate immune cell (primary immune effect)

% % %if no down regulation is assumed.=> Volume effect really help to fit
% % the data
% rho=0.002;    %T-cell production by live Tumor
% Psi=6.40;     %T-cell production by Radiation damaged Tumor cells
% omega=0.0010; %Primary immune trigger
% k = 0.000;     %Down regulate immune cell (primary immune effect)

eps_k=omega/mu*(rho*omega/lambda_Ln/mu)^(2/3);
T_inf=lambda_Ln*mu/omega/rho
Dd_inf=lambda_Ln/(lambda_T)*(exp(mu) -1)*mu/omega/rho
alpha= ((1-mu)/2.5)^(2/3) *(1.5+mu)/2.5
eps_k= alpha*omega/mu/T_inf^(2/3)
max_T_inf = T_inf*(1/alpha*eps_k/(k+1e-9))^1.5
T_inf_delta = (9-22*alpha)/(9-25*alpha)*T_inf

%=========================================================================
%[2] Select directory of DVH
%
%Volume at day 0 = 20cc, day_Dx = GTV volume, day_Dx=round(log(GTV/20)/mu)
flag_GRID=1; %1: include GRID dose, 0: neglect GRID dose
DoseScale=1; %by default GRID blk dose scale=1. If blk = 15Gy and Lattice=20Gy, then it is 20/15
directoryname = 'H:\Research\Idea\Grid\DVH_LATTICE\10 Miller SIB'; fx=-4+[0:3]; fxdose=4;
%directoryname = 'H:\Research\Idea\Grid\DVH_LATTICE\10 Miller Blk'; fx=-4+[0:4]; fxdose=4;
%directoryname = 'H:\Research\Idea\Grid\DVH_LATTICE\10 Miller Lattice'; fx=-4+[0:4]; fxdose=4; fxdose=4;fx=[1:12]; fxdose=2; DoseScale=1;

%=========================================================================
%[2-1] Read dDVH 
%

files=dir([directoryname,'/*.dvh']);
if length(files)<1; display('There is no dvh files'); end
for i=1:length(files)
    tempname=files(i).name; 
    if contains(tempname,'GTV')
        DVH=readDVH(files(i));
        Dose=DoseScale*DVH(:,1)/100; %cGy=>Gy
        dV  =DVH(:,2);     %cc
    end
end
%sprintf('Target Volume at the time of Sim: %5.1f [cc]',sum(dV))
%day_Dx=round(log(sum(dV)/20)/mu)

%=========================================================================
%[2-2] Initialize
%
days = 0:2000;
dose = zeros(length(days),length(dV)); 
Tx=0;
if Tx==1 %First Tx
    day_Dx=95; %day of CT acquisition
    day_Tx=day_Dx+12; %day of first treatment.
    dose(day_Tx+[0,1,2,5,6],:)=6; % [Tx1] Uniform dose to everywhere [Gy]
elseif Tx==2 %2nd Tx
    day_Dx=66; %day of CT acquisition
    day_Tx=day_Dx+10; %day of first treatment.
    dose(day_Tx+[0,2,4,7,9],:)=6; % [Tx2] Uniform dose to everywhere [Gy]
elseif Tx==3 %3rd Tx
    day_Dx=94; %day of CT acquisition
    day_Tx=day_Dx+18; %day of first treatment.
    dose(day_Tx+[0,1,2,3],:)=4; % [Tx3] Uniform dose to everywhere [Gy]
    dose(day_Tx-1,:)=Dose'; % [Tx3] Dose from MIM file, SIB plan (4Gy+15Gy SFRT)
elseif Tx==0 %All treatment
    day_Dx=95; %day of CT acquisition after 20cc tumor volume, 2019-5-31
    day_Tx1=day_Dx+12; %day of first treatment.
    dose(day_Tx1+[0,1,2,5,6],:)=6; % [Tx1] Uniform dose to everywhere [Gy]
    day_Tx2=day_Dx+datenum(2021,2,8)-datenum(2019,5,31);
    dose(day_Tx2+[0,2,4,7,9],:)=6; % [Tx2] Uniform dose to everywhere [Gy]
    day_Tx3=day_Dx+datenum(2022,8,9)-datenum(2019,5,31);
    dose(day_Tx3+[0,1,2,3,4],:)=4; % [Tx3] Uniform dose to everywhere [Gy]
    dose(day_Tx3-1,:)=Dose'; % [Tx3] Dose from MIM file, SIB plan (4Gy+15Gy SFRT)
end
dose2_T=RS_T(1)*dose+RS_T(2)*dose.^2;
dose_spread=1;
if dose_spread ==1
    weight=normpdf(days,15,3); weight(1)=1-sum(weight(2:end)); %mean 3 and variance 1.5 for mouse, (15,3) for human
    %weight=diff([0,logncdf(exp(days),5,1.5)]); %both method gives similar answer 
    Sn_T=ones(length(days),length(dV));
    for day=1:days(end)
        Sn_T(day:end,:)=Sn_T(day:end,:).*exp(-dose2_T(day,:).*(weight(1:end-day+1)'*ones(1,length(dV))));
    end
else
    Sn_T=exp(-dose2_T);
end
%Dendrite cell and T cell radiation sensitiviy is high and considered
%happending mostly in the same day
Sn_L=exp( -(RS_L(1)*dose+RS_L(2)*dose.^2) );


T =zeros(length(days),length(dV));
Ln=zeros(length(days),length(dV));
eps=zeros(length(days),1);
Zp=zeros(length(days),1);
Zs=zeros(length(days),1)+0.0;
DC=ones(length(days),length(dV));  %Dendrite cell
D =zeros(length(days),length(dV)); %Doomed cell

%T(1,:)=dV'; % Volume from MIM file
T(1,:)=dV'/sum(dV)*20; % starting volume in cc at day 0
%Immunotherapy 
%p1: concentration of anti PD1 drug 
%c4: concentration of anti CTLA4 drug 0, 150
c4=0;
p1=zeros(length(days),1); %assuming concentration is homogeneous
%p1(day_Dx+(1:60))=2.1;





%=========================================================================
%[3] Modeling
%
%Sn_L: survival rate of Ln from radiation from k_day>=day
%Sn_T: survival rate of Tumor from radiation
s=1; %sensitivity of eps
h = waitbar(0,'Please wait...');

for day=1:days(end)
    i=day+1;
    Zmax=Zp(day)+Zs(day); 
    
    T(i,:)  = T(day,:).*Sn_T(day,:)*exp(mu-Zmax);
    eps(i)  = 0.999*tanh(s* dot ((1-Sn_T(day,:)),T(day,:))/(sum(T(day,:)+D(day,:))) ); %0.999 (to make eps <1)
    DC(i,:) = ( Sn_L(day,:)./Sn_T(day,:).*DC(day,:) + (1-DC(day,:)).*lambda_DC ).* (1-eps(i));%Density of Dendrite cells, DC
    Ln(i,:) = (1-lambda_Ln)*Sn_L(day,:).*Ln(day,:)+rho*T(i,:)+Psi*eps(i).*DC(i,:).*T(i,:);
    
    %Zp, Zs are determined by 
    Zp(i) = omega*sum(Ln(i,:),2)/(1+k*(sum(T(i,:),2)^(2/3))*sum(Ln(i,:),2)/(1+p1(i)));
    Zs(i) = Zs(day)+gamma*(1+c4)/(r+c4)*Zp(i);
    D(i,:)  = (1-lambda_T)*D(day,:)+(1-Sn_T(day,:)).*T(day,:)+Sn_T(day,:).*T(day,:)*exp(mu)*(1-exp(-Zmax));
    
    waitbar(day/days(end),h)
end
close(h)



x=[0, 609, 1148];
y=[343,137,323.6];

figure(1)
set(gca,'FontSize',13, 'FontName','Arial')
set(gcf,'Position',[626 410 831 425])

subplot(131), 
        %plot(days-day_Dx, Zp+Zs, days-day_Dx, Zp, 'o', days-day_Dx,Zs,'x', 'LineWidth',2); axis([0 1800 0 0.1]), legend('Zp+Zs','Zp','Zs'); grid;
        plot(days-day_Dx, Zp+Zs, 'k-',days,days*0+mu,'k:', 'LineWidth',2); axis([0 1800 0 0.1]), 
        xlabel('Day'); ylabel('Immune effect, Z_n'); grid
        set(gca,'FontSize',13, 'FontName','Arial'), %hold on
subplot(132), 
        plot(days-day_Dx,sum(T,2),days-day_Dx,sum(T,2)+sum(D,2),'k', x,y,'ro','LineWidth',2), 
        handle_leg=legend('T_n','T_n+D_n', 'measurement'); handle_leg.FontAngle='italic';
        axis([0 200 0 900]); xlabel('Day'); ylabel('Tumor volume [cc]'); grid
        set(gca,'FontSize',13, 'FontName','Arial'), %hold on
subplot(133), 
        plot(days-day_Dx,sum(T,2),days-day_Dx,sum(T,2)+sum(D,2),'k', x,y,'ro', 'LineWidth',2),  
        handle_leg=legend('T_n','T_n+D_n','measurement'); handle_leg.FontAngle='italic';
        axis([0 1800 0 900]); xlabel('Day'); ylabel('Tumor volume [cc]'); grid
        set(gca,'FontSize',13, 'FontName','Arial'), %hold on


figure(2)
set(gca,'FontSize',13, 'FontName','Arial')
set(gcf,'Position',[926   145   684   690])
subplot(311), 
        %plot(days-day_Dx, Zp+Zs, days-day_Dx, Zp, 'o', days-day_Dx,Zs,'x', 'LineWidth',2); axis([0 1800 0 0.1]), legend('Zp+Zs','Zp','Zs'); grid;
        plot(days-day_Dx, Zp+Zs, 'k-',days,days*0+mu,'k:', 'LineWidth',2); axis([0 1400 0 0.2]), 
        handle_leg=legend('Z_n','\mu', 'measurement'); handle_leg.FontAngle='italic';
        xlabel('Day'); ylabel('Immune effect, Z_n'); %grid
        set(gca,'FontSize',13, 'FontName','Arial'), %hold on
subplot(312), 
        plot(days-day_Dx,sum(T,2),days-day_Dx,sum(T,2)+sum(D,2),'k', x,y,'ro','LineWidth',2), 
        handle_leg=legend('T_n','T_n+D_n', 'measurement'); handle_leg.FontAngle='italic';
        axis([0 200 0 900]); xlabel('Day'); ylabel('Tumor volume [cc]'); %grid
        set(gca,'FontSize',13, 'FontName','Arial'), %hold on
subplot(313), 
        plot(days-day_Dx,sum(T,2),days-day_Dx,sum(T,2)+sum(D,2),'k', x,y,'ro', 'LineWidth',2),  
        %handle_leg=legend('T_n','T_n+D_n','measurement'); handle_leg.FontAngle='italic';
        axis([0 1400 0 900]); xlabel('Day'); ylabel('Tumor volume [cc]'); %grid
        set(gca,'FontSize',13, 'FontName','Arial'), %hold on
        
Anot1=annotation('textbox',[0.09,0.89,0.1,0.1],'String','(A)', 'LineStyle','none','FontSize',20);
Anot2=annotation('textbox',[0.09,0.59,0.1,0.1],'String','(B)', 'LineStyle','none','FontSize',20);
Anot3=annotation('textbox',[0.09,0.29,0.1,0.1],'String','(C)', 'LineStyle','none','FontSize',20);











%=========================================================================
%  Subroutines
%
%Read Pinnacle dDVH file
%This file is exported using Naichang's Conformity Check.
%DVH files are found at /home/p3rtp/Export/DVH/{MRN}

function dvh=readDVH(file)
dvh=[];
fid=fopen([file.folder,'\',file.name]);
    NumberOfPoints=0;
    tline = fgetl(fid);
    while ischar(tline)
        [T,R] = strtok(tline,'=');
        tline=fgetl(fid);
        if strcmp(T,'NumberOfPoints '); NumberOfPoints=str2double(R(2:end-1)); continue; end
        if strcmp(T,'Points[] ')
            for i=1:NumberOfPoints
                dvh(end+1,:)=sscanf(tline,'%f,%f');
                tline=fgetl(fid);
            end
            break;
        end
    end
fclose(fid);

end
