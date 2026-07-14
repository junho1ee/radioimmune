%
% Cancer research: Mathematical Modeling of Cancer Immunotherapy
%
% Cancer Immunology on SFRT: Figure generation
%
%The original Asperud approach is 
%(1) infiltration is triggered by radiation and infiltration is the most
%important 

%=========================================================================
%[1] Tumor Immunology parameter 
%
RS_T = [0.05, 0.05/4.4]; %Radiosensitivity of Tumor, [alpha, beta]
RS_L = [0.182, 0.143];   %Radiosensitivity of T-Cell/Dendrite Cells, [alpha, beta]
rho=0.1e0;   %T-cell production by live Tumor
Psi=1300;    %T-cell production by Radiation damaged Tumor cells
Psi=200;     %T-cell production by Radiation damaged Tumor cells, more reasonable
mu =log(2)/3.2;  %0.217 (Tpot=3.2d), 0.187 (3.7d), 0.03=Td of 23days, 0.05=13.9days
lambda_T = 1-exp(-log(2)/15);%Decay constant of doomed cell clearing [day], 15 for mice, 17 for human
lambda_DC= 1-exp(-log(2)/15);%Recovery constant (>15 days, maybe 30 day) of Dendrite cell;
lambda_Ln= 1-exp(-log(2)/15);%Decay constant (~15 day) of T-cell;
omega=0.05; %Primary immune trigger
gamma=0.01; %Secondary 0.009 (Fig2) 0 or 0.009 (Fig 3), 0.0314 (Fig 4), 0.128, 0.0314 (Fig 5)
r = 5;       %r is normalization factor, gamma/r is secondary immune effect
k = 0.0;   %Down regulate immune cell (primary immune effect)
%k = 1.60;   %Down regulate immune cell (primary immune effect)

%Simulation condition for Fig 4
Psi=300;    %T-cell production by Radiation damaged Tumor cells, more reasonable
rho=0.5;    %T-cell production by live Tumor
omega=0.135;%Primary immune trigger
gamma=0.00;  %Secondary =0
k = 1.1;    %Down regulate immune cell, 1.1 for fig 4,5,6 and 0.55 for fig 7



T_inf=lambda_Ln*mu/omega/rho
Dd_inf=lambda_Ln/lambda_T*(exp(mu) -1)*mu/omega/rho
alpha= ((1-mu)/2.5)^(2/3) *(1.5+mu)/2.5
eps_k= alpha*omega/mu/T_inf^(2/3)
max_T_inf = T_inf*(1/alpha*eps_k/(k+1e-9))^1.5
T_inf_delta = (9-22*alpha)/(9-25*alpha)*T_inf
%=========================================================================
%[2] Select directory of DVH
%
% DefaultPathName='../DVH';
% ButtonName = 'No';
% while strcmp( ButtonName, 'No') 
%     directoryname = uigetdir(DefaultPathName, 'Pick a DVH Directory');
%     if length(directoryname) == 1 %if cancel button is pressed
%         ButtonName = questdlg('Do you really want to quit?','','Yes','No','No');
%         if strcmp(ButtonName, 'Yes');  return; end
%     else
%         ButtonName = 'Yes';
%     end
% end
directoryname = 'H:\Research\Idea\Grid\DVH\17-10MLC_10X';

%=========================================================================
%[2-1] Read dDVH at depth of 5cm
%
files=dir([directoryname,'/*.dvh']);
if length(files)<1; display('There is no dvh files'); end
for i=1:length(files)
    tempname=files(i).name; %Filename is ???_DXX.dvh: XX=depth of tissue in cm
    tempname=tempname(end-6:end-4); %D04 eg.
    tempdepth=str2double(tempname(2:3));
    if isempty(tempdepth); continue; end
    if tempname(1)~='D'; continue; end
    %if tempdepth<1 || tempdepth>20; continue; end
    if tempdepth~=5; continue; end
    DVH=readDVH(files(i));
    Dose=DVH(:,1)/100; %cGy=>Gy
    dV  =DVH(:,2);     %cc
    
    Dose=Dose/Dose(find(dV>0,1,'last'));% Normalize to max dose
end

% %Read clinical Lattice plan
% [NUM,TXT,RAW]=xlsread('..\Patients\Miller, Sarah-86623579 1500cGy_GRID.csv');
% Dose=NUM(1:end-1,1);
% dV=-diff(NUM(:,3)); %1:Dose, 2:GTV, 3:CTV, 4:Sphere
% 


%=========================================================================
%[2-2] Initialize
%
days = 0:900;
day_SFRT=10; %treatment day of SFRT, 10 day after 0.03cc implant for Fig 4, 5,7, and 15days for Fig 6
dose = zeros(length(days),length(dV)); 
%dose(day_SFRT,:)=Dose'; %Dose from MIM file
%dose(day_SFRT,:)=10*Dose'; %maximum dose [Gy] x Grid Dose [normalized to 1Gy]
dose(day_SFRT+1,:)=10; %Uniform dose. SF is considered 1 day after radiation
% dose(day_SFRT+1,round(end*5/10):end)=0; %Uniform dose only to half of the volume
%dose(day_SFRT+0+(1:4),:)=15; %Uniform dose to everywhere [Gy]
%dose(day_SFRT+0+(1:4),round(end/2):end)=0; %Uniform dose to half the volume
dose2_T=RS_T(1)*dose+RS_T(2)*dose.^2;
dose_spread=1;
if dose_spread ==1
    weight=normpdf(days,3,1.5); weight(1)=1-sum(weight(2:end)); %mean 5 and variance 1.5 for mouse, (15,3) for human
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
Zs=zeros(length(days),1);
DC=ones(length(days),length(dV));  %Dendrite cell
D =zeros(length(days),length(dV)); %Doomed cell

%T(1,:)=dV'; % Volume from MIM file
%T(1,:)=dV'/sum(dV)*20; % starting volume in cc at day 0
T(1,:)=dV'/sum(dV)*0.03; % starting volume in cc at day 0
%Immunotherapy 
%p1: concentration of anti PD1 drug 
%c4: concentration of anti CTLA4 drug 0, 150
c4=0;
p1=zeros(length(days),1); %assuming concentration is homogeneous
%p1(day_SFRT+(1:60))=2.1;





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
    
    %assuming very slow decay, it was mentioned but simplified to be
    %constant in the paper
    %Zs(i) = 0.99*Zs(day)+gamma*(1+c4)/(r+c4)*(Zp(i));
    
    %%pharmacokinetics: assumption: 100% decay/day
    %p1(i)=0.0*p1(day)+p1(i);

    waitbar(day/days(end),h)
end
close(h)

figure(2)
set(gca,'FontSize',13, 'FontName','Arial')
set(gcf,'Position',[926 410 831 425])
%[3-1] Rejection probability
%mu_m = mu/2.5 %Metastatic growth rate was assumped smaller than primary growth rate
%Sigma=1.5 for Fig 2: Standard deviation of 1.5 days for mouse and 3 for human
%y=1/2*(1+erf((log(Zp+Zs)-log(mu/2.5))/sqrt(2)/3)); 
%subplot(131), plot(days, Zp+Zs, days, Zp, 'o', days,Zs,'x', days, sum(Ln,2)/1000, days, sum(A,2)/1000); xlim([0 350]),axis([0 150 0 0.2]), legend('Zp+Zs','Zp','Zs','Ln','A'); grid;
%subplot(141), plot(days-day_SFRT, Zp+Zs, days-day_SFRT, Zp, 'o', days-day_SFRT,Zs,'x', days-day_SFRT,omega*sum(Ln,2)); xlim([0 350]),axis([0 100 0 50]), legend('Zp+Zs','Zp','Zs'); grid;
subplot(131),set(gca,'FontSize',12, 'FontName','Arial'), hold on
    plot(days-day_SFRT, Zp, 'k', days,days*0+mu,'k:', 'linewidth',2); xlim([0 350]),axis([0 30 0 1]), 
    %legend('Z','\mu ');
    xlabel('Days (n)'); ylabel('Immune effect, Z_n'); grid on;
subplot(132),set(gca,'FontSize',12, 'FontName','Arial'), hold on
    plot(days-day_SFRT, sum(T(1,:))*exp(mu*days),'k:',days-day_SFRT,sum(D,2)+sum(T,2),'k', 'linewidth',2)
    %plot(days,days*0+Dd_inf+T_inf,'k--', 'linewidth',1,'Color', 0.2*[1 1 1]), 
    legend('e^\mu ','Tumor volume'); 
    axis([0 30 0 2+0*round(19*max(Dd_inf))/10]); xlabel('Days (n)'); ylabel('Tumor volume (T_n + D_n) [cc]'); grid on
subplot(133),set(gca,'FontSize',12, 'FontName','Arial'), hold on
    plot(days-day_SFRT, sum(T(1,:))*exp(mu*days),'k:',days-day_SFRT,sum(D,2)+sum(T,2),'k', 'linewidth',2)
    %plot(days,days*0+Dd_inf+T_inf,'k--', 'linewidth',1,'Color', 0.2*[1 1 1]), 
    %legend('e^\mu ','Tumor volume'); 
    axis([0 150 0 2+0*round(19*max(Dd_inf))/10]); xlabel('Days (n)'); ylabel('Tumor volume (T_n + D_n) [cc]'); grid on
% subplot(122),set(gca,'FontSize',12), hold on
%     plot(days-day_SFRT, sum(T(1,:))*exp(mu*days),'k:',days-day_SFRT,sum(T,2),'k', 'linewidth',2)
%     plot(days, days*0+T_inf,'k--','linewidth',1,'Color', 0.2*[1 1 1]), 
%     axis([0 50 0 2+0*round(12*max(T_inf))/10]); xlabel('Days (n)'); ylabel('Live Tumor Volume, T_n [cc]'); grid 

  Anot1=annotation('textbox',[0.06,0.87,0.1,0.1],'String','(A)', 'LineStyle','none','FontSize',16);
  Anot2=annotation('textbox',[0.34,0.87,0.1,0.1],'String','(B)', 'LineStyle','none','FontSize',16);
  Anot3=annotation('textbox',[0.62,0.87,0.1,0.1],'String','(C)', 'LineStyle','none','FontSize',16);
    
    
    
Z=Zp+Zs;
CLn=sum(Ln,2);
AA=[Zp, Zs, sum(T,2)];



figure(3)
set(gca,'FontSize',12)
set(gcf,'Position',[926 410 831 425])
if ~isempty(find(dose(day_SFRT+1,:)==0,1)) && ~isempty(find(dose(day_SFRT+1,:)>0,1)); flag_SFRT=1; else; flag_SFRT=0; end
if flag_SFRT==1
    subplot(121),set(gca,'FontSize',13, 'FontName','Arial'), hold on
        plot(days-day_SFRT,sum(D,2)+sum(T,2),'k', 'linewidth',2)
        title('(A) 50% irradiation')
        axis([0 150 0 2+0*round(19*max(Dd_inf))/10]); xlabel('Days (n)'); ylabel('Tumor volume [cc]'); grid on
end
if flag_SFRT==0
    subplot(122),set(gca,'FontSize',13, 'FontName','Arial'), hold on
        plot(days-day_SFRT,sum(D,2)+sum(T,2),'k', 'linewidth',2)
        title('(B) 100% irradiation')
        axis([0 150 0 2+0*round(19*max(Dd_inf))/10]); xlabel('Days (n)'); ylabel('Tumor volume [cc]'); grid on
end

%legend('10Gy','15Gy','20Gy')








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
