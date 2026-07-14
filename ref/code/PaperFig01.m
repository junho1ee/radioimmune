%
% Cancer research: Mathematical Modeling of Cancer Immunotherapy
%
% Cancer Immunology on SFRT: Figure generation
%


%=========================================================================
%[1] Tumor Immunology parameter 
%
RS_T = [0.05, 0.05/4.4]; %Radiosensitivity of Tumor, [alpha, beta]
RS_L = [0.182, 0.143];   %Radiosensitivity of T-Cell/Dendrite Cells, [alpha, beta]
Psi=1300;    %T-cell production by Radiation damaged Tumor cells
Psi=300;     %T-cell production by Radiation damaged Tumor cells, more reasonable
mu = 0.217;  %log(2)/3.2;  %0.217 (Tpot=3.2d), 0.187 (3.7d), 0.03=Td of 23days, 0.05=13.9days
gamma=0.0;   %Secondary =0
r = 5;       %r is normalization factor, gamma/r is secondary immune effect

fig_flg=3;% figure 1, 2 or 3
if fig_flg==1 || fig_flg==2
    %Simulation condition for Fig 1 & 2
    lambda_T = 1-exp(-log(2)/17.5);%Decay constant of doomed cell clearing [day], 15 for mice, 17 for human
    lambda_DC= 1-exp(-log(2)/17.5);%Recovery constant (>15 days, maybe 30 day) of Dendrite cell;
    lambda_Ln= 1-exp(-log(2)/1.7);%Decay constant (~15 day) of T-cell;
    rho=0.1;    %T-cell production by live Tumor
    omega=0.05; %Primary immune trigger
    k = 0.013;   %Down regulate immune cell, 0 (fig 1),0.010, 0.012, 0.013 (fig 2)
else
    %Simulation condition for Fig 3
    lambda_T = 1-exp(-log(2)/15);%Decay constant of doomed cell clearing [day], 15 for mice, 17 for human
    lambda_DC= 1-exp(-log(2)/15);%Recovery constant (>15 days, maybe 30 day) of Dendrite cell;
    lambda_Ln= 1-exp(-log(2)/15);%Decay constant (~15 day) of T-cell;
    rho=0.5;    %T-cell production by live Tumor
    omega=0.135;%Primary immune trigger
    k = 0.50083;    %Down regulate immune cell
end 
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
day_SFRT=10; %treatment day of SFRT
dose = zeros(length(days),length(dV)); 
dose(day_SFRT,:)=Dose'; %Dose from MIM file
%dose(day_SFRT,:)=10*Dose'; %maximum dose [Gy] x Grid Dose [normalized to 1Gy]
% dose(day_SFRT,:)=15; %Uniform dose
% dose(day_SFRT,round(end*5/10):end)=0; %Uniform dose to half the volume
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
subplot(121),set(gca,'FontSize',13,'FontName','Arial'), hold on
    plot(days-day_SFRT, Zp, 'k', days,days*0+mu,'k:', 'linewidth',2); xlim([0 350]),axis([0 200 0 1]), 
    xlabel('Days (n)'); ylabel('Immune effect, Z_n'); grid on;
if fig_flg==1
    %This is for figure 1 
    subplot(122),set(gca,'FontSize',13,'FontName','Arial'), hold on
        plot(days-day_SFRT, sum(T(1,:))*exp(mu*days),'k:',days-day_SFRT,sum(T,2),'k',days-day_SFRT,sum(D,2),'k-.', 'linewidth',2)
        plot(days, days*0+T_inf,'k--',days,days*0+Dd_inf,'k--', 'linewidth',1,'Color', 0.2*[1 1 1]), 
        legend('e^\mu ','Viable tumor, T_n','Doomed cells, D_n'); axis([0 200 0 round(19*max(Dd_inf))/10]); xlabel('Days (n)'); ylabel('Volume of T_n and D_n [cc]'); grid on
        Anot1=annotation('textbox',[0.05,0.85,0.1,0.1],'String','(A)', 'LineStyle','none','FontSize',16);
        Anot2=annotation('textbox',[0.48,0.85,0.1,0.1],'String','(B)', 'LineStyle','none','FontSize',16);
elseif fig_flg==2 || fig_flg==3
    %This is for figure 3 
    subplot(122),set(gca,'FontSize',13,'FontName','Arial'); hold on
        plot(days-day_SFRT, sum(T(1,:))*exp(mu*days),'k:',days-day_SFRT,sum(T,2),'k', 'linewidth',2)
        plot(days, days*0+T_inf,'k--','linewidth',1,'Color', 0.2*[1 1 1]); 
        axis([0 200 0 round(19*max(Dd_inf))/10]); xlabel('Days (n)'); ylabel('Live Tumor Volume, T_n [cc]'); grid on
        Anot1=annotation('textbox',[0.05,0.85,0.1,0.1],'String','(A)', 'LineStyle','none','FontSize',16);
        Anot2=annotation('textbox',[0.48,0.85,0.1,0.1],'String','(B)', 'LineStyle','none','FontSize',16);
        cp1=findobj('Parent',gca,'Type','Line');
        legend([cp1(2), cp1(5), cp1(8)],'\kappa =0.010','\kappa =0.012','\kappa =0.013')
end
    
    
    
Z=Zp+Zs;
CLn=sum(Ln,2);
AA=[Zp, Zs, sum(T,2)];










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
