% Demo for paper "Saliency Detection via Graph-Based Manifold Ranking" 
% by Chuan Yang, Lihe Zhang, Huchuan Lu, Ming-Hsuan Yang, and Xiang Ruan
% To appear in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2013), Portland, June, 2013.


addpath('./others/');
addpath('./slic/');       %���г����طָ�ĺ���
%%------------------------set parameters---------------------%%
theta=10; % control the edge weight  ���Ʊ�ԵȨ��
alpha=0.01;% control the balance of two items in manifold ranking cost function  ������������ɱ������������ƽ��
% spnumber=3000;% superpixel number  ����������
imgRoot='./test/';% test image path
saldir='./saliencymap/';% the output path of the saliency map
supdir='./superpixels/';% the superpixel label file path
foredir='./foreground/';%the foreground map
mkdir(supdir);  %ָ�����ļ����´����µ��ļ���
mkdir(saldir);
mkdir(foredir);
% imnames=dir([imgRoot '*' 'jpg']);  %�г���ǰĿ¼�·��ϵ��ļ��к��ļ�

%% ��ȡ���ݲ���������Ȳ���ͼ
fid=fopen('C:\Users\dell\Desktop\Cd\20120111', 'r');
[data,~]=fread(fid,[2240*2,2420],'float'); 
a1=data(1:2:end,:);
b1=data(2:2:end,:);
c1=a1+i*b1;
amli=abs(c1);

fid=fopen('C:\Users\dell\Desktop\Cd\20130119', 'r');
[data2,count]=fread(fid,[2240*2,2420],'float'); 
a2=data2(1:2:end,:);
b2=data2(2:2:end,:);
c2=a2+i*b2;
amli2=abs(c2);

w = fspecial('gaussian',[9,9],5);
amli = imfilter(amli,w,'replicate'); %��˹�˲�
amli2 = imfilter(amli2,w,'replicate'); %��˹�˲�
[r,c]=size(amli);
 div_log=zeros(r,c);  %��ǰ����ռ�

 for i=1:r
    for k=1:c
        div_log(i,k)=abs(log(amli2(i,k)+1)-log(amli(i,k)+1));   %��ͼ�������б�������
    end
 end
% div_log_min=min(min(div_log),[],2);
% div_log_max=max(max(div_log),[],2);
 div_mean=3*mean(mean(div_log));
 for i=1:r
    for k=1:c
%         div_log(i,k)=(div_log(i,k)-div_log_min)/(div_log_max-div_log_min);    %���й�һ��
        if div_log(i,k)<div_mean
            div_log(i,k)=0;
        end
    end
 end
div_log=medfilt2(div_log,[11,11]);%��ֵ�˲�
figure
imshow(div_log);
%% ���г����طָ�
%  [D, Am, C,h] = slic(A, 2500,10, 1.5);
 [superpixels] = slic(div_log, 1500,10);     
 I=drawregionboundaries(superpixels, div_log, [255 0 0]);
 figure
 imshow(I);
%%
% for ii=1:length(imnames)   
%     disp(ii);
%     imname=[imgRoot imnames(ii).name]; 
%     div_log=imread(imname);
%     [input_im,w]=removeframe(imname);% run a pre-processing to remove the image frame ����Ԥ������ɾ��ͼ��߿�
    [m,n] = size(div_log);
    w=[m,n,1,m,1,n];
    
    
%%----------------------generate superpixels--------------------%%
%     imname=[imname(1:end-4) '.bmp'];% the slic software support only the '.bmp' image
%     comm=['SLICSuperpixelSegmentation' ' ' imname ' ' int2str(20) ' ' int2str(spnumber) ' ' supdir];
%     system(comm);    
%     spname=[supdir imnames(ii).name(1:end-4)  '.dat'];
%     superpixels=ReadDAT([m,n],spname); % superpixels���ǳ����طָ�Ľ����һ��label�������ִ�������صı�ǩ�������
%     [superpixels] = slic(div_log, 1500,30);     
%     I=drawregionboundaries(superpixels, div_log, [255 0 0]);
%     imshow(I);

%%----------------------design the graph model--------------------------%%
% compute the feature (mean color in lab color space)  ��������(ʵ������ɫ�ռ��е�ƽ����ɫ)
% for each node (superpixels)
    spnum=max(superpixels(:));% the actual superpixel number  ʵ�ʵĳ���������
    input_vals=reshape(div_log, m*n, 1); %��input_im���г�m*n�У�k�еľ���
    gray_vals=zeros(spnum,1);   %spnum�ǳ���������
    inds=cell(spnum,1);  %����Ԫ��
    for i=1:spnum
        inds{i}=find(superpixels==i);  %�洢��ͬ��ǩ�����ص�λ��
        gray_vals(i,1)=mean(input_vals(inds{i},:),1);
    end  
    seg_vals=gray_vals;
%    lab_vals = colorspace('Lab<-', gray_vals);  %��rgbתΪLab
%     seg_vals=reshape(lab_vals,spnum,3);% feature for each superpixel
 
 % get edges
    adjloop=AdjcProcloop(superpixels,spnum);
    edges=[];
    for i=1:spnum
        indext=[];
        ind=find(adjloop(i,:)==1);   %�ҵ����i���������ڽӵ���Χ�����أ�����λ��
        for j=1:length(ind)
            indj=find(adjloop(ind(j),:)==1);  %�ڽӵ��ڽӵ�����
            indext=[indext,indj];
        end
        indext=[indext,ind];
        indext=indext((indext>i));   %��֤������ͼ�����ظ�
        indext=unique(indext);
        if(~isempty(indext))
            ed=ones(length(indext),2);
            ed(:,2)=i*ed(:,2);
            ed(:,1)=indext;
            edges=[edges;ed];      %����һ������ߣ��洢���˽ڵ�
        end
    end

% compute affinity matrix  �����������W  ϡ���ڽ�
    weights = makeweights(edges,seg_vals,theta);  %theta���Ʊ�ԵȨ�أ�seg_vals��ÿ�������ص�Lab��Ϣ������ÿ������ߵ�Ȩ��
    W = adjacency(edges,weights,spnum);

% learn the optimal affinity matrix (eq. 3 in paper) ѧϰ�����׺;���
    dd = sum(W);                                  %����������һ������
    D = sparse(1:spnum,1:spnum,dd); clear dd;     %�Ⱦ���W�������
    optAff =(D-alpha*W)\eye(spnum);               %alpha������������ɱ������������ƽ�⣬eye(spnum)���ص�λ����
    mz=diag(ones(spnum,1));                      
    mz=~mz;
    optAff=optAff.*mz;                            % (D-alpha*W)-1����ͬһ��ͼ��ֻ��Ҫ��һ��
 
%%-----------------------------stage 1--------------------------%%
% compute the saliency value for each superpixel  ����ÿ�������ص�������ֵ
% with the top boundary as the query ʹ�ö����߽���Ϊ��ѯ
    Yt=zeros(spnum,1);
    bst=unique(superpixels(1,1:n));    %��һ����Ϊ�����߽�
    Yt(bst)=1;
    bsalt=optAff*Yt;                   %f=(D-alpha*W)-1*y
    bsalt=(bsalt-min(bsalt(:)))/(max(bsalt(:))-min(bsalt(:))); %��һ��
    bsalt=1-bsalt;                     %��������

% down
    Yd=zeros(spnum,1);
    bsd=unique(superpixels(m,1:n));
    Yd(bsd)=1;
    bsald=optAff*Yd;
    bsald=(bsald-min(bsald(:)))/(max(bsald(:))-min(bsald(:)));
    bsald=1-bsald;
   
% right
    Yr=zeros(spnum,1);
    bsr=unique(superpixels(1:m,1));
    Yr(bsr)=1;
    bsalr=optAff*Yr;
    bsalr=(bsalr-min(bsalr(:)))/(max(bsalr(:))-min(bsalr(:)));
    bsalr=1-bsalr;
  
% left
    Yl=zeros(spnum,1);
    bsl=unique(superpixels(1:m,n));
    Yl(bsl)=1;
    bsall=optAff*Yl;
    bsall=(bsall-min(bsall(:)))/(max(bsall(:))-min(bsall(:)));
    bsall=1-bsall;   
   
% combine 
    bsalc=(bsalt.*bsald.*bsall.*bsalr);
    bsalc=(bsalc-min(bsalc(:)))/(max(bsalc(:))-min(bsalc(:)));
    
% assign the saliency value to each pixel     Ϊÿ��������ͬ��ǩ�����ط���������ֵ
     tmapstage1=zeros(m,n);
     for i=1:spnum
        tmapstage1(inds{i})=bsalc(i);
     end
     tmapstage1=(tmapstage1-min(tmapstage1(:)))/(max(tmapstage1(:))-min(tmapstage1(:)));
     
     mapstage1=zeros(w(1),w(2));                   %w(1),w(2)=m,n
     mapstage1(w(3):w(4),w(5):w(6))=tmapstage1;    %�����һ�׶�������ͼ
     mapstage1=uint8(mapstage1*255);  

     outname=[saldir imnames(ii).name(1:end-4) '_stage1' '.png'];
     imwrite(mapstage1,outname);

%    bsalc=seg_vals(:,1);         %���������ص�������Ϣ
%     size_b=size(bsalc,1);
%      min1=min(bsalc);
%     max1=max(bsalc);
%    for i=1:size_b
%        bsalc(i,1)=(bsalc(i,1)-min1)/(max1-min1);
%    end
     
%%----------------------stage2-------------------------%%
% binary with an adaptive threhold (i.e. mean of the saliency map) ��������Ӧ�����ԵĶ�����(��������ӳ��ľ�ֵ)
    th=mean(bsalc);
    bsalc(bsalc<th)=0;
    bsalc(bsalc>=th)=1;%ǰ���㣬��ƽ��ֵΪ��ֵ�����ж�ֵ��
    
% compute the saliency value for each superpixel ����ÿ�������ص�������ֵ
    fsal=optAff*bsalc;    
    
% assign the saliency value to each pixel   ͬ��
    tmapstage2=zeros(m,n);
    for i=1:spnum
        tmapstage2(inds{i})=fsal(i);    
    end
    tmapstage2=(tmapstage2-min(tmapstage2(:)))/(max(tmapstage2(:))-min(tmapstage2(:)));

mapstage2=zeros(w(1),w(2));
    mapstage2(w(3):w(4),w(5):w(6))=tmapstage2;
     gray=rgb2gray(div_log);      %ת�ɻҶ�ͼ
     mapstage2_mean=mean(mapstage2(:));
     for i=1:w(1)
         for j=1:w(2)
             if(mapstage2(i,j)>=mapstage2_mean)
                 mapstage2(i,j)=1;
             else
                 mapstage2(i,j)=0;
             end
         end
     end
     mapstage2=mapstage2.*double(gray);
    figure(2);
     imshow(uint8(3*mapstage2));
%     outname=[saldir imnames(ii).name(1:end-4) '_stage2' '.png'];   
%     imwrite(mapstage2,outname);
%     
%foreground    
    J=superpixels;
idx=find(bsalc);    %����ǰ�����λ�û��߱�ǩ
[c,d]=size(idx);
for i=1:c
    for a=1:m       %m��nΪԭͼ��������
        for b=1:n
            if J(a,b)==idx(i)
                J(a,b)=NaN;
            else
                J(a,b)=J(a,b);
            end
        end
    end
end
J(~isnan(J))=0;   
J(isnan(J))=1;
outname=[foredir imnames(ii).name(1:end-4) '_foreground' '.png'];   
imwrite(J,outname);

    
% end

       
        


